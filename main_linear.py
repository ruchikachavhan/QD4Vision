import argparse
import builtins
from codecs import namereplace_errors
import math
from mimetypes import init
from modulefinder import Module
import os
import random
import shutil
import numpy as np
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
import models
from timm.models.convnext import convnext_base
from timm.models.resnet import resnet50
from  downstream_utils import get_dataset, get_train_transform, get_val_transform
import wandb
from downstream_utils import *
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import Ridge as LinReg
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))


model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base', 'convnext'] + torchvision_model_names

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', default=0.001, type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--baseline', action='store_true', help="Use resnet or hyper-resnet")
parser.add_argument('--test_dataset', default='cifar10', type=str)
parser.add_argument('--data_root', default='/raid/s2265822/TestDatasets/', type = str)
parser.add_argument('--num_encoders', default=5, type=int, help='Number of encoders')

parser.add_argument('--train-resizing', type=str, default='default', help='resize mode during training')
parser.add_argument('--val-resizing', type=str, default='default', help='resize mode during validation')
parser.add_argument('--no-hflip', action='store_true', help='no random horizontal flipping during training')
parser.add_argument('--color-jitter', action='store_true', help='apply jitter during training')
parser.add_argument('-sr', '--sample-rate', default=100, type=int,
                        metavar='N',
                        help='sample rate of training dataset (default: 100)')
# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--image_size', default=96, type=int,
                    help='image size')

parser.add_argument('--few_shot_reg', default=None, type=float,
                    help='image size')
parser.add_argument('-sc', '--num-samples-per-classes', default=None, type=int,
                        help='number of samples per classes.')

def main():
    main_worker()

def main_worker(config=None):
    args = parser.parse_args()
    global best_acc1

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    # Model 
    if args.baseline:
        if args.arch == 'resnet50':
            models_ensemble = resnet50(pretrained=True, num_classes=1000)
            # models_ensemble = models.BaselineResNet(num_classes = 1000, arch = args.arch)
        elif args.arch == 'convnext':
            models_ensemble = convnext_base(pretrained=True, num_classes=1000)
    else:
        if args.arch == 'resnet50':
            models_ensemble = models.BranchedResNet(N = args.num_encoders, num_classes = 1000, arch = args.arch)
        elif args.arch == 'convnext':
            models_ensemble = models.DiverseConvNext(N = args.num_encoders, num_classes = 1000)

    # optionally resume from a checkpoint, only for ensemble model
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            if args.gpu is None:
                checkpoint = torch.load(args.pretrained)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.pretrained, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']
            # if not args.baseline:
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            print(state_dict.keys())
            models_ensemble.load_state_dict(state_dict, strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pretrained, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # Change model fc layers for number of Classes
    feat_dim = 2048 if args.arch == 'resnet50' else 512
    if args.test_dataset in ['imagenet-r', 'imagenet-a']:
        models_ensemble.num_classes = 200
    elif args.test_dataset == 'imagenet-sketch':
        models_ensemble.num_classes = 1000

    if args.baseline:
        if args.arch == 'resnet50':
            models_ensemble.feat_dim = 2048 if args.arch == 'resnet50' else 512
            # if dataset_info[args.test_dataset]['mode'] == 'pose_estimation':
                # models_ensemble.global_pool = nn.Identity()
            models_ensemble.fc = nn.Identity()
        elif args.arch == 'convnext':
            models_ensemble.head.fc = nn.Identity()
    else:
        if args.arch == 'resnet50':
            feat_dim = 2048 if args.arch == 'resnet50' else 512
            models_ensemble.base_model.branches_fc = nn.ModuleList([nn.Identity() for i in range(args.num_encoders)])
            # if dataset_info[args.test_dataset]['mode'] == 'pose_estimation':
                # models_ensemble.base_model.global_pool = nn.Identity()
        elif args.arch == 'convnext':
            for ind in range(args.num_encoders):
                models_ensemble.base_model.head[ind].fc = nn.Identity()
    
    # print(models_ensemble)
    # freeze all layers but the last fc
    for name, param in models_ensemble.named_parameters():
        # if args.few_shot_reg is None:
        if args.arch == 'resnet50':
            if "fc" not in name:
                param.requires_grad = False
        elif args.arch == 'convnext':
            if "head" not in name:
                param.requires_grad = False
        # print(name, param.requires_grad)

    # infer learning rate before changing batch size, not done in hyoer-models
    init_lr = args.lr 
    models_ensemble = models_ensemble.cuda(args.gpu)
        
    # train_images, train_labels, val_images, val_labels, test_images, test_labels, num_classes =
    train_loader, val_loader, trainval_loader, test_loader, num_classes = get_feature_datasets(args, models_ensemble)
    # print("Dataset sizes", train_images.shape[0], val_images.shape[0], test_images.shape[0])

    # train_images, train_labels, val_images, val_labels, test_images, test_labels = train_images.cpu(), train_labels.cpu(), val_images.cpu(), val_labels.cpu(), test_images.cpu(), test_labels.cpu()

    X_train_feature, y_train, X_val_feature, y_val = get_features(
        train_loader, val_loader, models_ensemble, device = args.gpu, baseline = args.baseline, test=False
    )

    print(X_train_feature.shape, X_val_feature.shape)

    X_trainval_feature, y_trainval, X_test_feature, y_test = get_features(
            trainval_loader, test_loader, models_ensemble, device = args.gpu, baseline = args.baseline, test=False
        )

    if args.baseline:
        if dataset_info[args.test_dataset]['mode'] == 'regression':
            clf = LinearRegression(2048, num_classes, 'r2')
        elif dataset_info[args.test_dataset]['mode'] == 'pose_estimation':
            clf = LinearRegression(2048, num_classes, 'pca')
        else:
            clf = LogisticRegression(2048, num_classes, 'accuracy')
    else:
        if dataset_info[args.test_dataset]['mode'] == 'regression':
            clf = [LinearRegression(2048, num_classes, 'r2') for _ in range(args.num_encoders)]
        elif dataset_info[args.test_dataset]['mode'] == 'pose_estimation':
            clf = [LinearRegression(2048, num_classes, 'pca') for _ in range(args.num_encoders)]
        else:
            clf = [LogisticRegression(2048, num_classes, 'accuracy') for _ in range(args.num_encoders)]

    if dataset_info[args.test_dataset]['mode'] in ['regression', 'pose_estimation']:
        wd_range = torch.logspace(-2, 3, 45)
    else:
        wd_range = torch.logspace(-6, 5, 45)

    for wd in tqdm(wd_range, desc='Selecting best hyperparameters'):
        if args.baseline:
            if dataset_info[args.test_dataset]['mode'] in ['regression', 'pose_estimation']:
                C = wd.item()
                clf.set_params({'alpha': C})
            else:
                C = 1. / wd.item()
                clf.set_params({'C': C})
            val_acc = clf.fit_regression(X_train_feature, y_train, X_val_feature, y_val)
            test_acc = clf.fit_regression(X_trainval_feature, y_trainval, X_test_feature, y_test)
        else:
            for k in range(args.num_encoders):
                if dataset_info[args.test_dataset]['mode'] in ['regression', 'pose_estimation']:
                    C = wd.item()
                    clf[k].set_params({'alpha': C})
                else:
                    C = 1. / wd.item()
                    clf[k].set_params({'C': C})
            val_acc = [clf[k].fit_regression(X_train_feature[k], y_train, X_val_feature[k], y_val) for k in range(args.num_encoders)]
            test_acc = [clf[k].fit_regression(X_trainval_feature[k], y_trainval, X_test_feature[k], y_test) for k in range(args.num_encoders)]
        print(C, val_acc, test_acc)

# Testing classes and functions
def get_features(train_loader, test_loader, model, device, baseline, test=True):
    X_train_feature, y_train = inference(train_loader, model, device, baseline, 'train')
    X_test_feature, y_test = inference(test_loader, model, device, baseline, 'test' if test else 'val')
    return X_train_feature, y_train, X_test_feature, y_test

def inference(loader, model, device, baseline, split):
    model.eval()
    feature_vector = []
    labels_vector = []
    for data in tqdm(loader, desc=f'Computing features for {split} set'):
        batch_x, batch_y = data
        batch_x = batch_x.cuda(device)
        labels_vector.extend(np.array(batch_y))
        if baseline:
            features = model(batch_x).view(batch_x.shape[0], -1)
            feature_vector.extend(features.detach().cpu().numpy())
        else:
            _, features = model(batch_x, reshape = False)

            features = torch.cat(features).reshape(model.N, -1, 2048)
            feature_vector.append(features)

    if not baseline:
        feature_vector = torch.cat(feature_vector, dim = 1).cpu().detach().numpy()

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector, dtype=int)

    return feature_vector, labels_vector

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes, metric):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.metric = metric
        self.clf = LogReg(solver='lbfgs', multi_class='multinomial', warm_start=True)

        print('Logistic regression:')
        print(f'\t solver = L-BFGS')
        print(f"\t classes = {self.num_classes}")
        print(f"\t metric = {self.metric}")

    def set_params(self, d):
        self.clf.set_params(**d)

    def fit_regression(self, X_train, y_train, X_test, y_test):
        if self.metric == 'accuracy':
            self.clf.fit(X_train, y_train)
            test_acc = 100. * self.clf.score(X_test, y_test)
            return test_acc

        elif self.metric == 'mean per-class accuracy':
            self.clf.fit(X_train, y_train)
            pred_test = self.clf.predict(X_test)

            #Get the confusion matrix
            cm = confusion_matrix(y_test, pred_test)
            cm = cm.diagonal() / cm.sum(axis=1) 
            test_score = 100. * cm.mean()

            return test_score


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim, metric):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.metric = metric
        self.clf = LinReg(solver='auto')

        print('Linear regression:')
        print(f'\t solver = AUTO')
        print(f"\t classes = {self.output_dim}")
        print(f"\t metric = {self.metric}")

    def set_params(self, d):
        self.clf.set_params(**d)

    def fit_regression(self, X_train, y_train, X_test, y_test):
        if self.metric == 'r2':
            self.clf.fit(X_train, y_train)
            test_acc = 100. * self.clf.score(X_test, y_test)
            return test_acc

        elif self.metric == 'pca':
            self.clf.fit(X_train, y_train)
            pred_test = self.clf.predict(X_test)
            test_score = dist_acc((pred_test - y_test)**2)
            return test_score

if __name__ == '__main__':
    main()