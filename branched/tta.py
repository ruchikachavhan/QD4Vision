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
import torchvision
from temperature_scaling import cross_validate_temp_scaling, DummyDataset
from timm.models.convnext import convnext_base
from timm.models.resnet import resnet50
import wandb
from sklearn.metrics import r2_score
from downstream_utils import dataset_info, get_tta_dataset, dist_acc, load_backbone, get_feature_datasets
from pretrain_utils import accuracy, AverageMeter, ProgressMeter
from tqdm import tqdm
import json
import scipy
from main_linear import LogisticRegression, LinearRegression, set_params

sweep_config = {
                'method': 'random',
                'metric': {'goal': 'maximize', 'name': 'val_acc'},
                'parameters': {
                    'batch_size': {
                        'distribution': 'q_log_uniform',
                        'max': math.log(256),
                        'min': math.log(32),
                        'q': 1
                    },
                    'lr': {'distribution': 'uniform',
                                      'max': 0.5,
                                      'min': 0},
                    'weight_decay': {'distribution': 'uniform',
                                      'max': 0.00001,
                                      'min': 0},
                }
 }


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
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', default=2e-5, type=float,
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
parser.add_argument('--gpu', default=4, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--baseline', action='store_true', help="Use pretrained or QD model")
parser.add_argument('--vanilla', action='store_true', help="Use pretrained or QD model")

parser.add_argument('--test_dataset', default='VOC2007', type=str)
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
parser.add_argument('--stacking', action='store_true',
                    help='USe scikit learn stacking to classify')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')

parser.add_argument('--few_shot_reg', default=None, type=float,
                    help='image size')
parser.add_argument('-sc', '--num-samples-per-classes', default=None, type=int,
                        help='number of samples per classes.')

def main():
    main_worker()

def main_worker():
    args = parser.parse_args()
    global best_acc1

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    models_ensemble = load_backbone(args)
        
    if args.vanilla:
            train_loader, val_loader, trainval_loader, test_loader, num_classes = get_feature_datasets(args)
    else:
        train_loader, val_loader, trainval_loader, test_loader, num_classes = get_tta_dataset(args)
    X_train_feature, y_train, X_val_feature, y_val = get_features(
        train_loader, val_loader, models_ensemble, device = args.gpu, baseline = args.baseline, 
        num_classes = num_classes, mode = dataset_info[args.test_dataset]['mode'], test=False, 
        dataset_name = args.test_dataset, vanilla = args.vanilla
    )

    X_trainval_feature, y_trainval, X_test_feature, y_test = get_features(
            trainval_loader, test_loader, models_ensemble, device = args.gpu, baseline = args.baseline, 
            num_classes = num_classes, mode = dataset_info[args.test_dataset]['mode'], test=False, 
            dataset_name = args.test_dataset, tta= True, vanilla = args.vanilla
        )
    
    print(X_train_feature.shape, X_val_feature.shape, y_val.shape, X_trainval_feature.shape, X_test_feature.shape)

    if args.vanilla:
        num_feat = 2048 * 6
    else:
        num_feat = 2048
    if dataset_info[args.test_dataset]['mode'] == 'regression':
        clf = LinearRegression(num_feat, num_classes, 'r2')
    elif dataset_info[args.test_dataset]['mode'] == 'pose_estimation':
        clf = LinearRegression(num_feat, num_classes, 'pca')
    else:
        clf = LogisticRegression(num_feat, num_classes, dataset_info[args.test_dataset]['metric'])

        if dataset_info[args.test_dataset]['mode'] in ['regression', 'pose_estimation']:
            criterion = nn.L1Loss()
        elif dataset_info[args.test_dataset]['mode'] == 'classification':
            criterion = nn.CrossEntropyLoss()
    
    best_score = 0.0
    best_params = {}
    results = {}

    if dataset_info[args.test_dataset]['mode'] in ['regression', 'pose_estimation']:
        wd_range = torch.logspace(-2, 5, 100)
    else:
        wd_range = torch.logspace(-6, 5, 45)

    for wd in tqdm(wd_range, desc='Selecting best hyperparameters for classifier'):
        clf, C = set_params(clf, wd, dataset_info[args.test_dataset]['mode'])
        val_acc = clf.fit_regression(X_train_feature, y_train, X_val_feature, y_val)
        if val_acc > best_score:
            best_score = val_acc
            best_params["C"] = C

    clf, _ = set_params(clf, torch.tensor(best_params["C"]), dataset_info[args.test_dataset]['mode'])
    test_acc = clf.fit_regression(X_trainval_feature, y_trainval, X_test_feature, y_test)
    results['tta acc'] = test_acc
    results['best param'] = best_params["C"]
    print(results)
    # with open(f'results/tta/results_tta_{args.test_dataset}.json', 'w') as f:
    #     json.dump(results, f)


# Testing classes and functions
def get_features(train_loader, test_loader, model, device, baseline, num_classes, mode, test=True, vanilla = False, dataset_name=None, tta=False):
    X_train_feature, y_train = inference(train_loader, model, device, baseline, num_classes, mode, 'train', dataset_name = dataset_name, vanilla = vanilla, tta=False)
    X_test_feature, y_test = inference(test_loader, model, device, baseline, num_classes, mode, 'test' if test else 'val',  dataset_name = dataset_name, vanilla = vanilla, tta = tta)
    return X_train_feature, y_train, X_test_feature, y_test

def inference(loader, model, device, baseline, num_classes, mode, split, dataset_name, vanilla, tta = False):
    model.eval()
    feature_vector = []
    labels_vector = []
    iter = 0
    for data in tqdm(loader, desc=f'Computing features for {split} set'):
        batch_x, batch_y = data
        batch_x = batch_x.cuda(device)
        if dataset_name in ['300w', 'leeds_sports_pose', 'celeba']:
            batch_y = F.normalize(torch.tensor(batch_y), dim=1)
        labels_vector.extend(np.array(batch_y))
        if not vanilla:
            if not tta:
                features = model(batch_x).view(batch_x.shape[0], -1)
            else:
                feats = []
                for i in range(8):
                    feats.append(model(batch_x[:, i, :, :, :]).view(batch_x.shape[0], -1))
                feats = torch.stack(feats, dim=1)
                features = torch.mean(feats, dim=1)
        else:
            feats = []
            for i in range(6):
                feats.append(model(batch_x).view(batch_x.shape[0], -1))
            features = torch.stack(feats, dim=1).reshape(batch_x.shape[0], -1)
            print(features.shape)
        feature_vector.extend(features.detach().cpu().numpy())
        iter += 1

    if not baseline:
        feature_vector = torch.cat(feature_vector, dim = 1).cpu().detach().numpy()

    feature_vector = np.array(feature_vector)
    if mode == 'classification':
        labels_vector = np.array(labels_vector, dtype=int)
    else:
        labels_vector = np.array(labels_vector, dtype=float)

    return feature_vector, labels_vector


if __name__ == '__main__':
    main()
