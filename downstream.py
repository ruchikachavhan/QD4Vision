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
from torch.utils.data import DataLoader
import models
from timm.models.convnext import convnext_base
from timm.models.resnet import resnet50
from  downstream_utils import get_dataset, get_train_transform, get_val_transform
import wandb
from pretrain_utils import accuracy
import json
from test_datasets import CelebA, FacesInTheWild300W, LeedsSportsPose, AnimalPose, Causal3DIdent, ALOI, MPII
from r2score import r2_score

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

dataset_info = {
    'imagenet': {
        'class': datasets.CIFAR10, 'dir': 'CIFAR10', 'num_classes': 100,
        'splits': ['train', 'train', 'test'], 'split_size': 0.8,
        'mode': 'classification'
    },
    'dtd': {
        'class': None, 'dir': 'dtd', 'num_classes': 47,
        'splits': ['train', 'val', 'test'], 'split_size': 0.8,
        'mode': 'classification'
    },
    'sun': {
        'class': None, 'dir': 'sun', 'num_classes': 397,
        'splits': ['train', 'val', 'test'], 'split_size': 0.8,
        'mode': 'classification'
    },
    'pets': {
        'class': None, 'dir': 'pets', 'num_classes': 37,
        'splits': ['train', 'train', 'test'], 'split_size': 0.8,
        'mode': 'classification'
    },
    'rotation': {
        'class': None, 'dir': 'CIFAR10', 'num_classes': 25,
        'splits': ['train', 'val', 'test'], 'split_size': 0.7,
        'mode': 'classification'
    },
    'cub200': {
        'class': None, 'dir': 'CUB200', 'num_classes': 200,
        'splits': ['train', 'train',  'test'], 'split_size': 0.8,
        'mode': 'classification'
    },
    'cifar10': {
        'class': datasets.CIFAR10, 'dir': 'CIFAR10', 'num_classes': 10,
        'splits': ['train', 'train', 'test'], 'split_size': 0.7,
        'mode': 'classification'
    },
    'cifar100': {
        'class': None, 'dir': 'CIFAR100', 'num_classes': 100,
        'splits': ['train', 'train', 'test'], 'split_size': 0.7,
        'mode': 'classification'
    },
    'flowers': {
        'class': None, 'dir': 'flowers', 'num_classes': 102,
        'splits': ['train', 'valid', 'test'], 'split_size': 0.5,
        'mode': 'classification'
    }, 
    'celeba': {
        'class': CelebA, 'dir': 'CelebA', 'num_classes': 40,
        'splits': ['train', 'valid', 'test'], 'split_size': 0.5,
        'target_type': 'landmarks',
        'mode': 'regression'
    },
    '300w': {
        'class': FacesInTheWild300W, 'dir': '300W', 'num_classes': 136,
        'splits': ['train', 'val', 'test'], 'split_size': 0.5,
        'mode': 'regression'
    },
    'animal_pose':{
        'class': AnimalPose, 'dir': 'animal_pose', 'num_classes': 40,
        'splits': [], 'split_size': 0.8,
        'mode': 'pose_estimation'
    },
    'causal3d':{
        'class': Causal3DIdent, 'dir': 'Causal3d', 'num_classes': 10,
        'splits': ['train', 'test'], 'split_size': 0.8,
        'mode': 'regression'
    },
    'aloi':{
        'class': ALOI, 'dir': 'ALOI/png4/', 'num_classes': 1,
        'splits': [], 'split_size': 0.8,
        'mode': 'regression'
    },
    'mpii':{
        'class': MPII, 'dir': 'MPII', 'num_classes': 32,
        'splits': [], 'split_size': 0.8,
        'mode': 'pose_estimation'
    },
    'caltech101': {
        'class': None, 'dir': 'caltech-101', 'num_classes': 102,
        'splits': ['train', 'train', 'test'], 'split_size': 0.7,
        'mode': 'classification'
    },
    'leeds_sports_pose': {
        'class': LeedsSportsPose, 'dir': 'LeedsSportsPose', 'num_classes': 28,
        'splits': ['train', 'test', 'test'], 'split_size': 0.8,
        'mode': 'regression'
    }
}

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
parser.add_argument('-b', '--batch-size', default=256, type=int,
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

sweep_config = {
                'method': 'random',
                'metric': {'goal': 'minimize', 'name': 'val_loss'},
                'parameters': {
                    'lr': {'distribution': 'uniform',
                                      'max': 0.1,
                                      'min': 0},
                    'weight_decay': {'distribution': 'uniform',
                                      'max': 0.00001,
                                      'min': 0},
                    'optimizer': {'values': ['adam', 'sgd']}
                }
 }

def main():

    # sweep_id = wandb.sweep(sweep_config, project="Downstream")

    # wandb.agent(sweep_id, function = main_worker, count = 10)
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
            models_ensemble.fc = nn.Identity()
        elif args.arch == 'convnext':
            models_ensemble.head.fc = nn.Identity()
    else:
        if args.arch == 'resnet50':
            feat_dim = 2048 if args.arch == 'resnet50' else 512
            models_ensemble.base_model.branches_fc = nn.ModuleList([nn.Identity() for i in range(args.num_encoders)])
        elif args.arch == 'convnext':
            for ind in range(args.num_encoders):
                models_ensemble.base_model.head[ind].fc = nn.Identity()
    
    print(models_ensemble)
    # freeze all layers but the last fc
    for name, param in models_ensemble.named_parameters():
        # if args.few_shot_reg is None:
        if args.arch == 'resnet50':
            if "fc" not in name:
                param.requires_grad = False
        elif args.arch == 'convnext':
            if "head" not in name:
                param.requires_grad = False
        print(name, param.requires_grad)

    # infer learning rate before changing batch size, not done in hyoer-models
    init_lr = args.lr 
    models_ensemble = models_ensemble.cuda(args.gpu)

    print(models_ensemble)
    i=0
    if i == 0:
    # with wandb.init(project='Downstream', entity='my-team-ruch', config = config, mode="disabled"):
    #     config = wandb.config
        train_transform = get_train_transform(args.train_resizing, not args.no_hflip, args.color_jitter)
        val_transform = get_val_transform(args.val_resizing)
        if dataset_info[args.test_dataset]['mode'] in ['regression', 'pose_estimation']:
            num_classes = dataset_info[args.test_dataset]['num_classes']
            if len(dataset_info[args.test_dataset]['splits']) == 0:
                dataset = dataset_info[args.test_dataset]['class'](os.path.join(args.data_root, dataset_info[args.test_dataset]['dir'])
                                                    ,transform = train_transform)
                train_size = int(len(dataset)* dataset_info[args.test_dataset]['split_size'])
                train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
            else:
                train_dataset = dataset_info[args.test_dataset]['class'](os.path.join(args.data_root, dataset_info[args.test_dataset]['dir']), 
                                    split = dataset_info[args.test_dataset]['splits'][0], transform = train_transform)
                val_dataset = dataset_info[args.test_dataset]['class'](os.path.join(args.data_root, dataset_info[args.test_dataset]['dir']), 
                                    split = dataset_info[args.test_dataset]['splits'][1], transform = val_transform)
            print("lengths", len(train_dataset), len(val_dataset))
        else:
            train_dataset, val_dataset, num_classes = get_dataset(args.test_dataset, args.data_root, train_transform,
                                                                        val_transform, args.sample_rate,
                                                                        args.num_samples_per_classes)
        
        # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
        #                         num_workers=args.workers, drop_last=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.workers, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        print("train ", train_loader, val_loader)
        for split in ["train", "val"]:
            save_path = os.path.join("features", args.arch+ "_" + split + "_" + args.test_dataset+ ".pt")
            save_path_labels = os.path.join("features", args.arch+ "_" + split + "_" + args.test_dataset+ "_labels.pt")
            if not os.path.exists(save_path) or not os.path.exists(save_path_labels):
                if split == 'train':
                    get_model_features(train_loader, split , args.test_dataset, args.arch, models_ensemble, args.gpu, output_path = 'features', baseline = args.baseline)
                else:
                    get_model_features(val_loader, split, args.test_dataset, args.arch, models_ensemble, args.gpu, output_path = 'features', baseline = args.baseline)
            else:
                print("Found precomputed features in %s" % (save_path))
            if split == 'train':
                train_images = torch.load(save_path)
                train_labels = torch.load(save_path_labels)
            if split == 'val':
                val_images = torch.load(save_path)
                val_labels = torch.load(save_path_labels)

        if dataset_info[args.test_dataset] == 'classficiation':
            criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        else:
            criterion = nn.L1Loss().cuda(args.gpu)
        if args.baseline:
            classifier = nn.Linear(models_ensemble.num_feat, num_classes).cuda(args.gpu)
        else:
            classifier = nn.ModuleList([nn.Linear(models_ensemble.num_feat, num_classes).cuda(args.gpu) for _ in range(args.num_encoders)])
            classifier.train()
            # for n in range(0, args.num_encoders):
            # classifier.weight.data.normal_(mean=0.0, std=0.01)
            # classifier.bias.data.zero_()
        
        optimizer = torch.optim.SGD(list(classifier.parameters()), lr = args.lr, weight_decay = args.weight_decay)
        # if config.optimizer == 'Adam':
        #     optimizer = torch.optim.AdamW(list(classifier.parameters()), lr = args.lr, weight_decay = config.weight_decay)
        # elif config.optimizer == 'sgd':
        #     optimizer = torch.optim.SGD(list(classifier.parameters()), lr = config.lr, weight_decay = config.weight_decay, momentum=0.9)

        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.1, last_epoch=- 1, verbose=False)

    
        # wandb.watch(classifier, criterion, log="all")
        for epoch in range(0, args.epochs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            # epoch_results = {}
            print(" --- Training ----")
            train_fc(train_images, train_labels, classifier, optimizer, criterion, args, args.batch_size, epoch, train_mode = True)
            end.record()
            # Waits for everything to finish running
            torch.cuda.synchronize()
            time = start.elapsed_time(end)
            print("Time for training", start.elapsed_time(end))
            print(" --- Validation ----")
            train_fc(val_images, val_labels, classifier, optimizer, criterion, args, args.batch_size, epoch, train_mode = False)
            # scheduler.step()


def standardize(data):
    means = data.mean(2, keepdim=True)
    std = data.std(2, keepdim=True)
    return (data - means)/std


def train_fc(image_loader, label_loader, classifier, optimizer, criterion,  args, batch_size, epoch, train_mode):
    indices = torch.randperm(image_loader.size()[0])
    image_loader = image_loader[indices]
    label_loader = label_loader[indices]
    batches = torch.split(image_loader, batch_size)
    batches_y = torch.split(label_loader, batch_size)

    avg_loss = 0.0
    avg_acc = 0.0
    all_logits, all_labels = [], []
    # np.array([0.0 for _ in range(args.num_encoders)])
    for iter in range(len(batches)):
        x = batches[iter].cuda(args.gpu)
        y = batches_y[iter].cuda(args.gpu)
        if args.baseline:
            preds = classifier(x)
            loss = criterion(preds, y)
        else:
            loss = 0.0
            acc = []
            preds, acc = [], []
            loss = 0.0
            for k in range(args.num_encoders):
                pred = classifier[k](x[:, k, :])
                loss += criterion(pred, y)
                preds.append(pred)
                accuracy = get_scores(pred, y, args.test_dataset)
                acc.append(accuracy)

            avg_loss += loss.item()
            avg_acc += np.array(acc)

            if not train_mode:
                all_logits.append(torch.cat(preds).reshape(args.num_encoders, x.shape[0], -1).detach())
                all_labels.append(y)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            loss = optimizer.step()

        # if train_mode: 
        #     wandb.log({'epoch': epoch, 'train_loss': loss, "train_accuracy": accuracy})
        # else: 
        #     wandb.log({'epoch': epoch, 'val_loss': loss, "val_accuracy": accuracy})

    print("EPOCH", epoch, "Loss:" , avg_loss/(iter+1), "Accuracy:" , avg_acc/(iter+1))
    if not train_mode:
        all_logits = torch.cat(all_logits, dim=1)
        all_labels = torch.cat(all_labels, dim=0)
        lstsq_weights, lstsq_loss, lstsq_accuracies = get_lstsq_solution(all_logits, all_labels, 10, criterion, args.test_dataset)

        # Save LSTSQ search weights in a file 
        print(lstsq_weights.tolist(), lstsq_loss, lstsq_accuracies.item())
        results = {"lstsq_weights": lstsq_weights.tolist(), "loss": lstsq_loss, "accuracy": lstsq_accuracies.item()}
        with open(os.path.join("downstream_results" , args.test_dataset + "_search_results_" + '.json'), 'w') as f:
            json.dump(results, f)

        print("----------LSTSQ Search-----------")
        print("Weights:", lstsq_weights)
        print("Loss:", lstsq_loss)
        print("Accuracies:", lstsq_accuracies)
    

def get_lstsq_solution(logits, labels, num_classes, criterion, test_dataset):  
    if dataset_info[test_dataset]['mode'] in ['regression', 'pose_estimation']:
        A = logits.reshape(logits.shape[0], -1)
        A = torch.transpose(A, 0, 1).detach().cpu().numpy()
        labels_ = labels.reshape(-1).cpu().numpy()
    else:
        A = logits.reshape(logits.shape[0], -1)
        A = torch.transpose(A, 0, 1).detach().cpu().numpy()
        labels_ = F.one_hot(labels, num_classes = num_classes).reshape(-1).cpu().numpy()
        
    x1, _ , _, _ = np.linalg.lstsq(A, labels_)
    x1 = torch.from_numpy(x1).cuda(logits.device).float()
    logits_ = logits.permute(1, 2, 0)
    weighted_logits = torch.matmul(logits_, x1)
    loss = criterion(weighted_logits, labels).item()
    acc = get_scores(weighted_logits, labels, test_dataset)
    return x1.detach().cpu().numpy(), loss, acc

def get_scores(preds, labels, test_dataset):
    if dataset_info[test_dataset]['mode'] == 'regression':
        return r2_score(labels.flatten().detach().cpu().numpy(), preds.flatten().detach().cpu().numpy())
    elif dataset_info[test_dataset]['mode'] == 'pose_estimation':
        return dist_acc((preds-labels)**2)
    elif dataset_info[test_dataset]['mode'] == 'classification':
        return accuracy(preds, labels, topk=(1, 5))[0]

def dist_acc(dists, thr=0.01):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dists = dists.detach().cpu().numpy()
    dist_cal = np.not_equal(dists, 0.0)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1

def get_model_features(loader, split, dataset, arch, model, gpu, output_path = 'features', baseline = False):
    save_path = os.path.join(output_path, arch+ "_" + split + "_" + dataset)
    print("Converting images into features and saving in .pt file for split %s ...." % (split))

    all_features = []
    all_labels = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for iter, (images, labels) in enumerate(loader):
        if gpu is not None:
            images = images.cuda(gpu, non_blocking=True) 
            labels = labels.cuda(gpu, non_blocking=True)
        
        
        with torch.cuda.amp.autocast(False):
            if baseline:
                feats = model(images)
                feats = feats.detach().cpu()
                all_features.append(feats)
            else:
                _, feats = model(images, reshape = False)
                feats = torch.cat(feats).reshape(model.N, -1, model.num_feat)
                feats = feats.permute(1, 0, 2)
                feats = feats.detach().cpu()
                all_labels.append(labels)
                all_features.append(feats)

    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    time = start.elapsed_time(end)
    print("Time taken to featurise", split, start.elapsed_time(end))
    all_features = torch.cat(all_features, dim = 0)
    
    all_labels = torch.cat(all_labels)
    torch.save(all_features, save_path + ".pt")
    torch.save(all_labels, save_path + "_labels.pt")

if __name__ == '__main__':
    main()