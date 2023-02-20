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
from downstream_utils import *
from pretrain_utils import accuracy, AverageMeter, ProgressMeter
from tqdm import tqdm
import json
import scipy
from sklearn.preprocessing import minmax_scale


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
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', default=8e-4, type=float,
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
parser.add_argument('--warmup-epochs', default=20, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--baseline', action='store_true', help="Use pretrained or QD model")
parser.add_argument('--test_dataset', default='VOC2007', type=str)
parser.add_argument('--data_root', default='/raid/s2265822/TestDatasets/', type = str)
parser.add_argument('--num_encoders', default=5, type=int, help='Number of encoders')
parser.add_argument('--ft-mode', type=str, default='ft', help='resize mode during training')

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

def main_worker(config=None):
    args = parser.parse_args()
    global best_acc1

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    models_ensemble = load_backbone(args)
    print(models_ensemble)

    models_ensemble.fc = nn.Linear(2048, dataset_info[args.test_dataset]['num_classes'])
    models_ensemble = models_ensemble.cuda(args.gpu)
    print(models_ensemble)
    for param in models_ensemble.parameters():
        param.requires_grad = True
        
    train_loader, val_loader, trainval_loader, test_loader, num_classes = get_feature_datasets_ood(args)

    optimizer = torch.optim.Adam(models_ensemble.parameters(), lr=args.lr,
                                # momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    if dataset_info[args.test_dataset]['mode'] in ['regression', 'pose_estimation']:
        criterion = nn.L1Loss()
    elif dataset_info[args.test_dataset]['mode'] == 'classification':
        criterion = nn.CrossEntropyLoss()
        
    results = {}
    best_score = 0.0
    for epoch in range(args.epochs):
        if args.ft_mode == 'lpft':
            if epoch < args.warmup_epochs:
                for name, param in models_ensemble.named_parameters():
                    if 'fc' not in name:
                        param.requires_grad = False
            # Turn gradients on for all layers after linear probing
            elif epoch == args.warmup_epochs:
                for name, param in models_ensemble.named_parameters():
                    param.requires_grad = True 
             

        train_acc, train_loss = train(trainval_loader, models_ensemble, args.gpu, optimizer, dataset_info[args.test_dataset]['mode'], criterion, epoch, args, train_mode = True)
        # lr_scheduler.step()
        test_acc, test_loss = train(test_loader, models_ensemble, args.gpu, optimizer, dataset_info[args.test_dataset]['mode'], criterion, epoch, args, train_mode = False)
       
        print("Epoch: {}, Train Loss: {}, Train Acc: {}, Val Loss: {}, Val Acc:{}".format(epoch, train_loss, train_acc, test_loss, test_acc))
        epoch_results = {}
        epoch_results["Train acc"] = train_acc.item()
        epoch_results["Train loss"] = train_loss
        epoch_results["Val loss"] = test_loss
        epoch_results["Val acc"] = test_acc.item()
        # if not args.baseline:
        #     epoch_results["Weights"] = weights.tolist

        if test_acc > best_score:
            best_score = test_acc.item()

        results[epoch] = epoch_results
    results["Best score"] = best_score

    print(results)
    print("Best score: ", best_score)

    with open(os.path.join("results", "finetuning", 'results_ft_%s_%s.json'%(args.test_dataset, args.ft_mode)), 'w') as f:
        json.dump(results, f)

def train(loader, model, device, optimizer, mode, criterion, epoch, args, train_mode):
    model.eval()
    feature_vector = []
    labels_vector = []
    iter = 0
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    ce_losses = AverageMeter('CE Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    prefix_ = "Train: [{}]".format(epoch) if train_mode else "Val: [{}]".format(epoch)
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, ce_losses, top1],
        prefix=prefix_)

    end = time.time()   
    for data in tqdm(loader, desc=f'Training Epoch {epoch}'):
        batch_x, batch_y = data
        batch_x = batch_x.cuda(device).reshape(-1, 3, 224, 224)
        batch_y = batch_y.cuda(device)
        if args.test_dataset in ['leeds_sports_pose', '300w']:
            batch_y = nn.functional.normalize(batch_y, dim=1)

        output = model(batch_x).view(batch_x.shape[0], -1)
        loss = criterion(output, batch_y)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if mode == 'classification':
            acc1, _ = accuracy(output, batch_y, topk=(1, 5))
        elif mode == 'regression':
            acc1 = r2_score(batch_y.cpu().detach().numpy().reshape(-1), output.cpu().detach().numpy().reshape(-1))
        elif mode == 'pose_estimation':
            acc1 = dist_acc((output.cpu().detach().numpy() - batch_y.cpu().detach().numpy())**2)

        ce_losses.update(loss.item(), batch_x.size(0))
        top1.update(acc1, batch_x.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        torch.cuda.empty_cache()
        if iter % 10 == 0:
            progress.display(iter)
    
    return top1.avg, ce_losses.avg

if __name__ == '__main__':
    main()
