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

sweep_config = {
                'method': 'random',
                'metric': {'goal': 'minimize', 'name': 'val_loss'},
                'parameters': {
                    'lr': {'distribution': 'uniform',
                                      'max': 0.5,
                                      'min': 0},
                    'weight_decay': {'distribution': 'uniform',
                                      'max': 0.001,
                                      'min': 0}
                    }
 }

def main():

    sweep_id = wandb.sweep(sweep_config, project="Downstream")

    wandb.agent(sweep_id, function = main_worker, count = 10)
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

    # print(models_ensemble)
    # i=0
    # if i == 0:
    with wandb.init(project='Downstream', name = args.test_dataset+"_downstream", entity='my-team-ruch', config = config):
        config = wandb.config
        
        train_images, train_labels, val_images, val_labels, test_images, test_labels, num_classes = get_feature_datasets(args, models_ensemble)
        print("Dataset sizes", train_images.shape[0], val_images.shape[0], test_images.shape[0])

        if dataset_info[args.test_dataset]['mode'] == 'classification':
            criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        else:
            criterion = nn.L1Loss().cuda(args.gpu)

        if args.baseline:
            classifier = nn.Linear(models_ensemble.feat_dim, num_classes).cuda(args.gpu)
        else:
            classifier = nn.ModuleList([nn.Linear(models_ensemble.num_feat, num_classes).cuda(args.gpu) for _ in range(args.num_encoders)])
            classifier.train()
        
        optimizer = torch.optim.LBFGS(list(classifier.parameters()), lr = config.lr, max_iter = 10)
        
        # , weight_decay = config.weight_decay,  momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.1, last_epoch=- 1, verbose=False)

        wandb.watch(classifier, criterion, log="all")
        if args.baseline:
            results_path = open(os.path.join("features", 'baseline', args.arch + "_"+ args.test_dataset+ ".txt"), 'w')
        else:        
            results_path = open(os.path.join("features", args.arch + "_"+ args.test_dataset+ ".txt"), 'w')

        for epoch in range(0, args.epochs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            # epoch_results = {}
            train_loss, train_acc, _ = train_fc(train_images, train_labels, classifier, optimizer, criterion, args, len(train_images), epoch, train_mode = True)
            end.record()
            # Waits for everything to finish running
            torch.cuda.synchronize()
            time = start.elapsed_time(end)
            # print("Time for training", start.elapsed_time(end))
            # print(" --- Validation ----")
            val_loss, val_acc, lstsq_weights = train_fc(val_images, val_labels, classifier, optimizer, criterion, args, len(val_images), epoch, train_mode = False)
            test_loss, test_acc = test(epoch, test_images, test_labels, classifier, lstsq_weights, criterion, args)

            print("Epoch: %.2i Training loss: %.4f Training accuracy: %.4f Val loss: %.4f Val accuracy: %.4f Test loss: %.4f Test accuracy: %.4f"
                             % (epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))
            
            results_str = "Epoch: %.2i Training loss: %.4f Training accuracy: %.4f Val loss: %.4f Val accuracy: %.4f Test loss: %.4f Test accuracy: %.4f" % (epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)
            results_path.write(results_str)
            results_path.write("\n")

            if not args.baseline:
                print("----------LSTSQ Search-----------")
                print("Weights:", lstsq_weights)
                print("Loss:", val_loss)
                print("Accuracies:", val_acc)

            wandb.log({'epoch': epoch, 'train_loss': train_loss, "train_accuracy": train_acc, "val_loss": val_loss, "val_acc": val_acc, "test_loss": test_loss, "test_acc": test_acc})
            scheduler.step()




if __name__ == '__main__':
    main()