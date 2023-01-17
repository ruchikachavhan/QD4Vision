import argparse
import code
import math
import os
import time
import shutil
import warnings
import random

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
import torchvision.models as torchvision_models
import itertools
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision
import models
import datasets
from pretrain_utils import train, evaluate, save_checkpoint, RunningStats, adjust_coeff, adjust_learning_rate, get_pairwise_rowdiff, EarlyStopping
import json

# python train.py --multiprocessing-distributed --rank 0 --world-size 1 --dist-url "tcp://localhost:10001" --train_data imagenet --num_augs 6
# Models and arguments
torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base', 'convnext'] + torchvision_model_names

parser = argparse.ArgumentParser(description='Quality Diversity for Vision: Pretraining')
parser.add_argument('--data', default='/raid/s2265822/image-net100/',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--output_dir', default = '/raid/s2265822/qd4vision/saved_models/good_models1', type=str)
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0.0008, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
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
parser.add_argument('--num_encoders', default=5, type=int, help='Number of encoders')
parser.add_argument('--num_augs', default=6, type=int, help='Number of encoders')
parser.add_argument('--coeff', default=0.5, type=float, help='')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--baseline', action='store_true',
                    help='Use baseline (one backbone) models if true')
parser.add_argument('--train_data', default='imagenet',
                    help='path to dataset')


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
        print("WORLD SIZE", args.world_size)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node =  4
    # torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function

        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
            # builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # Number of classes according to dataset
    if args.train_data == 'imagenet':
        args.num_classes = 100
    elif "office31" in args.train_data:
        args.num_classes = 31
    elif "pacs" in args.train_data:
        args.num_classes = 7
    elif args.train_data == 'imagenet1k':
        args.num_classes = 1000

    if args.arch == 'resnet50':
        models_ensemble = models.BranchedResNet(N = args.num_encoders, num_classes = args.num_classes, arch = args.arch)
    elif args.arch == 'convnext':
        models_ensemble = models.DiverseConvNext(N = args.num_encoders, num_classes = args.num_classes)

    # optionally resume from a checkpoint, only for ensemble model
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            print(state_dict.keys())
            models_ensemble.load_state_dict(checkpoint['state_dict'], strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        
    
    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size / 256

    # Run DDP only for ensemble model
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # apply SyncBN
        # local_rank = int(os.environ['LOCAL_RANK'])
        models_ensemble = torch.nn.SyncBatchNorm.convert_sync_batchnorm(models_ensemble)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            models_ensemble.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            find_unused = False 
            models_ensemble = torch.nn.parallel.DistributedDataParallel(models_ensemble, device_ids=[args.gpu], find_unused_parameters=find_unused)
        else:
            models_ensemble.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            models_ensemble = torch.nn.parallel.DistributedDataParallel(models_ensemble, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        models_ensemble = models_ensemble.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(models_ensemble) # print model after SyncBatchNorm
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.AdamW(models_ensemble.parameters(), args.lr,
                                weight_decay=args.weight_decay)    
    scaler = torch.cuda.amp.GradScaler()

    # Data Loading code
    if args.num_augs == 2:
        augs_list = [datasets.dorsal_augmentations, datasets.ventral_augmentations, datasets.base_augs]
    elif args.num_augs == 5:
        # TODO: Need to change this to have combinations
        augs_list = datasets.combinations_default
        augs_list.append(datasets.base_augs)
    elif args.num_augs == 6:
        augs_list = datasets.combinations_default_edges
        augs_list.append(datasets.base_augs)


    if args.train_data in ['imagenet', 'imagenet1k']:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
    elif 'office31' in args.train_data:
        source_name = args.train_data.split("-")[-1]
        traindir = os.path.join(args.data, source_name, "images")
        valdir = os.path.join(args.data, source_name, "images")
    elif 'pacs' in args.train_data:
        sources = ['art_painting', 'cartoon', 'photo']
        traindir = [os.path.join(args.data, sources[i]) for i in range(len(sources))]



    if args.train_data == 'pacs':
        # For PACS, we split images from P, A, C into train/val
        all_train_datasets = [torchvision.datasets.ImageFolder(traindir[i],
                    datasets.CropsTransform(augs_list, args.num_augs)) for i in range(len(sources))]
        train_dataset = torch.utils.data.ConcatDataset(all_train_datasets)
        train_size = int(0.8*len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_datatset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    elif "office31" in args.train_data:
        # For Office-31, we use source domain images for training
        train_dataset = torchvision.datasets.ImageFolder(traindir,
                    datasets.CropsTransform(augs_list, args.num_augs))
        val_dataset = train_dataset
        # train_size = int(0.8*len(train_dataset))
        # val_size = len(train_dataset) - train_size
        # train_datatset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    else:
        train_dataset = torchvision.datasets.ImageFolder(traindir,
                    datasets.CropsTransform(augs_list, args.num_augs))
        val_dataset = torchvision.datasets.ImageFolder(valdir,
                    datasets.CropsTransform(augs_list, args.num_augs))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)        
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    print("Training and validation data path", traindir, valdir)
    print('Training and validation data size:', len(train_dataset), len(val_dataset))

    fname  = args.arch + "_" + args.train_data+"-supervised"+ str(args.num_augs) +'_checkpoint_%04d.pth.tar'
    coeff = args.coeff

    quality = []
    diversity = []
    best_loss = 0.0
    early_stopping = EarlyStopping(tolerance=5, min_delta=5)
    for epoch in range(0, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # Two instances of running stats: first one for mean feature and second one for covariance matrix
        running_mean = [RunningStats() for i in range(0, 2)] 

        train_avg_sim, train_acc1, train_acc5, train_loss = train(train_loader, models_ensemble, criterion, optimizer, scaler, epoch, args, running_mean, coeff)
        test_acc1, test_acc5, val_loss, val_avg_sim = evaluate(val_loader, models_ensemble, criterion, epoch, args)

        print("Training Accuracies of all backbones", train_acc1, train_acc5)
        print("Test Accuracies of all backbones", test_acc1, test_acc5)
        print("Train sim", train_avg_sim)
        print("Val sim", val_avg_sim, coeff)
        diff = get_pairwise_rowdiff(train_avg_sim).item()
        quality.append(test_acc1.mean().item())
        diversity.append(diff)
        dict = {"Q": quality, "D": diversity}
        with open(args.train_data + args.arch + "_" +  '_log_qd.json', "w") as f:
            json.dump(dict, f)
        
        np.save('train' + "_" +  args.train_data+ "_similarity_matrix_"+ str(args.num_augs)+ "augs.npy", train_avg_sim.detach().cpu().numpy())
        np.save("val" + "_" + args.train_data+ "_similarity_matrix_"+ str(args.num_augs)+ "augs.npy", val_avg_sim.detach().cpu().numpy())
        # early_stopping(train_loss, val_loss)
        # if early_stopping.early_stop:
        #     print("We are at epoch:", epoch)
        #     break
        # else:
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0): # only the first GPU saves checkpoint
            
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0): # only the first GPU saves checkpoint
                print("Saving checkpoint in", os.path.join(args.output_dir, fname % args.epochs))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': models_ensemble.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                }, is_best=False, filename=os.path.join(args.output_dir, fname % args.epochs))


if __name__ == '__main__':
    main()