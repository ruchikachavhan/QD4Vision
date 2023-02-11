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
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision
from tsa_resnet import create_tsa_resnet, Bottleneck 
import datasets
import wandb
import json
from pretrain_utils import train, evaluate, save_checkpoint

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base', 'convnext'] + torchvision_model_names

parser = argparse.ArgumentParser(description='Quality Diversity for Vision: Pretraining')

# Path args
parser.add_argument('--data', default='/raid/s2265822/image-net100/',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--output_dir', default = '/raid/s2265822/saved_models/qd4vision/adapters', type=str)

# DDP args
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
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')

# Training parameters
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model_type', default='adapters', type=str, 
                    help='which model type to use')
parser.add_argument('--n-eval-adapters', default=16, type=int, help='Number of adapters per convolution layer')
parser.add_argument('--num_adapters', default=5, type=int, help='Number of adapters per convolution layer')
parser.add_argument('--num_augs', default=5, type=int, help='Number of encoders')
parser.add_argument('--baseline', action='store_true',
                    help='Use baseline (one backbone) models if true')
parser.add_argument('--mixup', action='store_true',
                    help='Use mixup')
parser.add_argument('--cutmix', action='store_true',
                    help='Use cutmix')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument("--quality-warmup", type=int, default=20, 
                help="the number of iterations for which only supervised training will be done",)
parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")
parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")
parser.add_argument("--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)")
parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")

# Learning rates, scheduler, and EMA
parser.add_argument('--lr', '--learning-rate', default=1.0, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=8e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')

parser.add_argument('--lr-warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')

parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
parser.add_argument('--train_data', default='imagenet',
                    help='path to dataset')
parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
parser.add_argument("--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)")
parser.add_argument("--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters")
parser.add_argument("--model-ema-steps", type=int, default=32, help="the number of iterations that controls how often to update the EMA model (default: 32)",)
parser.add_argument("--model-ema-decay", type=float, default=0.99998, help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",)


def main():
    args = parser.parse_args()
    print(args)
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

    ngpus_per_node = torch.cuda.device_count()
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

    ############################### Model Instantiation ########################
    pretrained_state_dict = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).state_dict()
    # pretrained_state_dict.pop('fc.weight')
    # pretrained_state_dict.pop('fc.bias')
    model_adapters = create_tsa_resnet(pretrained_state_dict, Bottleneck, [3, 4, 6, 3], True)

    ############################### Optimizers and schedulers ###########################
    if args.opt.startswith("sgd"):
        optimizer = torch.optim.SGD(
            model_adapters.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    elif args.opt == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model_adapters.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif args.opt == "adamw":
        optimizer = torch.optim.AdamW(model_adapters.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler()

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == 'cycliclr':
         main_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr = 0.01, max_lr = args.lr
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs], verbose = True
        )
    else:
        lr_scheduler = main_lr_scheduler

    ################################ DDP ##############################
     # Run DDP only for ensemble model
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # apply SyncBN
        # local_rank = int(os.environ['LOCAL_RANK'])
        model_adapters = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_adapters)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model_adapters.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            find_unused = True 
            model_adapters = torch.nn.parallel.DistributedDataParallel(model_adapters, device_ids=[args.gpu], find_unused_parameters=find_unused)
        else:
            model_adapters.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model_adapters = torch.nn.parallel.DistributedDataParallel(model_adapters, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model_adapters = model_adapters.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model_adapters) # print model after SyncBatchNorm

   ############################# Exponenetial moving average #############################
    if args.model_ema:
        model_without_ddp = model_adapters.module
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = ExponentialMovingAverage(model_without_ddp, device=torch.device(args.gpu), decay=1.0 - alpha)

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
            
            model_without_ddp.load_state_dict(state_dict, strict=False)            
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if args.model_ema:
                model_ema.load_state_dict(checkpoint["model_ema"])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 0
        # infer learning rate before changing batch size
   
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda(args.gpu)

    ############################# Data Loading code ############################
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')        

    train_dataset = torchvision.datasets.ImageFolder(traindir,
                datasets.CropsTransform(datasets.default_augmentations_edges, args.num_augs))
    val_dataset = torchvision.datasets.ImageFolder(valdir,
                datasets.CropsTransform_all(datasets.default_augmentations_edges, args.num_augs))

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


    fname  = "adapters_" + args.arch + "_" + args.train_data+"-supervised"+ str(args.num_augs) +'_checkpoint_%04d.pth.tar'
    quality = []
    diversity = []
    best_loss = 0.0

    with wandb.init(project="QD ImageNet pretraining with adapters", name=f"experiment_{'QD4v_adapters'}",  config={
        "learning_rate": args.lr,
        "architecture": "R50",
        "dataset": "ImageNet",
        "epochs": args.epochs,
      }):
        results_dict = {}
        for epoch in range(args.start_epoch, args.epochs):
            print("------------ EPOCH -------------", epoch + 1)
            if args.distributed:
                train_sampler.set_epoch(epoch)            

            train_acc1, train_acc5, train_loss = train(train_loader, model_adapters, criterion, 
                                    optimizer, scaler, epoch, args)
            lr_scheduler.step()
            print("Training Accuracy", train_acc1, "Training Loss", train_loss)
            result_file_name = os.path.join("adapters/train_results", "Epoch_" + str(epoch) + ".json")
            test_acc, similarity_matrix, diff, adapter_configurations = evaluate(val_loader, model_adapters, criterion, epoch, args)
            results_dict['test acc'] = test_acc.tolist()
            results_dict["similarity_matrix"] = similarity_matrix.tolist()
            results_dict["diff"] = diff.item()
            results_dict['config'] = [adapter_configurations[i].tolist() for i in range(len(adapter_configurations))]
            with open(result_file_name, 'w') as f:
                json.dump(results_dict, f)
        
            print("Number of configurations tested:", args.n_eval_adapters)
            print("Test accuracy of configurations", test_acc)
            print("Diversity", diff.item())
            print("Results saved in", result_file_name)
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank == 0): # only the first GPU saves checkpoint
                
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank == 0): # only the first GPU saves checkpoint
                    print("Saving checkpoint in", os.path.join(args.output_dir, fname % args.epochs))
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model_adapters.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        'scaler': scaler.state_dict(),
                        # 'model_ema': model_ema.state_dict(),
                    }, is_best=False, filename=os.path.join(args.output_dir, fname % args.epochs))


if __name__ == '__main__':
    main()