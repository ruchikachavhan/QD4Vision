import torch
from torch.utils.data import Subset
import random
import tllib.vision.datasets as datasets
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pretrain_utils import accuracy
import json
from test_datasets import CelebA, FacesInTheWild300W, LeedsSportsPose, AnimalPose, Causal3DIdent, ALOI, MPII
from r2score import r2_score
import os
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import models
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn


# dataset_dict = ['ImageList', 'Office31', 'OfficeHome', "VisDA2017", "OfficeCaltech", "DomainNet", "ImageNetR",
#            "ImageNetSketch", "Aircraft", "cub200", "StanfordCars", "StanfordDogs", "COCO70", "OxfordIIITPets", "PACS",
#            "DTD", "OxfordFlowers102", "PatchCamelyon", "Retinopathy", "EuroSAT", "Resisc45", "Food101", "SUN397",
#            "Caltech101", "CIFAR10", "CIFAR100"]

generator = torch.Generator()
generator.manual_seed(0)

dataset_info = {
    'DTD': {
        'class': None, 'dir': 'dtd/', 'num_classes': 47,
        'splits': ['train', 'validation', 'test'], 'split_size': 0.8,
        'mode': 'classification', 'metric': 'accuracy'
    },
    'Aircraft': {
        'class': None, 'dir': 'Aircraft', 'num_classes': 10,
        'splits': ['train', 'val', 'test'], 'split_size': 0.8,
        'mode': 'classification', 'metric': 'mean per-class accuracy'
    },
    'StanfordCars': {
        'class': None, 'dir': 'Cars/', 'num_classes': 397,
        'splits': ['train', 'val', 'test'], 'split_size': 0.8,
        'mode': 'classification', 'metric': 'accuracy'
    },
    'OxfordIIITPets': {
        'class': None, 'dir': 'pets/oxford-iiit-pet/', 'num_classes': 37,
        'splits': ['train', 'train', 'test'], 'split_size': 0.8,
        'mode': 'classification', 'metric': 'mean per-class accuracy'
    },
    'cub200': {
        'class': None, 'dir': 'CUB200', 'num_classes': 200,
        'splits': ['train', 'train',  'test'], 'split_size': 0.8,
        'mode': 'classification'
    },
    'CIFAR10': {
        'class': datasets.CIFAR10, 'dir': 'CIFAR10', 'num_classes': 10,
        'splits': ['train', 'val', 'test'], 'split_size': 0.7,
        'mode': 'classification', 'metric': 'accuracy'
    },
    'CIFAR100': {
        'class': datasets.CIFAR100, 'dir': 'CIFAR100_new/CIFAR100/', 'num_classes': 100,
        'splits': ['train', 'val', 'test'], 'split_size': 0.7,
        'mode': 'classification', 'metric': 'accuracy'
    },
    'OxfordFlowers102': {
        'class': None, 'dir': 'flowers102-new/flowers/', 'num_classes': 102,
        'splits': ['train', 'validation', 'test'], 'split_size': 0.5,
        'mode': 'classification', 'metric': 'mean per-class accuracy'
    }, 
    'celeba': {
        'class': CelebA, 'dir': 'CelebA', 'num_classes': 40,
        'splits': ['train', 'valid', 'test'], 'split_size': 0.5,
        'target_type': 'landmarks',
        'mode': 'regression', 'metric': 'r2'
    },
    '300w': {
        'class': FacesInTheWild300W, 'dir': '300W', 'num_classes': 136,
        'splits': ['train', 'valid', 'test'], 'split_size': 0.5,
        'mode': 'regression', 'metric': 'r2'
    },
    'animal_pose':{
        'class': AnimalPose, 'dir': 'animal_pose/animalpose_keypoint_new/', 'num_classes': 40,
        'splits': [], 'split_size': 0.6,
        'mode': 'pose_estimation', 'metric': 'pca'
    },
    'causal3d':{
        'class': Causal3DIdent, 'dir': 'Causal3d', 'num_classes': 10,
        'splits': ['train', 'test'], 'split_size': 0.6,
        'mode': 'regression', 'metric': 'r2'
    },
    'aloi':{
        'class': ALOI, 'dir': 'ALOI/png4/', 'num_classes': 24,
        'splits': [], 'split_size': 0.6,
        'mode': 'classification', 'metric': 'accuracy'
    },
    'mpii':{
        'class': MPII, 'dir': 'mpii', 'num_classes': 32,
        'splits': [], 'split_size': 0.6,
        'mode': 'pose_estimation', 'metric': 'pca'
    },
    'Caltech101': {
        'class': None, 'dir': 'Caltech101', 'num_classes': 102,
        'splits': ['train', 'train', 'test'], 'split_size': 0.7,
        'mode': 'classification', 'metric': 'mean per-class accuracy'
    },
    'leeds_sports_pose': {
        'class': LeedsSportsPose, 'dir': 'LeedsSportsPose', 'num_classes': 28,
        'splits': ['train', 'test'], 'split_size': 0.8,
        'mode': 'regression', 'metric': 'r2'
    }
}

def get_dataset(dataset_name, root, train_transform, val_transform, sample_rate=100, num_samples_per_classes=None):
    """
    When sample_rate < 100,  e.g. sample_rate = 50, use 50% data to train the model.
    Otherwise,
        if num_samples_per_classes is not None, e.g. 5, then sample 5 images for each class, and use them to train the model;
        otherwise, keep all the data.
    """
    if dataset_name == 'CIFAR10':
        dataset = datasets.CIFAR10
    elif dataset_name == 'CIFAR100':
        dataset = datasets.CIFAR100
    else:
        dataset = datasets.__dict__[dataset_name]
    if sample_rate < 100:
        train_dataset = dataset(root=root, split='train', sample_rate=sample_rate, download=True, transform=train_transform)
        test_dataset = dataset(root=root, split='test', sample_rate=100, download=True, transform=val_transform)
        num_classes = train_dataset.num_classes
    else:
        train_dataset = dataset(root=root, split='train', download=False, transform=train_transform)
        num_classes = train_dataset.num_classes
        if dataset_name in ['DTD', 'OxfordFlowers102']:
            # Val split is available for these datasets
            val_dataset = dataset(root=root, split='validation', download=False, transform=val_transform)
        else:
            train_size = int(0.8*len(train_dataset))
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size],
                                                generator = generator)

        test_dataset = dataset(root=root, split='test', download=False, transform=val_transform)
        

    return train_dataset, val_dataset, test_dataset, num_classes


def get_train_transform(resizing='default', random_horizontal_flip=True, random_color_jitter=False):
    """
    resizing mode:
        - default: take a random resized crop of size 224 with scale in [0.2, 1.];
        - res: resize the image to 224;
        - res.|crop: resize the image to 256 and take a random crop of size 224;
        - res.sma|crop: resize the image keeping its aspect ratio such that the
            smaller side is 256, then take a random crop of size 224;
        – inc.crop: “inception crop” from (Szegedy et al., 2015);
        – cif.crop: resize the image to 224, zero-pad it by 28 on each side, then take a random crop of size 224.
    """
    if resizing == 'default':
        transform = T.RandomResizedCrop(224, scale=(0.2, 1.))
    elif resizing == 'res.':
        transform = T.Resize((224, 224))
    elif resizing == 'res.|crop':
        transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(224)
        ])
    elif resizing == "res.sma|crop":
        transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'inc.crop':
        transform = T.RandomResizedCrop(224)
    elif resizing == 'cif.crop':
        transform = T.Compose([
            T.Resize((224, 224)),
            T.Pad(28),
            T.RandomCrop(224),
        ])
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return T.Compose(transforms)

def get_val_transform(resizing='default'):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
        – res.|crop: resize the image such that the smaller side is of size 256 and
            then take a central crop of size 224.
    """
    if resizing == 'default':
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = T.Resize((224, 224))
    elif resizing == 'res.|crop':
        transform = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
        ])
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_feature_datasets(args, models_ensemble): 
    train_transform = get_train_transform(args.train_resizing, not args.no_hflip, args.color_jitter)
    val_transform = get_val_transform(args.val_resizing)

    if dataset_info[args.test_dataset]['mode'] in ['regression', 'pose_estimation'] or args.test_dataset == 'aloi':
        num_classes = dataset_info[args.test_dataset]['num_classes']
        # if no splilt is given, then divide training data into train and val
        if len(dataset_info[args.test_dataset]['splits']) == 0:
            dataset = dataset_info[args.test_dataset]['class'](os.path.join(args.data_root, dataset_info[args.test_dataset]['dir'])
                                                ,transform = train_transform)
            train_size = int(len(dataset)* dataset_info[args.test_dataset]['split_size'])
            val_size = (len(dataset) - train_size)//2
            test_size = len(dataset) - train_size - val_size
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], 
                                                                        generator = generator)
        else: 
            # If split is given, use the split to make dataloaders
            train_dataset = dataset_info[args.test_dataset]['class'](os.path.join(args.data_root, dataset_info[args.test_dataset]['dir']), 
                                split = 'train', transform = train_transform)
            if len(dataset_info[args.test_dataset]['splits']) == 3:
                val_dataset = dataset_info[args.test_dataset]['class'](os.path.join(args.data_root, dataset_info[args.test_dataset]['dir']), 
                                split = dataset_info[args.test_dataset]['splits'][1], transform = val_transform)
                test_dataset = dataset_info[args.test_dataset]['class'](os.path.join(args.data_root, dataset_info[args.test_dataset]['dir']), 
                                split = dataset_info[args.test_dataset]['splits'][2], transform = val_transform)
            elif(len(dataset_info[args.test_dataset]['splits'])) == 2: 
                # only contains train and val splits
                train_size = int(len(train_dataset)* dataset_info[args.test_dataset]['split_size'])
                val_size = len(train_dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], 
                                                                            generator = generator)
                test_dataset = dataset_info[args.test_dataset]['class'](os.path.join(args.data_root, dataset_info[args.test_dataset]['dir']), 
                                split = 'test', transform = val_transform)
                print("Datasets", len(train_dataset), len(val_dataset), len(test_dataset))
                
    else:
        print(os.path.join(args.data_root, dataset_info[args.test_dataset]['dir']))
        train_dataset, val_dataset, test_dataset, num_classes = get_dataset(args.test_dataset, os.path.join(args.data_root, dataset_info[args.test_dataset]['dir']), train_transform,
                                                                    val_transform, args.sample_rate,
                                                                    args.num_samples_per_classes)

    print("Datasets", len(train_dataset), len(val_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    trainval_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    trainval_loader = DataLoader(trainval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    return train_loader, val_loader, trainval_loader, test_loader, num_classes

def min_max_scale(data):
    # print(data.shape)
    min = data.min(1)[0].view(-1, 1)
    max = data.max(1)[0].view(-1, 1)
    # print(min.shape, max.shape)
    data = (data - min)/(max-min)
    return data 

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
            def loss_closure():
                pred = classifier(x)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                return loss
        else:
            loss = 0.0
            acc = []
            preds, acc = [], []
            loss = 0.0
            for k in range(args.num_encoders):
                pred = classifier[k](x[:, k, :])
                loss += criterion(pred, y)
                preds.append(pred)
                accuracy = get_scores(pred, y, args.test_dataset).cpu().numpy()
                acc.append(accuracy)

                avg_loss += loss.item()
                avg_acc += np.array(acc)

            if not train_mode:
                all_logits.append(torch.cat(preds).reshape(args.num_encoders, x.shape[0], -1).detach())
                all_labels.append(y)

        if train_mode:
            optimizer.step(loss_closure)

        pred = classifier(x)
        loss = criterion(pred, y)
        accuracy = get_scores(pred, y, args.test_dataset)
        avg_acc += accuracy
        avg_loss += loss.item()
        # optimizer.zero_grad()
        # loss.backward()
        # loss = optimizer.step()

    if not args.baseline:
        if not train_mode:
            all_logits = torch.cat(all_logits, dim=1)
            all_labels = torch.cat(all_labels, dim=0)
            
            lstsq_weights, lstsq_loss, lstsq_accuracies = get_lstsq_solution(all_logits, all_labels, 10, criterion, args.test_dataset)

            # Save LSTSQ search weights in a file 
            results = {"lstsq_weights": lstsq_weights.tolist(), "loss": lstsq_loss, "accuracy": lstsq_accuracies.item()}
            with open(os.path.join("downstream_results" , args.test_dataset + "_search_results_" + '.json'), 'w') as f:
                json.dump(results, f)

    if args.baseline or train_mode:
        return avg_loss/(iter+1), avg_acc/(iter+1), None
    else:
        return lstsq_loss, lstsq_accuracies, lstsq_weights


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

def test(epoch, image_loader, label_loader, classifier, lstsq_weights, criterion, args):
    indices = torch.randperm(image_loader.size()[0])
    image_loader = image_loader[indices]
    label_loader = label_loader[indices]
    batches = torch.split(image_loader, args.batch_size)
    batches_y = torch.split(label_loader, args.batch_size)

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
            acc = get_scores(preds, y, args.test_dataset)
            avg_acc += acc
            avg_loss += loss.item()
        else:
            loss = 0.0
            acc = []
            preds, acc = [], []
            loss = 0.0
            for k in range(args.num_encoders):
                pred = classifier[k](x[:, k, :])
                loss += criterion(pred, y)
                preds.append(pred)
                
            preds = torch.cat(preds).reshape(args.num_encoders, x.shape[0], -1)
            logits_ = preds.permute(1, 2, 0)
            weighted_logits = torch.matmul(logits_, lstsq_weights)
            loss = criterion(weighted_logits, y).item()
            acc = get_scores(weighted_logits, y, args.test_dataset)
            
    return avg_loss/(iter+1), avg_acc/(iter+1)
                

def get_scores(preds, labels, test_dataset):
    if dataset_info[test_dataset]['mode'] == 'regression':
        return r2_score(labels.flatten().detach().cpu().numpy(), preds.flatten().detach().cpu().numpy())
    elif dataset_info[test_dataset]['mode'] == 'pose_estimation':
        return dist_acc((preds-labels)**2)
    elif dataset_info[test_dataset]['mode'] == 'classification':
        return accuracy(preds, labels, topk=(1, 5))[0]

def dist_acc(dists, thr=0.001):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    # dists = dists.detach().cpu().numpy()
    dist_cal = np.not_equal(dists, 0.0)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1

def get_model_features(loader, split, dataset, arch, model, gpu, output_path = 'features', baseline = False):
    if baseline:
        save_path = os.path.join(output_path, 'baseline', arch+ "_" + split + "_" + dataset)
    else:
        save_path = os.path.join(output_path, arch+ "_" + split + "_" + dataset)
    print("Converting images into features and saving in .pt file for split %s ...." % (split), "in", save_path)

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
                all_labels.append(labels)
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


def load_backbone(args):
    # Model 
    if args.baseline:
        if args.arch == 'resnet50':
            # models_ensemble = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            models_ensemble = resnet50()
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

    if args.baseline:
        if args.arch == 'resnet50':
            models_ensemble.feat_dim = 2048 if args.arch == 'resnet50' else 512
            if dataset_info[args.test_dataset] in ['aloi', 'animal_pose', 'mpii']:
                models_ensemble.global_pool = nn.Identity()
            models_ensemble.fc = nn.Identity()
        elif args.arch == 'convnext':
            models_ensemble.head.fc = nn.Identity()
    else:
        if args.arch == 'resnet50':
            feat_dim = 2048 if args.arch == 'resnet50' else 512
            models_ensemble.base_model.branches_fc = nn.ModuleList([nn.Identity() for i in range(args.num_encoders + 1)])
        elif args.arch == 'convnext':
            for ind in range(args.num_encoders):
                models_ensemble.base_model.head[ind].fc = nn.Identity()
    
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
    models_ensemble = models_ensemble.cuda(args.gpu)
    return models_ensemble