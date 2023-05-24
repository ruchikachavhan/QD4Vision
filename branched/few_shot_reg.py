import random
from argparse import ArgumentParser
from functools import partial
from copy import deepcopy
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn.linear_model import LinearRegression
from downstream_utils import *
import torchvision
from torchvision.datasets import ImageFolder
from PIL import Image
import torchvision.models as torchvision_models
from main_linear import find_lstsq_weights
from sklearn.metrics import r2_score
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
from test_datasets import CelebA, FacesInTheWild300W, LeedsSportsPose, AnimalPose, Causal3DIdent, ALOI, MPII
from main_linear import inference, LinearRegression, set_params

dataset_info = {
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

def get_few_shot_datasets(args):
    train_transform = get_train_transform()
    val_transform = get_val_transform()
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
        
    if args.test_dataset in ['celeba', 'aloi','causal3d']:
        dataset = train_dataset
    else:
        dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset, test_dataset])
    few_shot_size = int(args.shot_size * len(dataset))
    val_size = int(0.2*(len(dataset) - few_shot_size))
    test_size = len(dataset) - few_shot_size - val_size
    # No seed here
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [few_shot_size, val_size, test_size])
    print("DATASET", len(dataset), len(train_dataset), len(val_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.workers)
    
    trainval_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    trainval_loader = DataLoader(trainval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    num_classes = dataset_info[args.test_dataset]['num_classes']

    return train_loader, val_loader, trainval_loader, test_loader, num_classes


def main(args):
    args = parser.parse_args()
    global best_acc1

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    backbone = load_backbone(args)
    if args.baseline:
        results_file = open(os.path.join("results/results_{}".format(args.num_encoders), "{}-moco".format(args.moco) if args.moco is not None else "supervised", "few_shot", args.test_dataset + "_" + str(args.shot_size) + "_baseline.txt"), 'w')
    else:
        results_file = open(os.path.join("results/results_{}".format(args.num_encoders), "{}-moco".format(args.moco) if args.moco is not None else "supervised",  "few_shot", args.test_dataset + "_" + str(args.shot_size) + ".txt"), 'w')
    
    all_accuracies = []
    for t in tqdm(range(args.num_tasks)):
        train_loader, val_loader, trainval_loader, test_loader = get_few_shot_datasets(args)

        X_train_feature, train_labels = inference(train_loader, backbone, args.gpu, 
                        args.baseline, dataset_info[args.test_dataset]['num_classes'],
                        dataset_info[args.test_dataset]['mode'], args.model_type, 'train')
        X_val_feature, val_labels = inference(val_loader, backbone, args.gpu, 
                        args.baseline, dataset_info[args.test_dataset]['num_classes'],
                        dataset_info[args.test_dataset]['mode'], args.model_type, 'val')
        X_test_feature, test_labels = inference(test_loader, backbone, args.gpu, 
                        args.baseline, dataset_info[args.test_dataset]['num_classes'],
                        dataset_info[args.test_dataset]['mode'], args.model_type, 'test')

        wd_range = torch.logspace(-2, 5, 100)
        if args.baseline:
            best_score = 0.0
            best_params = {}
            clf = LinearRegression(2048, dataset_info[args.test_dataset]['num_classes'], dataset_info[args.test_dataset]['metric'])
            for wd in tqdm(wd_range, desc='Selecting best hyperparameters for classifier'):
                C = set_params(clf, torch.tensor([wd]), dataset_info[args.test_dataset]['mode'])
                val_acc = clf.fit_regression(X_train_feature, train_labels, X_val_feature, val_labels)
                if val_acc > best_score:
                    best_score = val_acc
                    best_params["C"] = C

            print("Bestest score", best_score, best_params)
            if best_score < 0:
                best_score = 0.0
                best_params["C"] = 0.0

            print("Best score", best_score, best_params)
            C = set_params(clf, torch.tensor(best_params["C"]), dataset_info[args.test_dataset]['mode'])
            test_score = clf.fit_regression(X_train_feature, train_labels, X_test_feature, test_labels)
            print("test score", test_score)
            all_accuracies.append(test_score)
        else:
            iter = 0
            best_params = {}
            best_score = np.zeros(args.num_encoders + 1)
            clf = [LinearRegression(2048, dataset_info[args.test_dataset]['num_classes'], dataset_info[args.test_dataset]['metric']) for _ in range(args.num_encoders + 1)]
            for k in range(0, args.num_encoders+1):
                for wd in tqdm(wd_range, desc='Selecting best ridge hyperparameters for classifier' + str(k)):
                    C = set_params(clf[k], torch.tensor([wd]), dataset_info[args.test_dataset]['mode'])
                    val_acc = clf[k].fit_regression(X_train_feature[k], train_labels, X_val_feature[k], val_labels)
                    if val_acc > best_score[k]:
                        best_params[str(k)] = C
                        best_score[k] = val_acc
                if best_score[k] < 0.0:
                    best_score[k] = 0.0
                    best_params[str(k)] = 0.0
   
                print("Best hyper parameter for reg.", best_params[str(k)], best_score)
            
            print("----------------- Linear combination search ---------------")
            # Using best regulariser find linear combination weights using the val set
            val_preds = []
            for k in range(0, args.num_encoders+1):
                _ = set_params(clf[k], torch.tensor(best_params[str(k)]), dataset_info[args.test_dataset]['mode'])
                val_pred = clf[k].get_pred(X_val_feature[k])
                val_preds.append(val_pred)
            lstsq_weights = find_lstsq_weights(val_preds, val_labels, dataset_info[args.test_dataset]['num_classes'], mode = dataset_info[args.test_dataset]['mode'])
            lstsq_weights = minmax_scale(lstsq_weights)
            print(lstsq_weights)

            # From best linear combination and best regulariser, fit classifiers on train val set
            test_preds = []
            test_accuracies = []
            for k in range(0, args.num_encoders+1):
                _ = set_params(clf[k], torch.tensor(best_params[str(k)]), dataset_info[args.test_dataset]['mode'])
                test_acc = clf[k].fit_regression(X_train_feature[k], train_labels, X_test_feature[k], test_labels)
                test_pred = clf[k].get_pred(X_test_feature[k])
                test_preds.append(test_pred)
                test_accuracies.append(test_acc)
            
            test_preds = np.array(test_preds)
            test_preds = np.swapaxes(test_preds, 0, 2)
            weighted_preds = np.matmul(test_preds, lstsq_weights).squeeze(2)
            weighted_preds = np.transpose(weighted_preds)/sum(lstsq_weights)
            test_acc = clf[0].get_accuracy(weighted_preds, test_labels, dataset_info[args.test_dataset]['mode'])
            all_accuracies.append(test_acc)

            print("Test accuracy", test_acc)

    avg = np.mean(all_accuracies)
    std = np.std(all_accuracies) * 1.96 / np.sqrt(len(all_accuracies))
    results_file.write("Accuracy, " + str(avg))
    results_file.write("\n")
    results_file.write("Std, " + str(std))
    print("ACCURACY", avg, std)

    
if __name__ == '__main__':
    torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

    model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base', 'convnext'] + torchvision_model_names
    parser = ArgumentParser()
    parser.add_argument('--test_dataset', type=str, default='leeds_sports_pose')
    parser.add_argument('--data_root', type=str, default='../TestDatasets')
    parser.add_argument('--shot-size', type=float, default=0.05)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--num-tasks', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
    parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
    parser.add_argument('--baseline', action='store_true', help="Use resnet or hyper-resnet")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
    parser.add_argument('--num_encoders', default=5, type=int, help='Number of encoders')
    parser.add_argument('--moco',  default='im1k', type=str, help="Use MOCO pretrained model")
    parser.add_argument('--model-type', default='branched', type=str, 
                    help='which model type to use')

    args = parser.parse_args()
    main(args)