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
from temperature_scaling import cross_validate_temp_scaling, DummyDataset
from timm.models.convnext import convnext_base
from timm.models.resnet import resnet50
from  downstream_utils import get_dataset, get_train_transform, get_val_transform
import wandb
from downstream_utils import *
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import Ridge as LinReg
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, accuracy_score, r2_score
from sklearn.preprocessing import minmax_scale
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
import json
import scipy
from sklearn.model_selection import KFold


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
parser.add_argument('--baseline', action='store_true', help="Use pretrained or QD model")
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
        
    train_loader, val_loader, trainval_loader, test_loader, num_classes = get_feature_datasets(args)
    # train_images, train_labels, val_images, val_labels, test_images, test_labels = train_images.cpu(), train_labels.cpu(), val_images.cpu(), val_labels.cpu(), test_images.cpu(), test_labels.cpu()

    X_train_feature, y_train, X_val_feature, y_val = get_features(
        train_loader, val_loader, models_ensemble, device = args.gpu, baseline = args.baseline, 
        num_classes = num_classes, mode = dataset_info[args.test_dataset]['mode'], test=False
    )

    print(X_train_feature.shape, X_val_feature.shape, y_val.shape)

    X_trainval_feature, y_trainval, X_test_feature, y_test = get_features(
            trainval_loader, test_loader, models_ensemble, device = args.gpu, baseline = args.baseline, 
            num_classes = num_classes, mode = dataset_info[args.test_dataset]['mode'], test=False
        )

    print(X_trainval_feature.shape, y_trainval.shape)

    if args.baseline:
        if dataset_info[args.test_dataset]['mode'] == 'regression':
            clf = LinearRegression(2048, num_classes, 'r2')
        elif dataset_info[args.test_dataset]['mode'] == 'pose_estimation':
            clf = LinearRegression(2048, num_classes, 'pca')
        else:
            clf = LogisticRegression(2048, num_classes, dataset_info[args.test_dataset]['metric'])
    else:
        if dataset_info[args.test_dataset]['mode'] == 'regression':
            clf = [LinearRegression(2048, num_classes, 'r2') for _ in range(args.num_encoders + 1)]
        elif dataset_info[args.test_dataset]['mode'] == 'pose_estimation':
            clf = [LinearRegression(2048, num_classes, 'pca') for _ in range(args.num_encoders + 1)]
        else:
            clf = [LogisticRegression(2048, num_classes, dataset_info[args.test_dataset]['metric']) for _ in range(args.num_encoders + 1)]
        if args.stacking:
            list_ensemble = []
            for i in range(len(clf)):
                list_ensemble.append((str(i), clf[i]))
            if dataset_info[args.test_dataset]['mode'] in ['regression', 'pose_estimation']:     
                stack_clf = StackingModel(classifiers = list_ensemble, mode=dataset_info[args.test_dataset]['mode'],
                                metric=dataset_info[args.test_dataset]['metric'])
            else:
                stack_clf = StackingModel(classifiers = list_ensemble, mode=dataset_info[args.test_dataset]['mode'],
                                metric=dataset_info[args.test_dataset]['metric'])
        
    if dataset_info[args.test_dataset]['mode'] in ['regression', 'pose_estimation']:
        wd_range = torch.logspace(-2, 5, 100)
    else:
        wd_range = torch.logspace(-6, 5, 45)


    if not args.baseline:
        best_params = {}
        best_score = np.zeros(args.num_encoders + 1)
        all_results = {}
        if args.stacking:
            results_file = open(os.path.join("results", args.test_dataset + "_stacking.json"), 'w')
        else:
            results_file = open(os.path.join("results", args.test_dataset + ".json"), 'w')
        for k in range(0, args.num_encoders+1):
            for wd in tqdm(wd_range, desc='Selecting best ridge hyperparameters for classifier' + str(k)):
                clf[k], C = set_params(clf[k], wd, dataset_info[args.test_dataset]['mode'])
                val_acc = clf[k].fit_regression(X_train_feature[k], y_train, X_val_feature[k], y_val)
                if val_acc > best_score[k]:
                    best_params[str(k)] = C
                    best_score[k] = val_acc
                
            print("Best hyper parameter for reg.", best_params[str(k)], best_score)

        all_results['HPO result'] = best_params
        all_results["best val accuracies"] = best_score.tolist()
            
        if not args.stacking:
            print("----------------- Linear combination search ---------------")
            # Using best regulariser find linear combination weights using the val set
            val_preds = []
            for k in range(0, args.num_encoders+1):
                clf[k], _ = set_params(clf[k], torch.tensor(best_params[str(k)]), dataset_info[args.test_dataset]['mode'])
                val_pred = clf[k].get_pred(X_val_feature[k])
                val_preds.append(val_pred)
            lstsq_weights = find_lstsq_weights(val_preds, y_val, num_classes, mode = dataset_info[args.test_dataset]['mode'])
            lstsq_weights = minmax_scale(lstsq_weights)
            all_results['weights'] = lstsq_weights.tolist()
            print(lstsq_weights)

        # From best linear combination and best regulariser, fit classifiers on train val set
        test_preds = []
        test_accuracies = []
        for k in range(0, args.num_encoders+1):
            clf[k], _ = set_params(clf[k], torch.tensor(best_params[str(k)]), dataset_info[args.test_dataset]['mode'])
            test_acc = clf[k].fit_regression(X_trainval_feature[k], y_trainval, X_test_feature[k], y_test)
            test_pred = clf[k].get_pred(X_test_feature[k])
            test_preds.append(test_pred)
            test_accuracies.append(test_acc)
        
        print("All test accuracies", test_accuracies)
        all_results['test accuracies'] = test_accuracies

        if not args.stacking:
            test_preds = np.array(test_preds)
            test_preds = np.swapaxes(test_preds, 0, 2)
            weighted_preds = np.matmul(test_preds, lstsq_weights).squeeze(2)
            weighted_preds = np.transpose(weighted_preds)/sum(lstsq_weights)
            test_acc = clf[0].get_accuracy(weighted_preds, y_test, dataset_info[args.test_dataset]['mode'])
        else:
            test_acc = stack_clf.fit_regression(X_trainval_feature, y_trainval, X_test_feature, y_test)

        all_results['Weighted test acc'] = test_acc
        print("Best test acc", test_acc)
        json.dump(all_results, results_file)

    else:
        best_score = 0.0
        best_params = {}
        results = {}
        # mean_y = torch.tensor(y_test).float().mean(0)
        # mean_y = torch.cat([mean_y for _ in range(y_test.shape[0])]).reshape(y_test.shape[0], -1)
        # print(mean_y, y_test)
        # mean_baseline = r2_score(y_test, mean_y)
        # print("-------------------- MEAN BASELINE ------------------------")
        # print(mean_baseline)
        results_file = open(os.path.join("results", args.test_dataset + "_randombaseline.json"), 'w')
        for wd in tqdm(wd_range, desc='Selecting best hyperparameters for classifier'):
            clf, C = set_params(clf, wd, dataset_info[args.test_dataset]['mode'])
            val_acc = clf.fit_regression(X_train_feature, y_train, X_val_feature, y_val)
            if val_acc > best_score:
                best_score = val_acc
                best_params["C"] = C
            
        
        clf, _ = set_params(clf, torch.tensor(best_params["C"]), dataset_info[args.test_dataset]['mode'])
        test_acc = clf.fit_regression(X_trainval_feature, y_trainval, X_test_feature, y_test)
        # ece, unscaled_ece = evaluate_temp_scaling(clf, X_test_feature, y_test, args.batch_size, dataset_info[args.test_dataset]['metric'])
        results['acc'] = test_acc
        results['best param'] = best_params["C"]
        json.dump(results, results_file)
        print(test_acc, best_params)

def set_params(clf, wd, mode):
    if mode in ['regression', 'pose_estimation']:
        C = wd.item()
        clf.set_params({'alpha': C})
    else:
        C = 1. / wd.item()
        clf.set_params({'C': C})
    return clf, C


def find_lstsq_weights(val_preds, y_val, num_classes, mode):
    val_preds = np.array(val_preds).reshape(len(val_preds), -1)
    val_preds = np.transpose(val_preds)
    if mode == 'classification':
        y_val_ = np.eye(num_classes)[y_val].reshape(-1)
    else:
        y_val_ = y_val.reshape(-1)
    
    lstsq_weights = np.linalg.lstsq(val_preds, y_val_)[0]
    # lstsq_weights = nnls(val_preds, y_val_)[0]
    lstsq_weights = np.expand_dims(lstsq_weights, 1)
    return lstsq_weights

def evaluate_temp_scaling(classifier, X_test_feature, y_test, batch_size, metric):
    orig_model = lambda x: torch.from_numpy(classifier.clf.decision_function(x.cpu().numpy())).to(torch.float32)
    test_dataset = DummyDataset(X_test_feature, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    if metric != 'mAP':
        ece, scaled_ece = cross_validate_temp_scaling(orig_model, test_loader, batch_size)
    else:
        ece, scaled_ece = None, None

    return ece, scaled_ece

# Testing classes and functions
def get_features(train_loader, test_loader, model, device, baseline, num_classes, mode, test=True):
    X_train_feature, y_train = inference(train_loader, model, device, baseline, num_classes, mode, 'train')
    X_test_feature, y_test = inference(test_loader, model, device, baseline, num_classes, mode, 'test' if test else 'val')
    return X_train_feature, y_train, X_test_feature, y_test

def inference(loader, model, device, baseline, num_classes, mode, split):
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
    if mode == 'classification':
        labels_vector = np.array(labels_vector, dtype=int)
    else:
        labels_vector = np.array(labels_vector, dtype=float)

    return feature_vector, labels_vector

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes, metric):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.metric = metric
        self.clf = LogReg(solver='lbfgs', multi_class='multinomial', warm_start=True)

        print('Logistic regression:')
        print(f'\t solver = LBFGS')
        print(f"\t classes = {self.num_classes}")
        print(f"\t metric = {self.metric}")

    def set_params(self, d):
        self.clf.set_params(**d)

    @ignore_warnings(category=ConvergenceWarning)
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

    def get_pred(self, X):
        return self.clf.predict_log_proba(X)

    def get_accuracy(self, y_pred, y_true, mode=None):
        if self.metric == 'accuracy':
            y_pred = y_pred.argmax(1)
            return accuracy_score(y_true, y_pred) * 100.
        else:
            cm = confusion_matrix(y_true, y_pred.argmax(1))
            cm = cm.diagonal() / cm.sum(axis=1) 
            return 100. * cm.mean()

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

        elif self.metric == 'degree_loss':
            self.clf.fit(X_train, y_train)
            pred_test = self.clf.predict(X_test)
            return -(sum(abs(pred_test - y_test))) 
    
    def get_pred(self, X):
        return self.clf.predict(X)

    def get_accuracy(self, y_pred, y_true, mode):
        if mode == 'regression':
            return r2_score(y_true, y_pred) * 100.
        else:
            return dist_acc((y_pred - y_true)**2)

class StackingModel(nn.Module):
    def __init__(self, classifiers, mode, metric, stack_mode='features'):
        super().__init__()
        self.classifiers = classifiers
        self.metric = metric
        self.mode = mode
        if mode == 'classification':
            self.final_estimator = LogReg(solver='lbfgs', multi_class='multinomial', warm_start=True)
        else:
            self.final_estimator = LinearRegression(input_dim=2048*6, output_dim=None, metric=self.metric)
            self.wd_range = torch.logspace(-2, 5, 100)

        self.metric = metric
        self.kfold = KFold(5, shuffle = True, random_state=1)
        self.stack_mode = stack_mode

    def get_preds(self, X_train):
        preds = []
        for id, clf in self.classifiers:
            if self.stack_mode == 'features':
                pred = X_train[:, int(id), :]
            else:
                pred = clf.get_pred(X_train[:, int(id), :])
            preds.append(pred)
        preds = np.transpose(np.array(preds), (1, 0, 2))
        preds = preds.reshape(preds.shape[0], -1)
        return preds

    def fit_regression(self, X_trainval, y_trainval, X_test, y_test):
        # We assume all the base classifiers are tuned with val set before stacking
        X_trainval = np.transpose(X_trainval, (1, 0, 2))
        X_test = np.transpose(X_test, (1, 0, 2))

        best_score = 0.0
        best_params = {}
        # K fold validation
        for train, test in self.kfold.split(X_trainval):
            X_train, y_train = X_trainval[train], y_trainval[train]
            X_val, y_val =  X_trainval[test], y_trainval[test]
            
            # stack predictions
            preds = self.get_preds(X_train)
            val_preds = self.get_preds(X_val)
            trainval_preds = self.get_preds(X_trainval)
            test_preds = self.get_preds(X_test)
            # Fit final estimator 
            for wd in tqdm(self.wd_range, desc='Selecting best hyperparameters for final stacking classifier'):
                self.final_estimator, C = set_params(self.final_estimator, wd, self.mode)
                score = self.final_estimator.fit_regression(preds, y_train, val_preds, y_val)
                if score > best_score:
                    best_score = score
                    best_params["C"] = C

        print(best_params, best_score)
        self.final_estimator, _ = set_params(self.final_estimator, torch.tensor(best_params["C"]), self.mode)
        test_score = self.final_estimator.fit_regression(trainval_preds, y_trainval, test_preds, y_test)
        print("TEST", test_score)



if __name__ == '__main__':
    main()
