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
from sklearn.linear_model import LogisticRegression
from downstream_utils import *
from torchvision.datasets import ImageFolder
from PIL import Image
import torchvision.models as torchvision_models
from main_linear import find_lstsq_weights
from sklearn.metrics import confusion_matrix, precision_recall_curve, accuracy_score, r2_score
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

def load_fewshot_datasets(dataset='cifar10',
                          datadir='/data',
                          pretrain_data='imagenet100'):

    if pretrain_data == 'imagenet100':
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        transform = T.Compose([T.Resize(224, interpolation=Image.BICUBIC),
                               T.CenterCrop(224),
                               T.ToTensor(),
                               T.Normalize(mean, std)])

    elif pretrain_data == 'stl10':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        transform = T.Compose([T.Resize(96, interpolation=Image.BICUBIC),
                               T.CenterCrop(96),
                               T.ToTensor(),
                               T.Normalize(mean, std)])

    if dataset == 'cub200':
        train = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        test  = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        test.samples = train.samples + test.samples
        test.targets = train.targets + test.targets

    elif dataset == 'fc100':
        train = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        test  = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        test.samples = train.samples + test.samples
        test.targets = train.targets + test.targets

    elif dataset == 'plant_disease':
        train = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        test  = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        test.samples = train.samples + test.samples
        test.targets = train.targets + test.targets

    return dict(test=test)


class FewShotBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, N, K, Q, num_iterations):
        self.N = N
        self.K = K
        self.Q = Q
        self.num_iterations = num_iterations

        labels = [label for _, label in dataset.samples]
        self.label2idx = defaultdict(list)
        for i, y in enumerate(labels):
            self.label2idx[y].append(i)

        few_labels = [y for y, indices in self.label2idx.items() if len(indices) <= self.K]
        for y in few_labels:
            del self.label2idx[y]

    def __len__(self):
        return self.num_iterations

    def __iter__(self):
        label_set = set(list(self.label2idx.keys()))
        for _ in range(self.num_iterations):
            labels = random.sample(label_set, self.N)
            indices = []
            for y in labels:
                if len(self.label2idx[y]) >= self.K+self.Q:
                    indices.extend(list(random.sample(self.label2idx[y], self.K+self.Q)))
                else:
                    tmp_indices = [i for i in self.label2idx[y]]
                    random.shuffle(tmp_indices)
                    indices.extend(tmp_indices[:self.K] + np.random.choice(tmp_indices[self.K:], size=self.Q).tolist())
            yield indices

def main(args):
    cudnn.benchmark = True
    device = args.gpu

    # DATASETS
    datasets = load_fewshot_datasets(dataset=args.dataset,
                                     datadir=args.datadir)
    print(datasets['test'])
    if not args.baseline:
        args.Q = 2 * args.Q
    build_sampler    = partial(FewShotBatchSampler,
                               N=args.N, K=args.K, Q=args.Q, num_iterations=args.num_tasks)
    build_dataloader = partial(torch.utils.data.DataLoader,
                               num_workers=args.num_workers)
    testloader  = build_dataloader(datasets['test'],  batch_sampler=build_sampler(datasets['test']))

    backbone = load_backbone(args)
    backbone.eval()

    all_accuracies = []
    if args.baseline:
        results_file = open(os.path.join("results", "few_shot", args.dataset + "_" + args.K + "_baseline.txt"), 'w')
    else:
        results_file = open(os.path.join("results", "few_shot", args.dataset + "_" + args.K + ".txt"), 'w')
    for i, (batch, _) in tqdm(enumerate(testloader)):
        with torch.no_grad():
            batch = batch.to(device)
            B, C, H, W = batch.shape
            batch = batch.view(args.N, args.K+args.Q, C, H, W)

            train_batch  = batch[:, :args.K].reshape(args.N*args.K, C, H, W)
            test_batch   = batch[:, args.K:].reshape(args.N*args.Q, C, H, W)
            train_labels = torch.arange(args.N).unsqueeze(1).repeat(1, args.K).to(device).view(-1)
            test_labels  = torch.arange(args.N).unsqueeze(1).repeat(1, args.Q).to(device).view(-1)

        with torch.no_grad():
            if args.baseline:
                X_train = backbone(train_batch)
                Y_train = train_labels

                X_test = backbone(test_batch)
                Y_test = test_labels
            else:
                _, X_train = backbone(train_batch, reshape = False)
                Y_train = train_labels

                _, X_test = backbone(test_batch, reshape = False)
                Y_test = test_labels

        if args.baseline:
            classifier = LogisticRegression(solver='liblinear').fit(X_train.cpu().numpy(),
                                                                        Y_train.cpu().numpy())
            acc = classifier.score(X_test.cpu().numpy(), Y_test.cpu().numpy())
            all_accuracies.append(acc)
        else:
            spt_preds = []
            qry_preds= []
            for k in range(0, args.num_encoders + 1):
                classifier = LogisticRegression(solver='liblinear').fit(X_train[k].cpu().numpy(),
                                                                        Y_train.cpu().numpy())
                spt_pred = classifier.predict_log_proba(X_train[k].cpu().numpy())
                qry_pred = classifier.predict_log_proba(X_test[k].cpu().numpy())
                spt_preds.append(spt_pred)
                qry_preds.append(qry_pred)

            lstsq_weights = find_lstsq_weights(qry_pred[0:args.Q//2], Y_test[0:args.Q//2].cpu().numpy(), args.N, mode='classification')
            lstsq_weights = minmax_scale(lstsq_weights)
            qry_preds = np.array(qry_preds)
            qry_preds = np.swapaxes(qry_preds, 0, 2)

            weighted_preds = np.matmul(qry_preds, lstsq_weights).squeeze(2)
            weighted_preds = np.transpose(weighted_preds)/sum(lstsq_weights)
            acc = accuracy_score(weighted_preds.argmax(1), Y_test.cpu().numpy())        
            all_accuracies.append(acc)

    avg = np.mean(all_accuracies)
    std = np.std(all_accuracies) * 1.96 / np.sqrt(len(all_accuracies))
    results_file.write("Accuracy, " + str(acc))
    results_file.write("\n")
    results_file.write("Std, " + str(std))
    print("ACCURACY", acc, std)
    # logger.log_msg(f'mean: {avg:.4f}±{std:.4f}')


if __name__ == '__main__':
    torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

    model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base', 'convnext'] + torchvision_model_names
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cub200')
    parser.add_argument('--datadir', type=str, default='/data')
    parser.add_argument('--N', type=int, default=5)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--Q', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-tasks', type=int, default=2000)
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

    args = parser.parse_args()
    # args.num_backbone_features = 512 if args.model.endswith('resnet18') else 2048
    main(args)
