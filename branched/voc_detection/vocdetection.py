import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.models as models
import torch.optim as optim
import numpy as np
import torchvision.datasets as datasets
import random
import os
import urllib.request
# from skimage import io
import shutil
from tqdm import tqdm
import tarfile
import requests
import xml.etree.ElementTree as ET
import cv2
import wandb
from voc_utils import *
from models import RCNN, Validator, ValDataset, RCNN_Trainer

config={'image_size':224, 'n_classes':21, 'bbox_reg': True, 'network': 'efficientnet-b0', 'max_proposals':2000, 'pad': 16}
train_config={'log_wandb':True, 'logging': ['plot'],
              'epochs': 3, 'batch_size':128, 'lr': 0.001, 'lr_decay':0.5, 'l2_reg': 1e-5}
train_config_classifer={'log_wandb':True, 'logging': ['plot'],
              'epochs': 1, 'batch_size':128, 'lr': 0.001, 'lr_decay':0.5, 'l2_reg': 1e-5}

voc_2012_classes=['background','Aeroplane',"Bicycle",'Bird',"Boat","Bottle","Bus","Car","Cat","Chair",'Cow',"Dining table","Dog","Horse","Motorbike",'Person', "Potted plant",'Sheep',"Sofa","Train","TV/monitor"]

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],std = [ 1., 1., 1. ]), 
                                transforms.ToPILImage() ])

def main():
    voc_dataset=VOC2012()

    val_datalen=len(os.listdir(voc_dataset.test_dir+'/Annotations'))
    train_datalen=len(os.listdir(voc_dataset.train_dir+'/Annotations'))
    print(train_datalen, val_datalen) # 17,125 total

    loader=RCNN_DatasetLoader(voc_dataset, config, train_config)    # `loader` is an instance of RCNN_dataset
    classifier_dataloader=RCNN_classifier_DatasetLoader(voc_dataset, config, train_config)    # `loader` is an instance of RCNN_dataset

    val_dataset = ValDataset(voc_dataset)
    model = RCNN(config, load_path = 'RCNN_checkpoint.pt').to('cuda') # 'RCNN_checkpoint.pt'
    validator = Validator(model, val_dataset)

    trainer=RCNN_Trainer(model, loader, None, 'cuda', validator)
    trainer.fine_tuning(train_config)

if __name__ == '__main__':
    main()
