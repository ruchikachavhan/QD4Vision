import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps
import math
import random
import itertools
import numpy as np
from PIL import Image, ImageFilter

class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Edges(object):
    """Canny Edges for images"""
    def __init__(self):
        self.convert_grayscale = True
    def __call__(self, x):
        # Converting the image to grayscale, as edge detection 
        # requires input image to be of mode = Grayscale (L)
        x = x.convert("L")
        # Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
        x = x.filter(ImageFilter.FIND_EDGES).convert('RGB')
        return x

img_size = 224
# Normalize
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

# Set of default augmentations
default_augmentations_edges = [
    transforms.RandomResizedCrop(img_size, scale=(0.08, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
    ], p=1.0),
    transforms.RandomGrayscale(p=1.0),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
]

class CropsTransform:
    """Returns two random crops of one image for each type of augmentation"""

    def __init__(self, augs_list, k):
        self.augs_list = []
        for k in range(1, len(augs_list) - 2):
            self.augs_list += list(itertools.combinations(augs_list[:-2], k)) 
        self.augs_list = [(transforms.Resize((img_size, img_size)), ) + self.augs_list[i]+ (transforms.ToTensor(), normalize) for i in range (len(self.augs_list))]
        self.k = k

    def __call__(self, x):
        index = np.random.randint(len(self.augs_list))
        aug = self.augs_list[index]
        t = transforms.Compose(aug)
        return t(x)

class CropsTransform_all:
    """Returns two random crops of one image for each type of augmentation"""

    def __init__(self, augs_list, k):
        self.augs_list = augs_list
        self.augs_list = [(transforms.Resize((img_size, img_size)), ) + (self.augs_list[i], ) + (transforms.ToTensor(), normalize) for i in range (len(self.augs_list) - 2)]
        self.augs_list.append([transforms.Resize((img_size, img_size)), transforms.ToTensor(), normalize])
        self.k = k

    def __call__(self, x):
        outputs = []
        for i in range(0, len(self.augs_list)):
            t = transforms.Compose(self.augs_list[i])
            outputs.append(t(x))
        outputs = torch.cat(outputs, dim = 0).reshape(len(self.augs_list), 3, img_size, img_size)
        return outputs
