#!/usr/bin/env python
# coding: utf-8
# adapted from the conversion code in `detectron2/tools/convert-torchvision-to-d2.py`

import sys, os
import pickle as pkl
import numpy as np
import torch

def convert_to_detectron(model_name, pytorch_dir, detectron_dir):
    """
    This function will convert a model from our
    common PyTorch format into the detectron2 format.
    """
    print(f'Converting {model_name}')
    input_path = f'{pytorch_dir}/{model_name}_v2.pth'
    output_path = f'{detectron_dir}/{model_name}.pkl'

    # try :
    obj = torch.load(input_path, map_location="cpu")
    # obj = obj['state_dict']
    print("Original Supervised keys:", obj.keys())
    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}
    
    print("Supervised keys:", res["model"].keys())
    # return res['model']

    # with open(output_path, "wb") as f:
    #     pkl.dump(res, f)
    #     if obj:
    #         print("Unconverted keys:", obj.keys())
    # except FileNotFoundError as e:
    #     print(f'Could not find model at {input_path}. Skipping {model_name}.')

def convert_to_detectron_augself(model_name, pytorch_dir, detectron_dir, baseline=False):

    """
    This function will convert a model from our
    common PyTorch format into the detectron2 format.
    """
    branch_idx = 4
    print(f'Converting {model_name}')
    if baseline:
        input_path = '../moco/resnet50_imagenet100_moco_baseline.pth'
    else:
        input_path = '../saved_models/qd4vision/resnet50_imagenet100_moco_augself.pth'
    output_path = f'{detectron_dir}/{model_name}.pkl'

#     # try :
    obj = torch.load(input_path, map_location="cpu")['backbone']
    # print("Original QD keys:", obj.keys())
    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}
    
    print("Supervised keys:", res["model"].keys())
    with open(output_path, "wb") as f:
        pkl.dump(res, f)
    #     if obj:
    #         print("Unconverted keys:", obj.keys())
    # except FileNotFoundError as e:
    #     print(f'Could not find model at {input_path}. Skipping {model_name}.')

def convert_to_detectron_qd(model_name, pytorch_dir, detectron_dir, input_path):
    """
    This function will convert a model from our
    common PyTorch format into the detectron2 format.
    """
    branch_idx = 0
    print(f'Converting {model_name}')
    output_path = f'{detectron_dir}/{model_name}_{branch_idx}.pkl'

    # try :
    obj = torch.load(input_path, map_location="cpu")['state_dict']
    print("Original QD keys:", obj.keys())
    
    branch_keys = {}
    for k in list(obj.keys()):
        if 'branches' not in k:
            new_k = k.replace('module.base_model.', '')
            branch_keys[new_k] = obj[k]

    for k in list(obj.keys()):
        if "module.base_model.branches_layer4.{}".format(branch_idx) in k:
            new_k = k.replace("module.base_model.branches_layer4.{}".format(branch_idx), "layer4")
            branch_keys[new_k] = obj[k]
        
    # Add the fc keys
    branch_keys['fc.weight'] = obj['module.base_model.branches_fc.{}.weight'.format(branch_idx)]
    branch_keys['fc.bias'] = obj['module.base_model.branches_fc.{}.bias'.format(branch_idx)]

    newmodel = {}
    for k in list(branch_keys.keys()):
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = branch_keys.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}
    print("Supervised keys:", res["model"].keys())

    with open(output_path, "wb") as f:
        pkl.dump(res, f)


def convert_to_detectron_qd_moco(model_name, pytorch_dir, detectron_dir, input_path):
    """
    This function will convert a model from our
    common PyTorch format into the detectron2 format.
    """
    branch_idx = 4
    print(f'Converting {model_name}')
    output_path = f'{detectron_dir}/qd_{model_name}_{branch_idx}.pkl'

    # try :
    obj = torch.load(input_path, map_location="cpu")['state_dict']
    print("Original QD keys:", obj.keys())
    
    branch_keys = {}
    for k in list(obj.keys()):
        if 'branches' not in k and 'encoder_q' in k:
            new_k = k.replace('module.encoder_q.base_model.', '')
            branch_keys[new_k] = obj[k]
            # print(k, new_k)

    # print(branch_keys.keys())

    for k in list(obj.keys()):
        if "module.encoder_q.base_model.branches_layer4.{}".format(branch_idx) in k:
            new_k = k.replace("module.encoder_q.base_model.branches_layer4.{}".format(branch_idx), "layer4")
            branch_keys[new_k] = obj[k]
            print(k, new_k)
        
    # # Add the fc keys
    # print(obj.keys())
    # branch_keys['fc.weight'] = obj['branches_fc.{}.weight'.format(branch_idx)]
    # branch_keys['fc.bias'] = obj['branches_fc.{}.bias'.format(branch_idx)]

    newmodel = {}
    for k in list(branch_keys.keys()):
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = branch_keys.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}
    print("MoCO keys:", res["model"].keys())

    # with open(output_path, "wb") as f:
    #     pkl.dump(res, f)

# For dorsal and ventral models
def convert_to_detectron_moco(model_name, pytorch_dir, detectron_dir, baseline=False):

    """
    This function will convert a model from our
    common PyTorch format into the detectron2 format.
    """
    branch_idx = 4
    print(f'Converting {model_name}')
    input_path = '/raid/s1409650/projects/ssl-invariances/models/{}.pth.tar'.format(model_name)
    output_path = f'{detectron_dir}/{model_name}.pkl'

#     # try :
    obj_init = torch.load(input_path, map_location="cpu")['state_dict']
    # print("Original QD keys:", obj.keys())

    
    # Remove module.encoder_q from keys
    obj = {}
    for k in obj_init.keys():
        if k.startswith("module.encoder_q."):
            obj[k.split("module.encoder_q.")[1]] = obj_init[k]

    print(obj.keys())

    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}
    
    print("Supervised keys:", res["model"].keys())
    with open(output_path, "wb") as f:
        pkl.dump(res, f)

MODELS = [
    # 'supervised',
    'qd',
    # 'simclr-v1',
    # 'simclr-v2',
    # 'moco-v1',
    # 'moco-v2',
    # 'byol',
    # 'swav',
    # 'deepcluster-v2',
    # 'sela-v2',
    # 'infomin',
    # 'insdis',
    # 'pirl',
    # 'pcl-v1',
    # 'pcl-v2'
]

pytorch_dir = 'models_pytorch'
detectron_dir = 'models_pytorch/detectron2'

os.makedirs(pytorch_dir, exist_ok=True)
os.makedirs(detectron_dir, exist_ok=True)

# for model_name in MODELS:
# k1 = convert_to_detectron('supervised', pytorch_dir, detectron_dir)
# k2 = convert_to_detectron_qd( 'qd', pytorch_dir, detectron_dir) 
# convert_to_detectron_augself('augself', pytorch_dir, detectron_dir, baseline = False)
# convert_to_detectron_augself('moco-im100-baseline', pytorch_dir, detectron_dir, baseline = True)
# convert_to_detectron_qd_moco('moco-im1k', pytorch_dir, detectron_dir, '../moco/checkpoimt_0200_good.pth.tar')
# convert_to_detectron_qd_moco('moco-im100', pytorch_dir, detectron_dir, '../moco/imagenet100_checkpoint_0200.pth.tar')
# convert_to_detectron_moco('ventral', pytorch_dir, detectron_dir)
convert_to_detectron_qd('diverse-ensemble', pytorch_dir, detectron_dir, '../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar')

# print("Checking keys")
# for k in k1.keys():
#     if not np.all(k1[k] == k2[k]):
#         print(k)  
        