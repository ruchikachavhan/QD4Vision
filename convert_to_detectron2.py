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
        #print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}
    
    print("Supervised keys:", res["model"].keys())
    return res['model']

    # with open(output_path, "wb") as f:
    #     pkl.dump(res, f)
    #     if obj:
    #         print("Unconverted keys:", obj.keys())
    # except FileNotFoundError as e:
    #     print(f'Could not find model at {input_path}. Skipping {model_name}.')

def convert_to_detectron_qd(model_name, pytorch_dir, detectron_dir):
    """
    This function will convert a model from our
    common PyTorch format into the detectron2 format.
    """
    branch_idx = 5
    print(f'Converting {model_name}')
    input_path = f'{pytorch_dir}/{model_name}.pth.tar'
    output_path = f'{detectron_dir}/{model_name}_{branch_idx}_.pkl'

    # try :
    obj = torch.load(input_path, map_location="cpu")
    obj = obj['state_dict']
    
    newmodel = {}

    for k in list(obj.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_model.'):
            # remove prefix
            obj[k[len("module.base_model."):]] = obj[k]
        # delete renamed or unused k
        del obj[k]

    for k in list(obj.keys()):
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            if t == 4:
                k = k.replace("branches_layer{}.{}".format(t, str(branch_idx)), "res{}".format(t + 1))
            else:
                k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        #print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}

    indxs = [0, 1, 2, 3, 4, 5]
    indxs.remove(branch_idx)
    final_res = {}
    key_list = ()
    for l in indxs:
        if l != branch_idx:
            key_list = key_list + ("branches_layer4.{}".format(l),)

    print(key_list)
    # key_list = ("branches_layer4.0", "branches_layer4.2", "branches_layer4.3", "branches_layer4.4", "branches_layer4.5")
    for k in res['model'].keys():
        # print("C", "branches_layer4.{}".format(t))
        if (k.startswith(key_list)):
            pass
        else:
            final_res[k] = res['model'][k]
    

    final_res['stem.fc.weight'] = res['model']['stem.branches_fc.{}.weight'.format(branch_idx)]
    final_res['stem.fc.bias'] = res['model']['stem.branches_fc.{}.bias'.format(branch_idx)]
    output_res = {}

    for k in final_res.keys():
        if not k.startswith("stem.branches_fc"):
            output_res[k] = final_res[k]

    return output_res
    # print("keys:", output_res.keys())
    # with open(output_path, "wb") as f:
    #     pkl.dump(output_res, f)
    # if obj:
    #     print("Unconverted keys:", obj.keys())
# except FileNotFoundError as e:
#     print(f'Could not find model at {input_path}. Skipping {model_name}.')


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
k1 = convert_to_detectron('supervised', pytorch_dir, detectron_dir)
k2 = convert_to_detectron_qd( 'qd', pytorch_dir, detectron_dir) 
print("Checking keys")
for k in k1.keys():
    if not np.all(k1[k] == k2[k]):
        print(k)  
        