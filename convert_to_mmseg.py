import torch
import os

def convert_baseline():
    checkpoint_name = "../qd4vision_icml/models_pytorch/supervised_v2.pth"
    new_ckpt_name = "../qd4vision_icml/models_pytorch/supervised_v2_mmseg.pth"
    checkpoint = torch.load(checkpoint_name)
    state_dict = checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'fc' not in k:
            k_new = 'backbone.' + k
            new_state_dict[k_new] = v
    torch.save(new_state_dict, new_ckpt_name)

def convert_augself(baseline):
    if baseline:
        checkpoint_name = "../moco/resnet50_imagenet100_moco_baseline.pth"
        new_ckpt_name = "../qd4vision_icml/models_pytorch/moco-im100-baseline_mmseg.pth"
    else:
        checkpoint_name = "../saved_models/qd4vision/resnet50_imagenet100_moco_augself.pth"
        new_ckpt_name = "../qd4vision_icml/models_pytorch/augself_v2_mmseg.pth"
    checkpoint = torch.load(checkpoint_name)['backbone']
    state_dict = checkpoint
    # print(state_dict.keys())
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'fc' not in k:
            k_new = 'backbone.' + k
            new_state_dict[k_new] = v
    torch.save(new_state_dict, new_ckpt_name)

def convert_moco1k_baseline(path, model_name):
    checkpoint_name = os.path.join(path, model_name + ".pth.tar")
    # "../moco/moco_v2_800ep_pretrain.pth.tar"
    new_ckpt_name = "../qd4vision_icml/models_pytorch/{}_mmseg.pth".format(model_name)
  
    checkpoint = torch.load(checkpoint_name)['state_dict']
    state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if 'fc' not in k and 'queue' not in k:
            k_new = 'backbone.' + k[len('module.encoder_q.'):]
            new_state_dict[k_new] = v

    print(new_state_dict.keys())
    torch.save(new_state_dict, new_ckpt_name)

def convert_qd_models(branch_idx, dataset='imagenet100', pretrain='moco'):
    if pretrain == 'moco':
        if dataset == 'imagenet1k':
            checkpoint_name = '../moco/checkpoimt_0200_good.pth.tar'
        elif dataset == 'imagenet100':
            checkpoint_name = '../moco/imagenet100_checkpoint_0200.pth.tar'
        new_ckpt_name = "../qd4vision_icml/models_pytorch/qd_moco_{}_{}_mmseg.pth.tar".format(dataset, branch_idx)
    else:
        # checkpoint_name = "../saved_models/qd4vision/resnet50_{}-supervised5_checkpoint_kl_0100_good1.pth.tar".format(dataset)
        checkpoint_name = "../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar"
        new_ckpt_name = "../qd4vision_icml/models_pytorch/{}_{}_mmseg.pth.tar".format(dataset, branch_idx)
    checkpoint = torch.load(checkpoint_name)
    state_dict = checkpoint['state_dict']
    print(state_dict.keys())
    new_state_dict = {}

    indxs = [0, 1, 2, 3, 4, 5]
    key_list = ()
    for l in indxs:
        if l != branch_idx:
            key_list = key_list + ("module.base_model.branches_layer4.{}".format(l),)

    for k, v in state_dict.items():
        if 'fc' not in k and 'encoder_k' not in k and 'queue' not in k:
            if k.startswith('module.base_model.branches_layer4.{}'.format(branch_idx)):
                k_new = 'backbone.' + k.replace('module.base_model.branches_layer4.{}'.format(branch_idx), 'layer4')
                new_state_dict[k_new] = v
                print(k_new, k, v.shape)
            elif k.startswith(key_list):
                pass
            else:
                k_new = 'backbone.' + k.replace('module.base_model.', '')
                new_state_dict[k_new] = v
                print(k_new, k, v.shape)

    print(new_state_dict.keys())
    torch.save(new_state_dict, new_ckpt_name)

# convert_baseline()
# convert_qd_models(0)
# convert_qd_models(1)
# convert_qd_models(2)
# convert_qd_models(3)
# convert_qd_models(4)
# convert_augself(baseline=True)
# convert_moco1k_baseline()
# convert_moco1k_baseline(path = '/raid/s1409650/projects/ssl-invariances/models/', model_name='dorsal')
# convert_moco1k_baseline(path = '/raid/s1409650/projects/ssl-invariances/models/', model_name='ventral')
convert_qd_models(0, dataset='diverse-ensemble', pretrain='sup')