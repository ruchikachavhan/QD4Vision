import torch


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


def convert_qd_models(branch_idx):
    checkpoint_name = "../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar"
    new_ckpt_name = "../qd4vision_icml/models_pytorch/qd_{}_mmseg.pth.tar".format(branch_idx)
    checkpoint = torch.load(checkpoint_name)
    state_dict = checkpoint['state_dict']
    new_state_dict = {}

    indxs = [0, 1, 2, 3, 4, 5]
    key_list = ()
    for l in indxs:
        if l != branch_idx:
            key_list = key_list + ("module.base_model.branches_layer4.{}".format(l),)

    for k, v in state_dict.items():
        if 'fc' not in k:
            if k.startswith('module.base_model.branches_layer4.{}'.format(branch_idx)):
                k_new = 'backbone.' + k.replace('module.base_model.branches_layer4.{}'.format(branch_idx), 'layer4')
                new_state_dict[k_new] = v
            elif k.startswith(key_list):
                pass
            else:
                k_new = 'backbone.' + k.replace('module.base_model.', '')
                new_state_dict[k_new] = v

    torch.save(new_state_dict, new_ckpt_name)

# convert_baseline()
convert_qd_models(0)
convert_qd_models(1)
convert_qd_models(2)
convert_qd_models(3)
convert_qd_models(4)