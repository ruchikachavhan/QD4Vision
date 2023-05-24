# OOD testing
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 3 --test_mode ood --test_dataset imagenet-a --data_root ../robust-imagenets
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 3 --test_mode ood --test_dataset imagenet-sketch --data_root ../robust-imagenets
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 3 --test_mode ood --test_dataset living17
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 3 --test_mode ood --test_dataset entity30

# MOCO
python branched/main_linear.py --pretrained ../moco/checkpoint_0200.pth.tar --moco im1k --gpu 1 --test_mode ood --test_dataset cifarstl
python branched/main_linear.py --pretrained ../moco/checkpoint_0200.pth.tar --moco im1k --gpu 1 --test_mode ood --test_dataset imagenet-a --data_root ../robust-imagenets
python branched/main_linear.py --pretrained ../moco/checkpoint_0200.pth.tar --moco im1k --gpu 1 --test_mode ood --test_dataset imagenet-r  --data_root ../robust-imagenets
python branched/main_linear.py --pretrained ../moco/checkpoint_0200.pth.tar --moco im1k --gpu 1 --test_mode ood --test_dataset domainnet
python branched/main_linear.py --pretrained ../moco/checkpoint_0200.pth.tar --moco im1k --gpu 1 --test_mode ood --test_dataset imagenet-sketch --data_root ../robust-imagenets


# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco im100 --gpu 1 --test_mode ood --test_dataset domainnet
# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco im100 --gpu 1 --test_mode ood --test_dataset cifarstl
# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco im100 --gpu 1 --test_mode ood --test_dataset imagenet-a --data_root ../robust-imagenets
# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco im100 --gpu 1 --test_mode ood --test_dataset imagenet-r --data_root ../robust-imagenets
# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco im100 --gpu 1 --test_mode ood --test_dataset imagenet-sketch --data_root ../robust-imagenets
# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco im100 --gpu 1 --test_mode ood --test_dataset living17
# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco im100 --gpu 1 --test_mode ood --test_dataset entity30

# python branched/main_linear_nn.py --pretrained ../moco/checkpoint_0200.pth.tar --moco im100 --gpu 1 --test_mode ood --test_dataset domainnet
