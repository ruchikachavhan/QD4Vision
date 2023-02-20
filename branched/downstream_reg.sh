# Regression/Dorsal invariance
python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 3 --test_dataset aloi
python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 3 --test_dataset animal_pose
python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 3 --test_dataset mpii
python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 3 --test_dataset causal3d