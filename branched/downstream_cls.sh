# Classification
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 5 --test_dataset CIFAR10
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 5 --test_dataset CIFAR100
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 5 --test_dataset OxfordFlowers102
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 5 --test_dataset Caltech101
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 5 --test_dataset DTD
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 5 --test_dataset StanfordCars
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 5 --test_dataset Aircraft

# OOD testing
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 5 --test_mode ood --test_dataset imagenet-a
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 5 --test_mode ood --test_dataset imagenet-sketch
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 5 --test_mode ood --test_dataset living17
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 5 --test_mode ood --test_dataset entity30

# MOCO
# python branched/main_linear.py --pretrained ../moco/checkpoint_0200.pth.tar --gpu 0 --moco --test_dataset Caltech101
# python branched/main_linear.py --pretrained ../moco/checkpoint_0200.pth.tar --gpu 0 --moco --test_dataset StanfordCars
# python branched/main_linear.py --pretrained ../moco/checkpoint_0200.pth.tar --gpu 0 --moco --test_dataset CIFAR10
# python branched/main_linear.py --pretrained ../moco/checkpoint_0200.pth.tar --gpu 0 --moco --test_dataset CIFAR100
# python branched/main_linear.py --pretrained ../moco/checkpoint_0200.pth.tar --gpu 0 --moco --test_dataset DTD

# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --gpu 0 --moco im100 --test_dataset Aircraft
# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --gpu 0 --moco im100 --test_dataset OxfordFlowers102
# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --gpu 0 --moco im100  --test_dataset Caltech101
# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --gpu 0 --moco im100  --test_dataset StanfordCars
# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --gpu 0 --moco im100  --test_dataset CIFAR10
# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar--gpu 0 --moco im100  --test_dataset CIFAR100
# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --gpu 0 --moco im100  --test_dataset DTD



# python branched/main_linear.py --model-type adapters --pretrained ../saved_models/qd4vision/resnet50_imagenet100_adapters-supervised5_checkpoint_kl_layer12_0200.pth.tar --gpu 4 --test_dataset Aircraft

# python branched/main_linear_baseline_ensemble.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar --gpu 5 --test_dataset CIFAR10
# python branched/main_linear_baseline_ensemble.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar --gpu 5 --test_dataset CIFAR100
# python branched/main_linear_baseline_ensemble.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar --gpu 5 --test_dataset Aircraft
# python branched/main_linear_baseline_ensemble.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar --gpu 5 --test_dataset StanfordCars
# python branched/main_linear_baseline_ensemble.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar --gpu 5 --test_dataset OxfordFlowers102
# python branched/main_linear_baseline_ensemble.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar --gpu 5 --test_dataset Caltech101

python branched/main_linear.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 5 --test_dataset CIFAR10
python branched/main_linear.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 5 --test_dataset CIFAR100
python branched/main_linear.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 5 --test_dataset Caltech101
python branched/main_linear.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 5 --test_dataset Aircraft
python branched/main_linear.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 5 --test_dataset StanfordCars


python branched/main_linear.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_6_0020.pth.tar --num_encoders 6 --gpu 5 --test_dataset CIFAR10
python branched/main_linear.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_6_0020.pth.tar --num_encoders 6 --gpu 5 --test_dataset CIFAR100
python branched/main_linear.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_6_0020.pth.tar --num_encoders 6 --gpu 5 --test_dataset Caltech101
python branched/main_linear.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_6_0020.pth.tar --num_encoders 6 --gpu 5 --test_dataset Aircraft
python branched/main_linear.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_6_0020.pth.tar --num_encoders 6 --gpu 5 --test_dataset DTD
python branched/main_linear.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_6_0020.pth.tar --num_encoders 6 --gpu 5 --test_dataset StanfordCars