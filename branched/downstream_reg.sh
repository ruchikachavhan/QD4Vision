# Regression/Dorsal invariance
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 0 --test_dataset aloi
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 0 --test_dataset animal_pose
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 0 --test_dataset mpii
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --gpu 0 --test_dataset causal3d

# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --gpu 0 --moco --test_dataset aloi
# python branched/main_linear.py --pretrained ../moco/checkpoimt_0200_good.pth.tar --gpu 0 --moco im1k --test_dataset causal3d
# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --gpu 0 --moco --test_dataset animal_pose
# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --gpu 0 --moco --test_dataset mpii

# python branched/main_linear_nn.py --baseline --moco im1k --gpu 1 --test_dataset 300w 
# python branched/main_linear_nn.py --baseline --moco im1k --gpu 1 --test_dataset leeds_sports_pose
# python branched/main_linear_nn.py --baseline --moco im1k --gpu 1 --test_dataset causal3d

# python branched/main_linear_nn.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco im100 --gpu 1 --test_dataset 300w 
# python branched/main_linear_nn.py --pretrained ../moco/checkpoint_0200.pth.tar --moco im1k  --gpu 1 --test_dataset leeds_sports_pose 
# python branched/main_linear_nn.py --pretrained ../moco/checkpoint_0200.pth.tar --moco im1k --gpu 1 --test_dataset causal3d


# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco im100 --gpu 1 --test_dataset aloi 
# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco im100  --gpu 1 --test_dataset animal_pose 
# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco im100 --gpu 1 --test_dataset mpii
# python branched/main_linear.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco im100 --gpu 1 --test_dataset causal3d


# python branched/main_linear_nn.py --baseline --moco --gpu 1 --test_dataset leeds_sports_pose --few-shot-reg --shot-size 0.05
# python branched/main_linear_nn.py --baseline --moco --gpu 1 --test_dataset leeds_sports_pose --few-shot-reg --shot-size 0.1
# python branched/main_linear_nn.py --baseline --moco --gpu 1 --test_dataset leeds_sports_pose --few-shot-reg --shot-size 0.2
# python branched/main_linear_nn.py --pretrained ../moco/checkpoint_0200.pth.tar --moco --gpu 1 --test_dataset leeds_sports_pose --few-shot-reg --shot-size 0.05
# python branched/main_linear_nn.py --pretrained ../moco/checkpoint_0200.pth.tar --moco --gpu 1 --test_dataset leeds_sports_pose --few-shot-reg --shot-size 0.1
# python branched/main_linear_nn.py --pretrained ../moco/checkpoint_0200.pth.tar --moco  --gpu 1 --test_dataset leeds_sports_pose --few-shot-reg --shot-size 0.2

# python branched/main_linear_baseline_ensemble.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar --gpu 5 --test_dataset animal_pose
# python branched/main_linear_baseline_ensemble.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar --gpu 5 --test_dataset mpii
# python branched/main_linear_baseline_ensemble.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar --gpu 5 --test_dataset aloi
# python branched/main_linear_baseline_ensemble.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar --gpu 5 --test_dataset causal3d

# python branched/main_linear.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 4 --test_dataset animal_pose
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 4 --test_dataset mpii
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 4 --test_dataset aloi
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 4 --test_dataset causal3d

# python branched/main_linear.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_6_0020.pth.tar --num_encoders 6 --gpu 5 --test_dataset animal_pose
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_6_0020.pth.tar --num_encoders 6 --gpu 5 --test_dataset mpii
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_6_0020.pth.tar --num_encoders 6 --gpu 5 --test_dataset aloi
# python branched/main_linear.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_6_0020.pth.tar --num_encoders 6 --gpu 5 --test_dataset causal3d --batch-size 64

# python branched/main_linear_nn.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 5 --test_dataset 300w --batch-size 16
# python branched/main_linear_nn.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 5 --test_dataset leeds_sports_pose --batch-size 16
# python branched/main_linear_nn.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 5 --test_dataset celeba --epochs 1
# python branched/main_linear_nn.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_6_0020.pth.tar --num_encoders 6 --gpu 5 --test_dataset 300w --batch-size 16
# python branched/main_linear_nn.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_6_0020.pth.tar --num_encoders 6 --gpu 5 --test_dataset leeds_sports_pose --batch-size 16
# python branched/main_linear_nn.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_6_0020.pth.tar --num_encoders 6 --gpu 5 --test_dataset celeba --epochs 1

python branched/main_linear_nn.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 5 --test_dataset 300w --batch-size 16  --few-shot-reg --shot-size 0.2
python branched/main_linear_nn.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 5 --test_dataset leeds_sports_pose --batch-size 16 --few-shot-reg --shot-size 0.2
python branched/main_linear_nn.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 5 --test_dataset celeba --epochs 1 --few-shot-reg --shot-size 0.2
python branched/main_linear_nn.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_6_0020.pth.tar --num_encoders 6 --gpu 5 --test_dataset 300w --batch-size 16 --few-shot-reg --shot-size 0.2
python branched/main_linear_nn.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_6_0020.pth.tar --num_encoders 6 --gpu 5 --test_dataset leeds_sports_pose --batch-size 16 --few-shot-reg --shot-size 0.2
python branched/main_linear_nn.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_6_0020.pth.tar --num_encoders 6 --gpu 5 --test_dataset celeba --epochs 1 --few-shot-reg --shot-size 0.2


