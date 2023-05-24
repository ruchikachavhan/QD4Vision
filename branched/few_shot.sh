# python branched/few_shot.py --baseline --tta  --gpu 1 --datadir ../TestDatasets/ --num-tasks 20 --K 1 --test_dataset flowers
# python branched/few_shot.py --baseline --tta  --gpu 1 --datadir ../TestDatasets/ --num-tasks 20 --K 5 --test_dataset flowers
# python branched/few_shot.py --baseline --tta  --gpu 1 --datadir ../TestDatasets/ --num-tasks 20 --K 1 --test_dataset plant_disease
# python branched/few_shot.py --baseline --tta --gpu 1 --datadir ../TestDatasets/ --num-tasks 20 --K 5 --test_dataset plant_disease
# python branched/few_shot.py --baseline --tta  --gpu 1 --datadir ../TestDatasets/ --num-tasks 20 --K 1 --test_dataset cub200
# python branched/few_shot.py --baseline --tta --gpu 1 --datadir ../TestDatasets/ --num-tasks 20 --K 5 --test_dataset cub200


# python branched/few_shot.py --baseline --pretrained ../moco/moco_v2_800ep_pretrain.pth.tar --moco  im1k --gpu 1 --datadir ../TestDatasets/ --num-tasks 2000 --K 1 --test_dataset flowers
# python branched/few_shot.py --baseline --pretrained ../moco/moco_v2_800ep_pretrain.pth.tar --moco  im1k --gpu 1 --datadir ../TestDatasets/ --num-tasks 2000 --K 5 --test_dataset flowers
# python branched/few_shot.py --baseline --pretrained ../moco/moco_v2_800ep_pretrain.pth.tar --moco  im1k --gpu 1 --datadir ../TestDatasets/ --num-tasks 2000 --K 1 --test_dataset plant_disease
# python branched/few_shot.py --baseline --pretrained ../moco/moco_v2_800ep_pretrain.pth.tar --moco im1k --gpu 1 --datadir ../TestDatasets/ --num-tasks 2000 --K 5 --test_dataset plant_disease
# python branched/few_shot.py --baseline --pretrained ../moco/moco_v2_800ep_pretrain.pth.tar --moco im1k  --gpu 1 --datadir ../TestDatasets/ --num-tasks 2000 --K 1 --test_dataset cub200
# python branched/few_shot.py --baseline --pretrained ../moco/moco_v2_800ep_pretrain.pth.tar --moco  im1k --gpu 1 --datadir ../TestDatasets/ --num-tasks 2000 --K 5 --test_dataset cub200


# python branched/few_shot.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco  im100 --gpu 1 --datadir ../TestDatasets/ --num-tasks 2000 --K 1 --test_dataset flowers
# python branched/few_shot.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco  im100 --gpu 1 --datadir ../TestDatasets/ --num-tasks 2000 --K 5 --test_dataset flowers
# python branched/few_shot.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco  im100 --gpu 1 --datadir ../TestDatasets/ --num-tasks 2000 --K 1 --test_dataset plant_disease
# python branched/few_shot.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco im100 --gpu 1 --datadir ../TestDatasets/ --num-tasks 2000 --K 5 --test_dataset plant_disease
# python branched/few_shot.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco im100 --gpu 1 --datadir ../TestDatasets/ --num-tasks 2000 --K 1 --test_dataset cub200
# python branched/few_shot.py ---pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco im100 --gpu 1 --datadir ../TestDatasets/ --num-tasks 2000 --K 5 --test_dataset cub200
# python branched/few_shot.py --pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco im100 --gpu 1 --datadir ../TestDatasets/ --num-tasks 2000 --K 1 --test_dataset fc100
# python branched/few_shot.py ---pretrained ../moco/imagenet100_checkpoint_0200.pth.tar --moco im100 --gpu 1 --datadir ../TestDatasets/ --num-tasks 2000 --K 5 --test_dataset fc100

# python branched/few_shot.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar --gpu 0 --datadir ../TestDatasets/ --num-tasks 2000 --K 1 --test_dataset flowers
# python branched/few_shot.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar --gpu 0 --datadir ../TestDatasets/ --num-tasks 2000 --K 5 --test_dataset flowers
# python branched/few_shot.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar --gpu 0 --datadir ../TestDatasets/ --num-tasks 2000 --K 1 --test_dataset plant_disease
# python branched/few_shot.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar --gpu 0 --datadir ../TestDatasets/ --num-tasks 2000 --K 5 --test_dataset plant_disease
# python branched/few_shot.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar --gpu 0 --datadir ../TestDatasets/ --num-tasks 2000 --K 1 --test_dataset cub200
# python branched/few_shot.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar --gpu 0 --datadir ../TestDatasets/ --num-tasks 2000 --K 5 --test_dataset cub200
# python branched/few_shot.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar --gpu 0 --datadir ../TestDatasets/ --num-tasks 2000 --K 1 --test_dataset fc100
# python branched/few_shot.py --pretrained ../saved_models/qd4vision/resnet50_imagenet1k_branched-supervised6_checkpoint_diverse_baseline_0020.pth.tar --gpu 0 --datadir ../TestDatasets/ --num-tasks 2000 --K 5 --test_dataset fc100

# python branched/few_shot.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 0 --datadir ../TestDatasets/ --num-tasks 2000 --K 1 --test_dataset fc100
python branched/few_shot.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 0 --datadir ../TestDatasets/ --num-tasks 2000 --K 5 --test_dataset fc100
python branched/few_shot.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 0 --datadir ../TestDatasets/ --num-tasks 2000 --K 1 --test_dataset plant_disease
python branched/few_shot.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 0 --datadir ../TestDatasets/ --num-tasks 2000 --K 5 --test_dataset plant_disease
python branched/few_shot.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 0 --datadir ../TestDatasets/ --num-tasks 2000 --K 1 --test_dataset cub200
python branched/few_shot.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 0 --datadir ../TestDatasets/ --num-tasks 2000 --K 5 --test_dataset cub200
python branched/few_shot.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 0 --datadir ../TestDatasets/ --num-tasks 2000 --K 1 --test_dataset flowers
python branched/few_shot.py --pretrained ../saved_models/qd4vision/ablation/resnet50_imagenet1k_branched-supervised6_checkpoint_3_0020.pth.tar --num_encoders 3 --gpu 0 --datadir ../TestDatasets/ --num-tasks 2000 --K 5 --test_dataset flowers



