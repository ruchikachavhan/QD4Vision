# # Baseline 
# python branched/measure_invariances.py  --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --data ../../imagenet1k/ --transform resized_crop --gpu 1
# python branched/measure_invariances.py  --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --data ../../imagenet1k/ --transform h_flip --gpu 1
# python branched/measure_invariances.py  --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --data ../../imagenet1k/ --transform v_flip --gpu 1
# python branched/measure_invariances.py  --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --data ../../imagenet1k/ --transform scale --gpu 1
# python branched/measure_invariances.py  --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --data ../../imagenet1k/ --transform shear --gpu 1
# python branched/measure_invariances.py  --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --data ../../imagenet1k/ --transform shear --gpu 1
# python branched/measure_invariances.py  --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --data ../../imagenet1k/ --transform rotation --gpu 1
# python branched/measure_invariances.py  --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --data ../../imagenet1k/ --transform translation --gpu 1
# python branched/measure_invariances.py  --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --data ../../imagenet1k/ --transform deform --gpu 1

python branched/measure_invariances.py --moco im1k --pretrained moco/checkpoint_0200.pth.tar --data ../../imagenet1k/ --transform resized_crop --gpu 1
python branched/measure_invariances.py --moco im1k --pretrained moco/checkpoint_0200.pth.tar --data ../../imagenet1k/ --transform h_flip --gpu 1
python branched/measure_invariances.py --moco im1k --pretrained moco/checkpoint_0200.pth.tar --data ../../imagenet1k/ --transform scale --gpu 1
python branched/measure_invariances.py --moco im1k --pretrained moco/checkpoint_0200.pth.tar --data ../../imagenet1k/ --transform rotation --gpu 1
