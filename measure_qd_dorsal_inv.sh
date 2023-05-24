# python branched/measure_invariances.py  --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --data ../../imagenet1k/ --transform grayscale --gpu 3 
# python branched/measure_invariances.py  --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --data ../../imagenet1k/ --transform brightness --gpu 3 --k 4
# python branched/measure_invariances.py  --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --data ../../imagenet1k/ --transform contrast --gpu 3 --k 4
# python branched/measure_invariances.py  --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --data ../../imagenet1k/ --transform saturation --gpu 3 --k 4
# python branched/measure_invariances.py  --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --data ../../imagenet1k/ --transform hue --gpu 3 --k 4
# python branched/measure_invariances.py  --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --data ../../imagenet1k/ --transform blur --gpu 3 --k 4
# python branched/measure_invariances.py  --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --data ../../imagenet1k/ --transform sharpness --gpu 3 --k 4
# python branched/measure_invariances.py  --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --data ../../imagenet1k/ --transform equalize --gpu 3 
# python branched/measure_invariances.py  --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --data ../../imagenet1k/ --transform invert --gpu 3
# python branched/measure_invariances.py  --pretrained ../saved_models/qd4vision/resnet50_imagenet1k-supervised5_checkpoint_kl_0100_good1.pth.tar --data ../../imagenet1k/ --transform posterize --gpu 3 


python branched/measure_invariances.py  --moco im1k --pretrained moco/checkpoint_0200.pth.tar --data ../../imagenet1k/ --transform grayscale --gpu 1 
python branched/measure_invariances.py --moco im1k  --pretrained moco/checkpoint_0200.pth.tar --data ../../imagenet1k/ --transform brightness --gpu 1 --k 4
python branched/measure_invariances.py  --moco im1k --pretrained moco/checkpoint_0200.pth.tar --data ../../imagenet1k/ --transform contrast --gpu 1 --k 4
python branched/measure_invariances.py --moco im1k  --pretrained moco/checkpoint_0200.pth.tar --data ../../imagenet1k/ --transform blur --gpu 1 --k 4