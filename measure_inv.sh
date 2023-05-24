# Baseline 
# python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform resized_crop --gpu 4
# python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform scale --gpu 4
# python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform shear --gpu 4
# python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform rotation --gpu 4
# python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform translation --gpu 4
# python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform deform --gpu 4

python branched/measure_invariances.py  --baseline --moco im1k --pretrained ../moco/moco_v2_800ep_pretrain.pth.tar --data ../../imagenet1k/ --transform resized_crop --gpu 4
python branched/measure_invariances.py  --baseline --moco im1k --pretrained ../moco/moco_v2_800ep_pretrain.pth.tar --data ../../imagenet1k/ --transform scale --gpu 4
python branched/measure_invariances.py  --baseline --moco im1k --pretrained ../moco/moco_v2_800ep_pretrain.pth.tar --data ../../imagenet1k/ --transform rotation --gpu 4
python branched/measure_invariances.py  --baseline --moco im1k --pretrained ../moco/moco_v2_800ep_pretrain.pth.tar --data ../../imagenet1k/ --transform hflip --gpu 4


python branched/measure_invariances.py  --baseline --moco im1k --pretrained ../moco/moco_v2_800ep_pretrain.pth.tar --data ../../imagenet1k/ --transform grayscale --gpu 4 
python branched/measure_invariances.py  --baseline --moco im1k --pretrained ../moco/moco_v2_800ep_pretrain.pth.tar --data ../../imagenet1k/ --transform brightness --gpu 4 --k 4
python branched/measure_invariances.py  --baseline --moco im1k --pretrained ../moco/moco_v2_800ep_pretrain.pth.tar --data ../../imagenet1k/ --transform contrast --gpu 4 --k 4
python branched/measure_invariances.py  --baseline --moco im1k --pretrained ../moco/moco_v2_800ep_pretrain.pth.tar --data ../../imagenet1k/ --transform blur --gpu 4 --k 4
