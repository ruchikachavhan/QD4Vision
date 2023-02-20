python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform grayscale --gpu 3 
python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform brightness --gpu 3 --k 4
python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform contrast --gpu 3 --k 4
python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform saturation --gpu 3 --k 4
python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform hue --gpu 3 --k 4
python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform blur --gpu 3 --k 4
python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform sharpness --gpu 3 --k 4
python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform equalize --gpu 3 
python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform invert --gpu 3
python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform posterize --gpu 3 