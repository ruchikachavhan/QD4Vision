# Baseline 
python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform resized_crop --gpu 4
python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform scale --gpu 4
python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform shear --gpu 4
python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform rotation --gpu 4
python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform translation --gpu 4
python branched/measure_invariances.py  --baseline --data ../../imagenet1k/ --transform deform --gpu 4