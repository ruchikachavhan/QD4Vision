python branched/few_shot.py --baseline --tta  --gpu 1 --datadir ../TestDatasets/ --num-tasks 20 --K 1 --test_dataset flowers
python branched/few_shot.py --baseline --tta  --gpu 1 --datadir ../TestDatasets/ --num-tasks 20 --K 5 --test_dataset flowers
python branched/few_shot.py --baseline --tta  --gpu 1 --datadir ../TestDatasets/ --num-tasks 20 --K 1 --test_dataset plant_disease
python branched/few_shot.py --baseline --tta --gpu 1 --datadir ../TestDatasets/ --num-tasks 20 --K 5 --test_dataset plant_disease
python branched/few_shot.py --baseline --tta  --gpu 1 --datadir ../TestDatasets/ --num-tasks 20 --K 1 --test_dataset cub200
python branched/few_shot.py --baseline --tta --gpu 1 --datadir ../TestDatasets/ --num-tasks 20 --K 5 --test_dataset cub200