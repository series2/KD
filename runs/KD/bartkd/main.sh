accelerate launch one_method.py --config yamls/KD/bartkd/main/cola.yaml --common yamls/KD/bartkd/main/common.yaml
accelerate launch one_method.py --config yamls/KD/bartkd/main/mrpc.yaml --common yamls/KD/bartkd/main/common.yaml
accelerate launch one_method.py --config yamls/KD/bartkd/main/rte.yaml --common yamls/KD/bartkd/main/common.yaml
accelerate launch one_method.py --config yamls/KD/bartkd/main/stsb.yaml --common yamls/KD/bartkd/main/common.yaml