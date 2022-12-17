accelerate launch one_method.py --config yamls/KD/bartkd/hps/cola.yaml --common yamls/KD/bartkd/hps/common.yaml
accelerate launch one_method.py --config yamls/KD/bartkd/hps/mrpc.yaml --common yamls/KD/bartkd/hps/common.yaml
accelerate launch one_method.py --config yamls/KD/bartkd/hps/rte.yaml --common yamls/KD/bartkd/hps/common.yaml
accelerate launch one_method.py --config yamls/KD/bartkd/hps/stsb.yaml --common yamls/KD/bartkd/hps/common.yaml