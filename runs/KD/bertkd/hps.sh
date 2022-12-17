accelerate launch one_method.py --config yamls/KD/bertkd/hps/cola.yaml --common yamls/KD/bertkd/hps/common.yaml
accelerate launch one_method.py --config yamls/KD/bertkd/hps/mrpc.yaml --common yamls/KD/bertkd/hps/common.yaml
accelerate launch one_method.py --config yamls/KD/bertkd/hps/rte.yaml --common yamls/KD/bertkd/hps/common.yaml
accelerate launch one_method.py --config yamls/KD/bertkd/hps/stsb.yaml --common yamls/KD/bertkd/hps/common.yaml