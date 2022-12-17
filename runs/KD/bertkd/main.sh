accelerate launch one_method.py --config yamls/KD/bertkd/main/cola.yaml --common yamls/KD/bertkd/main/common.yaml
accelerate launch one_method.py --config yamls/KD/bertkd/main/mrpc.yaml --common yamls/KD/bertkd/main/common.yaml
accelerate launch one_method.py --config yamls/KD/bertkd/main/rte.yaml --common yamls/KD/bertkd/main/common.yaml
accelerate launch one_method.py --config yamls/KD/bertkd/main/stsb.yaml --common yamls/KD/bertkd/main/common.yaml