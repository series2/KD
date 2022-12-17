accelerate launch one_method.py --config yamls/RAIL/c-bert/hps/cola.yaml --common yamls/RAIL/c-bert/hps/common.yaml
accelerate launch one_method.py --config yamls/RAIL/c-bert/hps/mrpc.yaml --common yamls/RAIL/c-bert/hps/common.yaml
accelerate launch one_method.py --config yamls/RAIL/c-bert/hps/rte.yaml --common yamls/RAIL/c-bert/hps/common.yaml
accelerate launch one_method.py --config yamls/RAIL/c-bert/hps/stsb.yaml --common yamls/RAIL/c-bert/hps/common.yaml