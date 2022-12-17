accelerate launch one_method.py --config yamls/RAIL/c-bert/main/cola.yaml --common yamls/RAIL/c-bert/main/common.yaml
accelerate launch one_method.py --config yamls/RAIL/c-bert/main/mrpc.yaml --common yamls/RAIL/c-bert/main/common.yaml
accelerate launch one_method.py --config yamls/RAIL/c-bert/main/rte.yaml --common yamls/RAIL/c-bert/main/common.yaml
accelerate launch one_method.py --config yamls/RAIL/c-bert/main/stsb.yaml --common yamls/RAIL/c-bert/main/common.yaml