accelerate launch one_method.py --config yamls/RAIL/c-bart/hps/cola.yaml --common yamls/RAIL/c-bart/hps/common.yaml
accelerate launch one_method.py --config yamls/RAIL/c-bart/hps/mrpc.yaml --common yamls/RAIL/c-bart/hps/common.yaml
accelerate launch one_method.py --config yamls/RAIL/c-bart/hps/rte.yaml --common yamls/RAIL/c-bart/hps/common.yaml
accelerate launch one_method.py --config yamls/RAIL/c-bart/hps/stsb.yaml --common yamls/RAIL/c-bart/hps/common.yaml