accelerate launch one_method.py --config yamls/RAIL/c-bart/main/cola.yaml --common yamls/RAIL/c-bart/main/common.yaml
accelerate launch one_method.py --config yamls/RAIL/c-bart/main/mrpc.yaml --common yamls/RAIL/c-bart/main/common.yaml
accelerate launch one_method.py --config yamls/RAIL/c-bart/main/rte.yaml --common yamls/RAIL/c-bart/main/common.yaml
accelerate launch one_method.py --config yamls/RAIL/c-bart/main/stsb.yaml --common yamls/RAIL/c-bart/main/common.yaml