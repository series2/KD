accelerate launch one_method.py --config yamls/nokd/bart/main/cola.yaml --common yamls/nokd/bart/main/common.yaml
accelerate launch one_method.py --config yamls/nokd/bart/main/mrpc.yaml --common yamls/nokd/bart/main/common.yaml
accelerate launch one_method.py --config yamls/nokd/bart/main/rte.yaml --common yamls/nokd/bart/main/common.yaml
accelerate launch one_method.py --config yamls/nokd/bart/main/stsb.yaml --common yamls/nokd/bart/main/common.yaml