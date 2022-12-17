accelerate launch one_method.py --config yamls/nokd/bart/hps/cola.yaml --common yamls/nokd/bart/hps/common.yaml
accelerate launch one_method.py --config yamls/nokd/bart/hps/mrpc.yaml --common yamls/nokd/bart/hps/common.yaml
accelerate launch one_method.py --config yamls/nokd/bart/hps/rte.yaml --common yamls/nokd/bart/hps/common.yaml
accelerate launch one_method.py --config yamls/nokd/bart/hps/stsb.yaml --common yamls/nokd/bart/hps/common.yaml