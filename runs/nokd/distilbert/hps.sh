accelerate launch one_method.py --config yamls/nokd/distilbert/hps/cola.yaml --common yamls/nokd/distilbert/hps/common.yaml
accelerate launch one_method.py --config yamls/nokd/distilbert/hps/mrpc.yaml --common yamls/nokd/distilbert/hps/common.yaml
accelerate launch one_method.py --config yamls/nokd/distilbert/hps/rte.yaml --common yamls/nokd/distilbert/hps/common.yaml
accelerate launch one_method.py --config yamls/nokd/distilbert/hps/stsb.yaml --common yamls/nokd/distilbert/hps/common.yaml