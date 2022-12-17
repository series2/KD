accelerate launch one_method.py --config yamls/nokd/distilbert/main/cola.yaml --common yamls/nokd/distilbert/main/common.yaml
accelerate launch one_method.py --config yamls/nokd/distilbert/main/mrpc.yaml --common yamls/nokd/distilbert/main/common.yaml
accelerate launch one_method.py --config yamls/nokd/distilbert/main/rte.yaml --common yamls/nokd/distilbert/main/common.yaml
accelerate launch one_method.py --config yamls/nokd/distilbert/main/stsb.yaml --common yamls/nokd/distilbert/main/common.yaml

accelerate launch one_method.py --config yamls/nokd/distilroberta/hps/cola.yaml --common yamls/nokd/distilroberta/hps/common.yaml
accelerate launch one_method.py --config yamls/nokd/distilroberta/hps/mrpc.yaml --common yamls/nokd/distilroberta/hps/common.yaml
accelerate launch one_method.py --config yamls/nokd/distilroberta/hps/rte.yaml --common yamls/nokd/distilroberta/hps/common.yaml
accelerate launch one_method.py --config yamls/nokd/distilroberta/hps/stsb.yaml --common yamls/nokd/distilroberta/hps/common.yaml