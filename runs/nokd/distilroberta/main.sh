accelerate launch one_method.py --config yamls/nokd/distilroberta/main/cola.yaml --common yamls/nokd/distilroberta/main/common.yaml
accelerate launch one_method.py --config yamls/nokd/distilroberta/main/mrpc.yaml --common yamls/nokd/distilroberta/main/common.yaml
accelerate launch one_method.py --config yamls/nokd/distilroberta/main/rte.yaml --common yamls/nokd/distilroberta/main/common.yaml
accelerate launch one_method.py --config yamls/nokd/distilroberta/main/stsb.yaml --common yamls/nokd/distilroberta/main/common.yaml