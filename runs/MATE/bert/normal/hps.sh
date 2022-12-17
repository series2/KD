# accelerate launch mate_training.py --config yamls/MATE/bert/normal/hps/cola.yaml --common yamls/MATE/bert/normal/hps/common.yaml
# accelerate launch mate_training.py --config yamls/MATE/bert/normal/hps/mrpc.yaml --common yamls/MATE/bert/normal/hps/common.yaml
# accelerate launch mate_training.py --config yamls/MATE/bert/normal/hps/rte.yaml --common yamls/MATE/bert/normal/hps/common.yaml
accelerate launch mate_training.py --config yamls/MATE/bert/normal/hps/stsb.yaml --common yamls/MATE/bert/normal/hps/common.yaml