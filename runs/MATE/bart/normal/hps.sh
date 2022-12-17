accelerate launch mate_training.py --config yamls/MATE/bart/normal/hps/cola.yaml --common yamls/MATE/bart/normal/hps/common.yaml
accelerate launch mate_training.py --config yamls/MATE/bart/normal/hps/mrpc.yaml --common yamls/MATE/bart/normal/hps/common.yaml
accelerate launch mate_training.py --config yamls/MATE/bart/normal/hps/rte.yaml --common yamls/MATE/bart/normal/hps/common.yaml
accelerate launch mate_training.py --config yamls/MATE/bart/normal/hps/stsb.yaml --common yamls/MATE/bart/normal/hps/common.yaml