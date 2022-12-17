accelerate launch mate_training.py --config yamls/MATE/bart/normal/main/cola.yaml --common yamls/MATE/bart/normal/main/common.yaml
accelerate launch mate_training.py --config yamls/MATE/bart/normal/main/mrpc.yaml --common yamls/MATE/bart/normal/main/common.yaml
accelerate launch mate_training.py --config yamls/MATE/bart/normal/main/rte.yaml --common yamls/MATE/bart/normal/main/common.yaml
accelerate launch mate_training.py --config yamls/MATE/bart/normal/main/stsb.yaml --common yamls/MATE/bart/normal/main/common.yaml