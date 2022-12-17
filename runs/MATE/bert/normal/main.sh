accelerate launch mate_training.py --config yamls/MATE/bert/normal/main/cola.yaml --common yamls/MATE/bert/normal/main/common.yaml
accelerate launch mate_training.py --config yamls/MATE/bert/normal/main/mrpc.yaml --common yamls/MATE/bert/normal/main/common.yaml
accelerate launch mate_training.py --config yamls/MATE/bert/normal/main/rte.yaml --common yamls/MATE/bert/normal/main/common.yaml
accelerate launch mate_training.py --config yamls/MATE/bert/normal/main/stsb.yaml --common yamls/MATE/bert/normal/main/common.yaml