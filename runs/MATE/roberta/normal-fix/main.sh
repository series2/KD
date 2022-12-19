accelerate launch mate_training.py --config yamls/MATE/roberta/normal-fix/main/cola.yaml --common yamls/MATE/roberta/normal-fix/main/common.yaml
accelerate launch mate_training.py --config yamls/MATE/roberta/normal-fix/main/mrpc.yaml --common yamls/MATE/roberta/normal-fix/main/common.yaml
accelerate launch mate_training.py --config yamls/MATE/roberta/normal-fix/main/rte.yaml --common yamls/MATE/roberta/normal-fix/main/common.yaml
accelerate launch mate_training.py --config yamls/MATE/roberta/normal-fix/main/stsb.yaml --common yamls/MATE/roberta/normal-fix/main/common.yaml