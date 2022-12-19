#accelerate launch mate_training.py --config yamls/MATE/roberta/normal-fix/hps/cola.yaml --common yamls/MATE/roberta/normal-fix/hps/common.yaml
accelerate launch mate_training.py --config yamls/MATE/roberta/normal-fix/hps/mrpc.yaml --common yamls/MATE/roberta/normal-fix/hps/common.yaml
accelerate launch mate_training.py --config yamls/MATE/roberta/normal-fix/hps/rte.yaml --common yamls/MATE/roberta/normal-fix/hps/common.yaml
accelerate launch mate_training.py --config yamls/MATE/roberta/normal-fix/hps/stsb.yaml --common yamls/MATE/roberta/normal-fix/hps/common.yaml