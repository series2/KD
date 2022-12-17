accelerate launch mate_training.py --config yamls/MATE/cilda/cola.yaml --common yamls/MATE/cilda/common_balance.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda/mrpc.yaml --common yamls/MATE/cilda/common_balance.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda/qnli.yaml --common yamls/MATE/cilda/common_balance.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda/rte.yaml --common yamls/MATE/cilda/common_balance.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda/sst2.yaml --common yamls/MATE/cilda/common_balance.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda/stsb.yaml --common yamls/MATE/cilda/common_balance.yaml