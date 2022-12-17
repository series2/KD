accelerate launch mate_training.py --config yamls/MATE/cilda_balance-main/cola.yaml --common yamls/MATE/cilda_balance-main/common_balance.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda_balance-main/mrpc.yaml --common yamls/MATE/cilda_balance-main/common_balance.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda_balance-main/qnli.yaml --common yamls/MATE/cilda_balance-main/common_balance.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda_balance-main/rte.yaml --common yamls/MATE/cilda_balance-main/common_balance.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda_balance-main/sst2.yaml --common yamls/MATE/cilda_balance-main/common_balance.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda_balance-main/stsb.yaml --common yamls/MATE/cilda_balance-main/common_balance.yaml