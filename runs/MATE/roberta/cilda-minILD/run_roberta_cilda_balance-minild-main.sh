accelerate launch mate_training.py --config yamls/MATE/cilda-minild-main/cola.yaml --common yamls/MATE/cilda-minild-main/common_balance.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda-minild-main/mrpc.yaml --common yamls/MATE/cilda-minild-main/common_balance.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda-minild-main/rte.yaml --common yamls/MATE/cilda-minild-main/common_balance.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda-minild-main/stsb.yaml --common yamls/MATE/cilda-minild-main/common_balance.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda-minild-main/qnli.yaml --common yamls/MATE/cilda-minild-main/common_balance.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda-minild-main/sst2.yaml --common yamls/MATE/cilda-minild-main/common_balance.yaml