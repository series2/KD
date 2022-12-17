#accelerate launch mate_training.py --config yamls/MATE/cilda-minild/cola.yaml --common yamls/MATE/cilda-minild/common_balance.yaml
#accelerate launch mate_training.py --config yamls/MATE/cilda-minild/mrpc.yaml --common yamls/MATE/cilda-minild/common_balance.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda-minild/qnli.yaml --common yamls/MATE/cilda-minild/common_balance.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda-minild/rte.yaml --common yamls/MATE/cilda-minild/common_balance.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda-minild/sst2.yaml --common yamls/MATE/cilda-minild/common_balance.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda-minild/stsb.yaml --common yamls/MATE/cilda-minild/common_balance.yaml