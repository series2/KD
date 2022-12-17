accelerate launch mate_training.py --config yamls/MATE/cilda-main/cola.yaml --common yamls/MATE/cilda-main/common1024.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda-main/mrpc.yaml --common yamls/MATE/cilda-main/common1024.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda-main/rte.yaml --common yamls/MATE/cilda-main/common1024.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda-main/stsb.yaml --common yamls/MATE/cilda-main/common1024.yaml
#accelerate launch mate_training.py --config yamls/MATE/cilda-main/qnli.yaml --common yamls/MATE/cilda-main/common1024.yaml
#accelerate launch mate_training.py --config yamls/MATE/cilda-main/sst2.yaml --common yamls/MATE/cilda-main/common1024.yaml