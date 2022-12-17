accelerate launch mate_training.py --config yamls/MATE/cilda-MATE/cola.yaml --common yamls/MATE/cilda-MATE/common.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda-MATE/mrpc.yaml --common yamls/MATE/cilda-MATE/common.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda-MATE/rte.yaml --common yamls/MATE/cilda-MATE/common.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda-MATE/stsb.yaml --common yamls/MATE/cilda-MATE/common.yaml
#accelerate launch mate_training.py --config yamls/MATE/cilda-MATE/qnli.yaml --common yamls/MATE/cilda-MATE/common.yaml
#accelerate launch mate_training.py --config yamls/MATE/cilda-MATE/sst2.yaml --common yamls/MATE/cilda-MATE/common.yaml