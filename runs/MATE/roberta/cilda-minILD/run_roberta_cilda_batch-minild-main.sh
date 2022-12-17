#accelerate launch mate_training.py --config yamls/MATE/cilda_batch-minild-main/cola.yaml --common yamls/MATE/cilda_batch-minild-main/common_batch.yaml
#accelerate launch mate_training.py --config yamls/MATE/cilda_batch-minild-main/mrpc.yaml --common yamls/MATE/cilda_batch-minild-main/common_batch.yaml
#accelerate launch mate_training.py --config yamls/MATE/cilda_batch-minild-main/rte.yaml --common yamls/MATE/cilda_batch-minild-main/common_batch.yaml
#accelerate launch mate_training.py --config yamls/MATE/cilda_batch-minild-main/stsb.yaml --common yamls/MATE/cilda_batch-minild-main/common_batch.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda_batch-minild-main/qnli.yaml --common yamls/MATE/cilda_batch-minild-main/common_batch.yaml
accelerate launch mate_training.py --config yamls/MATE/cilda_batch-minild-main/sst2.yaml --common yamls/MATE/cilda_batch-minild-main/common_batch.yaml