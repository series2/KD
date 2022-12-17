accelerate launch mate_training.py --config yamls/MATE/mate-test/cola.yaml --common yamls/MATE/mate-test/common.yaml
accelerate launch mate_training.py --config yamls/MATE/mate-test/mrpc.yaml --common yamls/MATE/mate-test/common.yaml
accelerate launch mate_training.py --config yamls/MATE/mate-test/qnli.yaml --common yamls/MATE/mate-test/common.yaml
accelerate launch mate_training.py --config yamls/MATE/mate-test/rte.yaml --common yamls/MATE/mate-test/common.yaml
accelerate launch mate_training.py --config yamls/MATE/mate-test/sst2.yaml --common yamls/MATE/mate-test/common.yaml
accelerate launch mate_training.py --config yamls/MATE/mate-test/stsb.yaml --common yamls/MATE/mate-test/common.yaml