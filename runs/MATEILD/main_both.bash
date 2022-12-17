accelerate launch mate_training.py --config yamls/MATEILD/main/cola.yaml --common yamls/MATEILD/main/commons/bothILD.yaml
accelerate launch mate_training.py --config yamls/MATEILD/main/mrpc.yaml --common yamls/MATEILD/main/commons/bothILD.yaml
accelerate launch mate_training.py --config yamls/MATEILD/main/qnli.yaml --common yamls/MATEILD/main/commons/bothILD.yaml
accelerate launch mate_training.py --config yamls/MATEILD/main/sst2.yaml --common yamls/MATEILD/main/commons/bothILD.yaml
accelerate launch mate_training.py --config yamls/MATEILD/main/rte.yaml --common yamls/MATEILD/main/commons/bothILD.yaml
accelerate launch mate_training.py --config yamls/MATEILD/main/stsb.yaml --common yamls/MATEILD/main/commons/bothILD.yaml
