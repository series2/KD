# accelerate launch mate_training.py --config yamls/MATEILD/hps/cola.yaml --common yamls/MATEILD/hps/commons/bothILD.yaml
# accelerate launch mate_training.py --config yamls/MATEILD/hps/mrpc.yaml --common yamls/MATEILD/hps/commons/bothILD.yaml
# accelerate launch mate_training.py --config yamls/MATEILD/hps/rte.yaml --common yamls/MATEILD/hps/commons/bothILD.yaml
# accelerate launch mate_training.py --config yamls/MATEILD/hps/stsb.yaml --common yamls/MATEILD/hps/commons/bothILD.yaml
# accelerate launch mate_training.py --config yamls/MATEILD/hps/qnli.yaml --common yamls/MATEILD/hps/commons/bothILD.yaml
accelerate launch mate_training.py --config yamls/MATEILD/hps/sst2.yaml --common yamls/MATEILD/hps/commons/bothILD.yaml
