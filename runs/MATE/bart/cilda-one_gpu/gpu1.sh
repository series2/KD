# accelerate launch mate_training.py --config yamls/MATE/bart/cilda-one_gpu/hps/cola.yaml --common yamls/MATE/bart/cilda-one_gpu/hps/common_8_2e-05.yaml
# accelerate launch mate_training.py --config yamls/MATE/bart/cilda-one_gpu/hps/cola.yaml --common yamls/MATE/bart/cilda-one_gpu/hps/common_16_4e-06.yaml
accelerate launch mate_training.py --config yamls/MATE/bart/cilda-one_gpu/hps/stsb.yaml --common yamls/MATE/bart/cilda-one_gpu/hps/common_16_1e-05.yaml