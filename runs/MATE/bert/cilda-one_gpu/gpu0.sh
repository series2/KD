#accelerate launch mate_training.py --config yamls/MATE/bert/cilda-one_gpu/hps/cola.yaml --common yamls/MATE/bert/cilda-one_gpu/hps/common_8_1e-05.yaml
# accelerate launch mate_training.py --config yamls/MATE/bert/cilda-one_gpu/hps/cola.yaml --common yamls/MATE/bert/cilda-one_gpu/hps/common_8_2e-05.yaml
# accelerate launch mate_training.py --config yamls/MATE/bert/cilda-one_gpu/hps/cola.yaml --common yamls/MATE/bert/cilda-one_gpu/hps/common_8_4e-06.yaml
# accelerate launch mate_training.py --config yamls/MATE/bert/cilda-one_gpu/hps/cola.yaml --common yamls/MATE/bert/cilda-one_gpu/hps/common_16_1e-05.yaml
# accelerate launch mate_training.py --config yamls/MATE/bert/cilda-one_gpu/hps/cola.yaml --common yamls/MATE/bert/cilda-one_gpu/hps/common_16_2e-05.yaml
# accelerate launch mate_training.py --config yamls/MATE/bert/cilda-one_gpu/hps/cola.yaml --common yamls/MATE/bert/cilda-one_gpu/hps/common_16_4e-06.yaml

accelerate launch mate_training.py --config yamls/MATE/bert/cilda-one_gpu/main/cola.yaml --common yamls/MATE/bert/cilda-one_gpu/main/common.yaml