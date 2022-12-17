accelerate launch one_method.py --config yamls/RKD/main/cola.yaml --common yamls/RKD/commons/main/common_KD_batch_128.yaml
accelerate launch one_method.py --config yamls/RKD/main/mrpc.yaml --common yamls/RKD/commons/main/common_KD_batch_128.yaml
accelerate launch one_method.py --config yamls/RKD/main/rte.yaml --common yamls/RKD/commons/main/common_KD_batch_128.yaml
accelerate launch one_method.py --config yamls/RKD/main/stsb.yaml --common yamls/RKD/commons/main/common_KD_batch_128.yaml
accelerate launch one_method.py --config yamls/RKD/main/qnli.yaml --common yamls/RKD/commons/main/common_KD_batch_128.yaml
accelerate launch one_method.py --config yamls/RKD/main/sst2.yaml --common yamls/RKD/commons/main/common_KD_batch_128.yaml
