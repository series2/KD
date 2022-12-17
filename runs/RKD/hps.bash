accelerate launch one_method.py --config yamls/RKD/hps/cola.yaml --common yamls/RKD/commons/hps/common_111_batch_128.yaml
accelerate launch one_method.py --config yamls/RKD/hps/mrpc.yaml --common yamls/RKD/commons/hps/common_111_batch_128.yaml
accelerate launch one_method.py --config yamls/RKD/hps/rte.yaml --common yamls/RKD/commons/hps/common_111_batch_128.yaml
accelerate launch one_method.py --config yamls/RKD/hps/stsb.yaml --common yamls/RKD/commons/hps/common_111_batch_128.yaml
accelerate launch one_method.py --config yamls/RKD/hps/qnli.yaml --common yamls/RKD/commons/hps/common_111_batch_128.yaml
accelerate launch one_method.py --config yamls/RKD/hps/sst2.yaml --common yamls/RKD/commons/hps/common_111_batch_128.yaml
