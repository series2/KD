accelerate launch one_method.py --config yamls/KD/robertakd-main/stsb.yaml
accelerate launch one_method.py --config yamls/RAIL/c-rand-main/stsb.yaml
accelerate launch one_method.py --config yamls/RAIL/c-training-main/stsb.yaml
accelerate launch one_method.py --config yamls/RAIL/c-pretrained-main/stsb.yaml
accelerate launch mate_training.py --config yamls/MATE/normal-main/stsb.yaml
