accelerate launch one_method.py --config yamls/KD/robertakd/stsb.yaml
accelerate launch one_method.py --config yamls/RAIL/c-rand/stsb.yaml
accelerate launch one_method.py --config yamls/RAIL/c-training/stsb.yaml
accelerate launch one_method.py --config yamls/RAIL/c-pretrained/stsb.yaml
accelerate launch mate_training.py --config yamls/MATE/normal/stsb.yaml
