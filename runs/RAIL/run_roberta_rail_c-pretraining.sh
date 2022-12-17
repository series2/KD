accelerate launch ILD_pretraining.py --config yamls/RAIL/c-pretraining/cola.yaml
accelerate launch ILD_pretraining.py --config yamls/RAIL/c-pretraining/mrpc.yaml
accelerate launch ILD_pretraining.py --config yamls/RAIL/c-pretraining/qnli.yaml
accelerate launch ILD_pretraining.py --config yamls/RAIL/c-pretraining/rte.yaml
accelerate launch ILD_pretraining.py --config yamls/RAIL/c-pretraining/sst2.yaml
accelerate launch ILD_pretraining.py --config yamls/RAIL/c-pretraining/stsb.yaml