#accelerate launch one_method.py --config yamls/CatILD/main/cola.yaml --common yamls/CatILD/main/commons/128.yaml
#accelerate launch one_method.py --config yamls/CatILD/main/mrpc.yaml --common yamls/CatILD/main/commons/128.yaml
#accelerate launch one_method.py --config yamls/CatILD/main/rte.yaml --common yamls/CatILD/main/commons/128.yaml
#accelerate launch one_method.py --config yamls/CatILD/main/stsb.yaml --common yamls/CatILD/main/commons/128.yaml
accelerate launch one_method.py --config yamls/CatILD/main/qnli.yaml --common yamls/CatILD/main/commons/128.yaml
accelerate launch one_method.py --config yamls/CatILD/main/sst2.yaml --common yamls/CatILD/main/commons/128.yaml
