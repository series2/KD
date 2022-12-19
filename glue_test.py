import mytools
import torch
from datasets import load_dataset
import evaluate

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import pickle
import pandas as pd

def glue_test(args):
    conf = mytools.get_conf(args)
    
    if conf['task']=='stsb':
        is_regression = True

    # conf['labels'] is a list like ['entailment', 'not entailment']
    num_labels=len(conf['labels'])

    tokenizer = AutoTokenizer.from_pretrained(conf['tokenizer'])
    config = AutoConfig.from_pretrained(conf['student'], num_labels=num_labels, finetuning_task=conf['task'])

    pd_dataset = pd.read_table(conf['datapath'], header=0, index_col=0)
    skey1 = pd_dataset.columns[0]
    if len(pd_dataset.columns) == 2:
        skey2 = pd_dataset.columns[1]
    else:
        skey2 = None

    tokenized = []
    for t in pd_dataset.itertuples():
        text = (t[1],) if skey2 is None else (t[1], t[2])
        tokenized.append(tokenizer(*text, return_tensors='pt'))

    for i in range(conf['num_of_ex']):
        pred_all = []

        model = AutoModelForSequenceClassification.from_pretrained(conf['model'], config=config)
        # conf['checkpoint'] is like OUTPUTS/MATE/cilda-main/checkpoints/
        # conf['task_check'] is like cola/2e-05/
        check = torch.load(conf['checkpoints']+conf['task_check']+str(i)+'/check.bin')

        model.to('cuda')

        # check is a dict like {student:state_dict, teacher:state_dict, ...}
        if 'model_key' in conf:
            model.load_state_dict(check[conf['model_key']])

        samples_seen = 0

        for inputs in tokenized:
            inputs.to('cuda')
            outputs = model(**inputs)
            prediction = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            ### it is unclear that prediction is list.
            pred_all += prediction

        with open(conf['outdir']+str(i)+'/'+conf['task']+'.tsv', 'w') as f:
            if not is_regression:
                for j in range(len(pred_all)):
                    f.write(str(j)+'\t'+conf['label'][pred_all[j]])
        

        del model
        torch.cuda.empty_cache()




if __name__ == "__main__":
    args = mytools.parse_args()
    glue_test(args)