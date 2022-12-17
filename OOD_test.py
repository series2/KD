import mytools
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate import Accelerator
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

def OOD_test(args):
    conf = mytools.get_conf(args)
    raw_dataset = load_dataset(conf['task'], split='test')
    if conf['regression']:
        num_labels = 1
    else:
        label_list = raw_dataset.features["label"].names
        num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(conf['tokenizer'])
    config = AutoConfig.from_pretrained(conf['student'], num_labels=num_labels)

    def preprocess_function(examples):
        # Tokenize the texts
        #logger.info('tokenizing')
        padding = conf['pad_to_max_length']
        texts = (
            (examples[conf['skey1']],) if conf['skey2'] is None else (examples[conf['skey1']], examples[conf['skey2']])
        )
        result = tokenizer(*texts, padding=padding, max_length=conf['max_length'], truncation=True)

        result['labels'] = examples['label']
        return result
    
    results = []
    losses = []
    for i in range(conf['num_of_ex']):
        accelerator = Accelerator(log_with='all', logging_dir=conf['outdir'])
        processed_dataset = raw_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_dataset.column_names,
        )

        if conf['pad_to_max_length']:
            # If padding was already done ot max length, we use the default data collator that will just convert everything
            # to tensors.
            data_collator = default_data_collator
        else:
            # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
            # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
            # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

        dataloader = DataLoader(
            processed_dataset, shuffle=True, collate_fn=data_collator, batch_size=conf['batch_size']
        )
        model = AutoModelForSequenceClassification.from_pretrained(conf['model'], config=config)
        check = torch.load(conf['checkpoints'][i])
        metric = evaluate.load(conf['metric'])

        if 'model_key' in conf:
            model.load_state_dict(check[conf['model_key']])

        samples_seen = 0
        loss_sum = torch.tensor(0, dtype=torch.float, requires_grad=False).to(accelerator.device)

        model, dataloader = accelerator.prepare(
            model, dataloader
        )

        for step, batch in enumerate(dataloader):
            outputs = model(**batch)
            loss_sum += outputs.loss.detach().clone()
            predictions = outputs.logits.argmax(dim=-1) if not conf['regression'] else outputs.logits.squeeze()
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            if accelerator.num_processes > 1:
                if step == len(dataloader) - 1:
                    predictions = predictions[: len(dataloader.dataset) - samples_seen]
                    references = references[: len(dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
        if accelerator.is_main_process:
            print('integrating valid loss')
        accelerator.wait_for_everyone()

        results.append(metric.compute()[conf['metric']])
        
        loss = accelerator.gather(loss_sum)
        loss = torch.sum(loss)/len(dataloader)
        losses.append(loss.detach().clone())

        del model, loss, losses, results, metric
        torch.cuda.empty_cache()
    
    with open(conf['outdir']+'OOD_result.pkl', 'wb') as f:
        pickle.dump({'metric':conf['metcis'], 'results':results, 'losses':losses}, f)




if __name__ == "__main__":
    args = mytools.parse_args()
    OOD_test(args)