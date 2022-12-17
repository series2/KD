import os
import json
import argparse
import math
import time
import logging
import random
import copy

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.distributed.elastic.multiprocessing.errors import record

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

import evaluate

import datasets
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import DistributedDataParallelKwargs

from pytorch_memlab import profile

import GPUtil

import yaml
import KD_loss, KD_model, KD_admin
from adv_tools import mask_tokens
import mytools

import neptune.new as neptune

ILDs = ['RAIL_l', 'RAIL_c', 'CILDA']
ADVs = ['MATE', 'CILDA']
SetA = ['RAIL_c', 'CatILD', 'Bart_RAIL_c']
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task_to_metrics = {
    "cola": 'matthews_correlation',
    "mnli": 'accuracy',
    "mrpc": 'f1',
    "qnli": 'accuracy',
    "qqp": 'accuracy',
    "rte": 'accuracy',
    "sst2": 'accuracy',
    "stsb": 'pearson',
    "wnli": 'accuracy',
}

logger = get_logger(__name__)

def train(model, admin, task, conf, accelerator, num_labels, train_dataloader, valid_dataloader, metric, is_regression, lr, i):    
    KD = conf['method'] != 'normal'
    RAIL = 'RAIL' in conf['method']
    ADV = conf['method'] in ADVs
    ILD = conf['method'] in ILDs
    gradient_accumulation_steps = conf['num_accumulation_steps'] if 'num_accumulation_steps' in conf else 1

    if accelerator.is_main_process:
        run = neptune.init_run(
            project=conf['nep_proj'],
            api_token=conf['nep_token'],
        )
        neptune_params = {'method':conf['nep_method'], 'task' : task, 'batch_size' : conf['batch_size']*conf['device_num']*gradient_accumulation_steps, 'lr' : lr, 'ex_num': i}
        run['parameters'] = neptune_params
        print('neputune is initialized')

    accelerator.wait_for_everyone()
    learning_log = {}
    no_decay = ["bias", "LayerNorm.weight"]

    # set optimizer parameter
    if not KD:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": conf['wd'],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
    else:
        optimizer_grouped_parameters = admin.opt_parameter(model)
    # set optimizer parameter end
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    lr_scheduler = get_scheduler(
        name=conf['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=conf['num_warmup_steps'],
        num_training_steps=len(train_dataloader)*conf['epochs'],
    )
    model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )
    if conf['method'] in SetA:
        model.module.set_accelerator(accelerator)
    total_batch_size = conf['batch_size'] * accelerator.num_processes
    
    logger.info(f'training, lr : {lr}, ex_num : {i}')
    time_start = int(time.time())

    best_metric = 0
    for epoch in range(conf['epochs']):
        if RAIL:
            admin.select_layers(model)
        # one_epoch_log : dict keys is [train_loss, valid_loss, some metric]
        one_epoch_log = {}
        loss_sum = torch.tensor(0, dtype=torch.float, requires_grad=False).to(accelerator.device)
        if KD:
            admin.train_mode(model)
        else:
            model.train()
        if accelerator.is_main_process:
            print(f'epoch {epoch} : training')
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if KD:
                    #start = time.time()
                    outputs, loss, losses = model(**batch)
                    if accelerator.is_main_process:
                        admin.log_losses(run, losses, 'train')
                    #end = time.time()
                    #print('for ', end-start)
                    #start = time.time()
                    accelerator.backward(loss)
                    #end = time.time()
                    #print('back ', end-start)
                    loss_sum += loss.detach().clone()
                    
                else:
                    #start = time.time()
                    outputs = model(**batch)
                    #end = time.time()
                    #print('for ', end-start)
                    #start = time.time()
                    accelerator.backward(outputs.loss)
                    #end = time.time()
                    #print('back ', end-start)
                    loss_sum += outputs.loss.detach().clone()
                optimizer.step()
                lr_scheduler.step()

        # integrate loss
        if accelerator.is_main_process:
            print('integrating train loss')
        
        train_loss = accelerator.gather(loss_sum)
        train_loss = torch.sum(train_loss)/len(train_dataloader)
        one_epoch_log['train_loss'] = train_loss.item()
        # integrate loss end
        
        if KD:
            admin.eval_mode(model)
        else:
            model.eval()
        samples_seen = 0
        loss_sum = torch.tensor(0, dtype=torch.float, requires_grad=False).to(accelerator.device)
        if accelerator.is_main_process:
            print(f'epoch {epoch} : validation')

        for step, batch in enumerate(valid_dataloader):
            with torch.no_grad():
                if KD:
                    outputs, loss, losses = model(**batch)
                    if accelerator.is_main_process:
                        admin.log_losses(run, losses, 'valid')
                    loss_sum += loss.detach().clone()
                else:
                    outputs = model(**batch)
                    loss_sum += outputs.loss.detach().clone()
            
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            if accelerator.num_processes > 1:
                if step == len(valid_dataloader) - 1:
                    predictions = predictions[: len(valid_dataloader.dataset) - samples_seen]
                    references = references[: len(valid_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
        if accelerator.is_main_process:
            print('integrating valid loss')
        
        valid_loss = accelerator.gather(loss_sum)
        valid_loss = torch.sum(valid_loss)/len(valid_dataloader)
        one_epoch_log['valid_loss'] = valid_loss.item()
        
        del outputs, predictions, references
        torch.cuda.empty_cache()
        
        accelerator.wait_for_everyone()
        eval_metric = metric.compute()
        if accelerator.is_main_process:
            run['train_loss'].log(train_loss.item())
            run['valid_loss'].log(valid_loss.item())
            run[task_to_metrics[task]].log(eval_metric[task_to_metrics[task]])
        # checkpoint

        save_check = conf['save_check'] if 'save_check' in conf else True
        if accelerator.is_main_process and save_check:
            if eval_metric[task_to_metrics[task]] > best_metric:
                print('best check')
            # if 1:
                if 'outdir' in conf.keys():
                    check_dir = conf['outdir']+'/checkpoints/'+task+'/'+str(lr)+'/'+str(i)
                    os.makedirs(check_dir, exist_ok=True)
                    if KD:
                        model_checkpoint = admin.checkpoint(model.module)
                    else:
                        model_checkpoint = {'model':model.module.state_dict()}
                    checkpoint = {
                        "epoch" : epoch,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                        "random": random.getstate(),
                        "np_random": np.random.get_state(), # numpy.randomを使用する場合は必要
                        "torch": torch.get_rng_state(),
                        "torch_random": torch.random.get_rng_state(),
                        "cuda_random": torch.cuda.get_rng_state(), # gpuを使用する場合は必要
                        "cuda_random_all": torch.cuda.get_rng_state_all(), # 複数gpuを使用する場合は必要
                    }
                    checkpoint = checkpoint | model_checkpoint
                    print('saving checkpoint')
                    torch.save(checkpoint, check_dir+'/check.bin')
                    print('saving ends')
                    best_metric = eval_metric[task_to_metrics[task]]
        
        accelerator.wait_for_everyone()
        one_epoch_log = one_epoch_log | eval_metric
        learning_log[epoch] = one_epoch_log
        logger.info(f"epoch {epoch}: {learning_log[epoch]}, time: {int(time.time()-time_start)} seconds.")
    
    if accelerator.is_main_process:
        run.stop()
    accelerator.wait_for_everyone()

    df_log = pd.DataFrame.from_dict(learning_log, orient='index')
    # return last epoch's model and eval_metric
    return model, eval_metric, df_log
    #returned_model = model.module
    #return returned_model, eval_metric, df_log

def do_one_task(conf, task):
    KD = conf['method'] != 'normal'
    RAIL = 'RAIL' in conf['method']
    ADV = conf['method'] in ADVs
    ILD = conf['method'] in ILDs

    gradient_accumulation_steps = conf['num_accumulation_steps'] if 'num_accumulation_steps' in conf else 1

    #print('do_one_task starts.', GPUtil.showUtilization())
    outer_accelerator = Accelerator(log_with='all', logging_dir=conf['outdir'])
    if 'outdir' in conf.keys():
        df_logs_dir = conf['outdir']+'/df_logs/'+task
        models_dir = conf['outdir']+'/models/'+task
        results_dir = conf['outdir']+'/results/'
        log_dir = conf['outdir']+'/log/'+task
        if outer_accelerator.is_main_process:
            os.makedirs(models_dir, exist_ok=True)
            os.makedirs(df_logs_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)

    outer_accelerator.wait_for_everyone()

    logging.basicConfig(
        filename=log_dir+'/log.log', 
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(outer_accelerator.state, main_process_only=False)
    if outer_accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # eval_metrics
    # column : [exNO, lr, [metrics]]
    eval_metrics = pd.DataFrame(index=[], columns=[])

    if 'seed' in conf.keys():
        set_seed(conf['seed'])

    logger.info('loading dataset')
    raw_datasets = load_dataset("glue", task)
    is_regression = task == "stsb"
    
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
    
    logger.info('loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(conf['tokenizer'])

    sentence1_key, sentence2_key = task_to_keys[task]
# --------------------------------------------------------- #
    def preprocess_function(examples):
        # Tokenize the texts
        #logger.info('tokenizing')
        padding = conf['pad_to_max_length']
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=conf['max_length'], truncation=True)

        result['labels'] = examples['label']
        return result
    # It is not a good source code
    def preprocess_function_adv(examples):
        # Tokenize the texts
        padding = conf['pad_to_max_length']
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=conf['max_length'], truncation=True, return_tensors="np")
        input_ids_permuted, labels_permuted, mask_permuted = mask_tokens(torch.from_numpy(result['input_ids']).clone(), tokenizer, conf['mlm_prob'])
        result['input_ids_permuted'] = input_ids_permuted
        result['mask_permuted'] = mask_permuted
        result['labels'] = examples['label']
        return result
# --------------------------------------------------------- #
    best_model = None
    best_metric = 0
    best_lr = 0
    best_i = 0
    start_num = conf['start_num'] if 'start_num' in conf else 0

    for lr in conf['lr']:
        logger.info(f'learning rate is {lr}.')
        for i in range(start_num, conf['num_of_ex']):
            # repeat same experiment (initial state is changed)
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            accelerator = Accelerator(log_with='all', logging_dir=conf['outdir'], kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=gradient_accumulation_steps)
            metric = evaluate.load("glue", task)
            if accelerator.is_main_process:
                # df_log : timelapse of loss
                # each lr and ith experiment
                df_logs = pd.DataFrame(index=[], columns=[])

            logger.info(f'experiment {i}.')
            with accelerator.main_process_first():
                if ADV:
                    processed_datasets = raw_datasets.map(
                        preprocess_function_adv,
                        batched=True,
                        remove_columns=raw_datasets["train"].column_names,
                    )
                else:
                    processed_datasets = raw_datasets.map(
                        preprocess_function,
                        batched=True,
                        remove_columns=raw_datasets["train"].column_names,
                    )
            train_dataset = processed_datasets["train"]
            valid_dataset = processed_datasets["validation_matched" if task == "mnli" else "validation"]

            train_size = int(train_dataset.num_rows*conf['data_ratio'])
            valid_size = int(valid_dataset.num_rows*conf['data_ratio'])

            train_dataset = train_dataset.select(list(range(train_size)))
            valid_dataset = valid_dataset.select(list(range(valid_size)))

            if conf['pad_to_max_length']:
                # If padding was already done ot max length, we use the default data collator that will just convert everything
                # to tensors.
                data_collator = default_data_collator
            else:
                # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
                # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
                # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
                data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

            train_dataloader = DataLoader(
                train_dataset, shuffle=True, collate_fn=data_collator, batch_size=conf['batch_size']
            )
            valid_dataloader = DataLoader(valid_dataset, collate_fn=data_collator, batch_size=conf['batch_size'])
            logger.info('loading model')
            if conf['method'] == 'normal':
                config = AutoConfig.from_pretrained(conf['model'], num_labels=num_labels, finetuning_task=task)
                model = AutoModelForSequenceClassification.from_pretrained(
                    conf['model'],
                    config=config
                )
                admin = None
            elif conf['method'] == 'KD':
                model = KD_model.KD_model(conf, task, num_labels)
                admin = KD_admin.KD_admin(conf)
            elif conf['method'] == 'RAIL_l':
                model = KD_model.RAIL_l_model(conf, task, num_labels)
                admin = KD_admin.RAIL_l_admin(conf)
            elif conf['method'] == 'RAIL_c':
                model = KD_model.RAIL_c_model(conf, task, num_labels)
                admin = KD_admin.RAIL_c_admin(conf)
            elif conf['method'] == 'Bart_RAIL_c':
                model = KD_model.Bart_RAIL_c_model(conf, task, num_labels)
                admin = KD_admin.Bart_RAIL_c_admin(conf)
            elif conf['method'] == 'RKD':
                model = KD_model.RKD_model(conf, task, num_labels)
                admin = KD_admin.RKD_admin(conf)
            elif conf['method'] == 'CatILD':
                model = KD_model.CatILD_model(conf, task, num_labels)
                admin = KD_admin.CatILD_admin(conf)
            # loading model end
            returned_model, eval_metric, df_log = train(model, admin, task, conf, accelerator, num_labels, train_dataloader, valid_dataloader, metric, is_regression, lr, i)
            del model
            torch.cuda.empty_cache()
            if accelerator.is_main_process:
                df_logs = pd.concat([df_logs, df_log])
                if 'outdir' in conf.keys():
                    df_logs.to_csv(df_logs_dir+'/'+str(lr)+'_'+str(i)+'.csv')

            
            if task == "mnli":
                # Final evaluation on mismatched validation set
                valid_dataset = processed_datasets["validation_mismatched"]
                valid_size = int(valid_dataset.num_rows*conf['data_ratio'])
                valid_dataset = valid_dataset.select(list(range(valid_size)))
                valid_dataloader = DataLoader(
                    valid_dataset, collate_fn=data_collator, batch_size=conf['batch_size']
                )
                valid_dataloader = accelerator.prepare(valid_dataloader)

                returned_model.eval()
                for step, batch in enumerate(valid_dataloader):
                    if KD:
                        outputs, loss = returned_model(**batch)
                    else:
                        outputs = returned_model(**batch)
                    predictions = outputs.logits.argmax(dim=-1)
                    metric.add_batch(
                        predictions=accelerator.gather(predictions),
                        references=accelerator.gather(batch["labels"]),
                    )
                eval_metric['mis_acc']= metric.compute()['accuracy']
            eval_metric['exNO'] = i
            eval_metric['lr'] = lr
            eval_metric = pd.DataFrame(eval_metric, index=[len(eval_metrics)])
            eval_metrics = pd.concat([eval_metrics, eval_metric])

            if eval_metric.loc[len(eval_metrics)-1, task_to_metrics[task]] > best_metric:
                logger.info('This is the best!')
                best_metric = eval_metric.loc[len(eval_metrics)-1, task_to_metrics[task]]
                best_lr = lr
                best_i = i
                if 'outdir' in conf.keys():
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(returned_model)
                    del best_model
                    torch.cuda.empty_cache()
                    best_model = unwrapped_model
                    # unwrapped_model.save_pretrained(
                    #     outdir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    # )
                    # if accelerator.is_main_process:
                    #     tokenizer.save_pretrained(outdir)
                    del unwrapped_model

            # if accelerator.is_main_process:
            #     print('before delete model.', GPUtil.showUtilization())
            del returned_model
            torch.cuda.empty_cache()

            # if accelerator.is_main_process:
            #     print('after delete model.', GPUtil.showUtilization())

            del accelerator, metric
            if i != conf['num_of_ex']-1:
                del admin
            if 'outdir' in conf.keys() and outer_accelerator.is_main_process:
                check_dir = conf['outdir']+'/random_states/'+task+'/'+str(lr)+'/'+str(i)
                os.makedirs(check_dir, exist_ok=True)
                random_state = {
                    "random": random.getstate(),
                    "np_random": np.random.get_state(), # numpy.randomを使用する場合は必要
                    "torch": torch.get_rng_state(),
                    "torch_random": torch.random.get_rng_state(),
                    "cuda_random": torch.cuda.get_rng_state(), # gpuを使用する場合は必要
                    "cuda_random_all": torch.cuda.get_rng_state_all(), # 複数gpuを使用する場合は必要
                }
                print('saving random state')
                torch.save(random_state, check_dir+'/random_state.bin')
                print('saving ends')
            torch.cuda.empty_cache()
            outer_accelerator.wait_for_everyone()
            
    if 'outdir' in conf.keys():
        save_dir = models_dir+'/'+str(best_lr)+'_'+str(best_i)
        os.makedirs(save_dir, exist_ok=True)
        if KD:
            admin.save(best_model, save_dir, outer_accelerator)
        else:
            best_model.save_pretrained(
                save_dir, is_main_process=outer_accelerator.is_main_process, save_function=outer_accelerator.save
            )
            
        del best_model
        torch.cuda.empty_cache()
        if outer_accelerator.is_main_process:
            tokenizer.save_pretrained(save_dir)
            eval_metrics.to_csv(results_dir+task+'.csv')
    
    outer_accelerator.wait_for_everyone()
    del outer_accelerator
    torch.cuda.empty_cache()
    #print('do_one_task ends.', GPUtil.showUtilization())

def one_method(config_filename):
    conf = mytools.get_conf(args)

    tasks = conf['tasks']
    if 'random_state_dir' in conf:
        random_state = torch.load(conf['rand_state_dir'])
        random.setstate(random_state['random'])
        np.random.set_state()(random_state['np_random'])
        torch.set_rng_state(random_state['torch'])
        torch.cuda.set_rng_state(random_state['cuda_random'])
        torch.cuda.set_rng_state_all(random_state['cuda_random_all'])
    
    for task in tasks:
        time_start=time.time()
        do_one_task(conf, task)
        time_end=time.time()
        if 'outdir' in conf.keys():
            time_dir = conf['outdir']+'/time/'
            os.makedirs(time_dir, exist_ok=True)
            with open(time_dir+'/'+task+'.json', 'w') as f:
                json.dump({'time' : int(time_end-time_start)}, f)

if __name__ == "__main__":
    args = mytools.parse_args()
    one_method(args)
