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
from adv_tools import mask_tokens, mask_tokens_fix
import mytools

import neptune.new as neptune

import tracemalloc

ILDs = ['RAIL_l', 'RAIL_c', 'CILDA', 'CILDA_minILD', 'MATEILD', 'Bart_CILDA']
ADVs = ['MATE', 'CILDA']

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
    print('train function. device = ', accelerator.device)
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
    print('neputune is initialized. device = ', accelerator.device)
    
    # accelerator.wait_for_everyone()
    learning_log = {}
    no_decay = ["bias", "LayerNorm.weight"]

    # set optimizer parameter
    optimizer_grouped_parameters = admin.opt_parameter(model)
    g_optimizer_grouped_parameters = admin.g_opt_parameter(model)
    # set optimizer parameter end
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    g_optimizer = torch.optim.AdamW(g_optimizer_grouped_parameters, lr=5e-7, maximize=True)

    lr_scheduler = get_scheduler(
        name=conf['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=conf['num_warmup_steps'],
    )
    g_lr_scheduler = get_scheduler(
        name=conf['lr_scheduler_type'],
        optimizer=g_optimizer,
        num_warmup_steps=conf['num_warmup_steps'],
    )
    model, optimizer, g_optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, g_optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )
    print('model is prepared')
    if ILD:
        model.module.set_accelerator(accelerator)
    total_batch_size = conf['batch_size'] * accelerator.num_processes
    
    logger.info(f'training, lr : {lr}, ex_num : {i}')
    time_start = int(time.time())

    best_metric = 0
    n_student_iter = conf['n_student_iter'] if 'n_student_iter' in conf else 100
    n_generator_iter = conf['n_generator_iter'] if 'n_generator_iter' in conf else 10
    idx_pseudo = 0
    n_repeat_batch = n_generator_iter + n_student_iter

    for epoch in range(conf['epochs']):
        one_epoch_log = {}
        loss_sum = torch.tensor(0, dtype=torch.float, requires_grad=False).to(accelerator.device)
        g_loss_sum = torch.tensor(0, dtype=torch.float, requires_grad=False).to(accelerator.device)
        if accelerator.is_main_process:
            print(f'epoch {epoch} : training')
        # accelerator.wait_for_everyone()
        g_count = 0
        s_count = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                if idx_pseudo % n_repeat_batch < n_generator_iter:
                    # train generator
                    admin.set_adv(model.module, True)
                    admin.g_train_mode(model)
                    g_optimizer.zero_grad()
                    outputs, loss, losses = model(**batch)
                    if accelerator.is_main_process:
                        admin.log_losses(run, losses, 'train')
                    accelerator.backward(loss)
                    g_loss_sum += loss.detach().clone()
                    g_optimizer.step()
                    g_lr_scheduler.step()
                    g_count += (1*conf['device_num'])
                elif idx_pseudo % n_repeat_batch < (n_generator_iter + n_student_iter):
                    # train student
                    admin.set_adv(model.module, False)
                    admin.s_train_mode(model)
                    optimizer.zero_grad()
                    outputs, loss, losses = model(**batch)
                    if accelerator.is_main_process:
                        admin.log_losses(run, losses, 'train')
                    accelerator.backward(loss)
                    loss_sum += loss.detach().clone()
                    optimizer.step()
                    lr_scheduler.step()
                    s_count +=(1*conf['device_num'])
                idx_pseudo +=1
        # integrate loss
        if accelerator.is_main_process:
            print('integrating train loss')
        # accelerator.wait_for_everyone()
        
        train_loss = accelerator.gather(loss_sum)
        if s_count != 0:
            train_loss = torch.sum(train_loss)/s_count
        else:
            train_loss = torch.sum(train_loss)
        one_epoch_log['train_loss'] = train_loss.item()

        g_train_loss = accelerator.gather(g_loss_sum)
        if g_count != 0:
            g_train_loss = torch.sum(g_train_loss)/g_count
        else:
            g_train_loss = torch.sum(g_train_loss)
        one_epoch_log['g_train_loss'] = g_train_loss.item()
        # integrate loss end
        
        admin.eval_mode(model)
        admin.set_adv(model.module, False)
        samples_seen = 0
        loss_sum = torch.tensor(0, dtype=torch.float, requires_grad=False).to(accelerator.device)
        if accelerator.is_main_process:
            print(f'epoch {epoch} : validation')
        # accelerator.wait_for_everyone()
        for step, batch in enumerate(valid_dataloader):
            with torch.no_grad():
                outputs, loss, losses = model(**batch)
                if accelerator.is_main_process:
                    admin.log_losses(run, losses, 'valid')
                loss_sum += loss.detach().clone()
            
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
        # accelerator.wait_for_everyone()

        valid_loss = accelerator.gather(loss_sum)
        valid_loss = torch.sum(valid_loss)/len(valid_dataloader)
        one_epoch_log['valid_loss'] = valid_loss.item()
        
        del outputs, predictions, references
        torch.cuda.empty_cache()
        
        # accelerator.wait_for_everyone()
        eval_metric = metric.compute()
        if accelerator.is_main_process:
            run['train_loss'].log(train_loss.item())
            run['g_train_loss'].log(g_train_loss.item())
            run['valid_loss'].log(valid_loss.item())
            run[task_to_metrics[task]].log(eval_metric[task_to_metrics[task]])
            run['gpu_memory_allocated'].log(torch.cuda.memory_allocated(torch.cuda.current_device()))
            run['gpu_memory_reserved'].log(torch.cuda.memory_reserved(torch.cuda.current_device()))
        # checkpoint
        save_check = conf['save_check'] if 'save_check' in conf else True
        if accelerator.is_main_process and save_check:
            if eval_metric[task_to_metrics[task]] > best_metric:
                print('best check')
            # if 1:
                if 'outdir' in conf.keys():
                    check_dir = conf['outdir']+'/checkpoints/'+task+'/'+str(lr)+'/'+str(i)
                    os.makedirs(check_dir, exist_ok=True)
                    model_checkpoint = admin.checkpoint(model.module)
                    checkpoint = {
                        "epoch" : epoch,
                        "optimizer": optimizer.state_dict(),
                        "g_optimizer": g_optimizer.state_dict(),
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
                    del checkpoint
                    torch.cuda.empty_cache()
        
        accelerator.wait_for_everyone()
        one_epoch_log = one_epoch_log | eval_metric
        learning_log[epoch] = one_epoch_log
        logger.info(f"epoch {epoch}: {learning_log[epoch]}, time: {int(time.time()-time_start)} seconds.")
    
    if accelerator.is_main_process:
        run.stop()
    # accelerator.wait_for_everyone()

    df_log = pd.DataFrame.from_dict(learning_log, orient='index')
    # return last epoch's model and eval_metric
    return model, eval_metric, df_log
    #returned_model = model.module
    #return returned_model, eval_metric, df_log

def do_one_task(conf, task):
    #tracemalloc.start(15)

    ADV = conf['method'] in ADVs
    ILD = conf['method'] in ILDs

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

    # outer_accelerator.wait_for_everyone()
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
    mask_fix = conf['mask_fix'] if 'mask_fix' in conf else False
# --------------------------------------------------------- #
    def preprocess_function_adv(examples):
        # Tokenize the texts
        padding = conf['pad_to_max_length']
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=conf['max_length'], truncation=True, return_tensors="np")

        if mask_fix:
            input_ids_permuted, labels_permuted, mask_permuted = mask_tokens_fix(torch.from_numpy(result['input_ids']).clone(), tokenizer, conf['mlm_prob'])
        else:
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
    gradient_accumulation_steps = conf['num_accumulation_steps'] if 'num_accumulation_steps' in conf else 1
    start_num = conf['start_num'] if 'start_num' in conf else 0
    for lr in conf['lr']:
        logger.info(f'learning rate is {lr}.')
        print('lr = ', lr)
        for i in range(start_num, conf['num_of_ex']):
            print('experiment ', i)
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
                processed_datasets = raw_datasets.map(
                    preprocess_function_adv,
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
                train_dataset, shuffle=True, collate_fn=data_collator, batch_size=conf['batch_size'], drop_last=True
            )
            valid_dataloader = DataLoader(valid_dataset, collate_fn=data_collator, batch_size=conf['batch_size'], drop_last=True)
            logger.info('loading model')
            if conf['method'] == 'MATE':
                model = KD_model.MATE_model(conf, task, num_labels)
                admin = KD_admin.MATE_admin(conf)
            if conf['method'] == 'CILDA':
                model = KD_model.CILDA_model(conf, task, num_labels)
                admin = KD_admin.CILDA_admin(conf)
            if conf['method'] == 'Bart_CILDA':
                model = KD_model.Bart_CILDA_model(conf, task, num_labels)
                admin = KD_admin.Bart_CILDA_admin(conf)
            if conf['method'] == 'CILDA_minILD':
                model = KD_model.CILDA_minILD_model(conf, task, num_labels)
                admin = KD_admin.CILDA_minILD_admin(conf)
            if conf['method'] == 'MATEILD':
                model = KD_model.MATEILD_model(conf, task, num_labels)
                admin = KD_admin.MATEILD_admin(conf)
            # loading model end
            returned_model, eval_metric, df_log = train(model, admin, task, conf, accelerator, num_labels, train_dataloader, valid_dataloader, metric, is_regression, lr, i)
            del model
            torch.cuda.empty_cache()
            if accelerator.is_main_process:
                df_logs = pd.concat([df_logs, df_log])
                if 'outdir' in conf.keys():
                    df_logs.to_csv(df_logs_dir+'/'+str(lr)+'_'+str(i)+'.csv')

            # accelerator.wait_for_everyone()

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
                    outputs, loss = returned_model(**batch)
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
                if 'outdir' in conf.keys() and accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(returned_model)
                    unwrapped_model.to('cpu')
                    torch.cuda.empty_cache()
                    best_model = unwrapped_model
                    del unwrapped_model
                # accelerator.wait_for_everyone()

            # if accelerator.is_main_process:
            #     print('before delete model.', GPUtil.showUtilization())
            del returned_model
            torch.cuda.empty_cache()

            # if accelerator.is_main_process:
            #     print('after delete model.', GPUtil.showUtilization())

            # accelerator.wait_for_everyone()
            del accelerator, metric, train_dataloader, valid_dataloader, train_dataset, valid_dataset
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
        start_num = 0
            

            
    if 'outdir' in conf.keys() and outer_accelerator.is_main_process:
        save_dir = models_dir+'/'+str(best_lr)+'_'+str(best_i)
        os.makedirs(save_dir, exist_ok=True)
        admin.save(best_model, save_dir, outer_accelerator)
            
        del best_model
        torch.cuda.empty_cache()
        if outer_accelerator.is_main_process:
            tokenizer.save_pretrained(save_dir)
            eval_metrics.to_csv(results_dir+task+'.csv')
    
    outer_accelerator.wait_for_everyone()

    # # --------------------------- debug
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('traceback')
    # if outer_accelerator.is_main_process:
    #     print("[ Top 10 ]")
    #     for stat in top_stats[:10]:
    #         print(stat)
    #         for line in stat.traceback.format():
    #             print(line)
    #         print("=====")
    # # -----------------------------

    del outer_accelerator
    torch.cuda.empty_cache()
    #print('do_one_task ends.', GPUtil.showUtilization())
    

def one_method(args):
    conf = mytools.get_conf(args)

    tasks = conf['tasks']
    if 'random_state_dir' in conf:
        random_state = torch.load(conf['random_state_dir'])
        random.setstate(random_state['random'])
        np.random.set_state(random_state['np_random'])
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
