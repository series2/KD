import os
import json
import argparse
import math
import time
import logging
import random
import copy
import pickle

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


def calc_kl(args):
    outer_accelerator = Accelerator(log_with='all', logging_dir=conf['outdir'])
    conf = mytools.get_conf(args)
    task = conf['task']
    if conf['task']=='stsb':
        is_regression = True

    tokenizer = AutoTokenizer.from_pretrained(conf['tokenizer'])
    sentence1_key, sentence2_key = task_to_keys[task]

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
    raw_datasets = load_dataset("glue", task)

    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1


    with outer_accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )
    if conf['pad_to_max_length']:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if outer_accelerator.use_fp16 else None))

    valid_dataset = processed_datasets["validation_matched" if task == "mnli" else "validation"]
    valid_dataloader = DataLoader(valid_dataset, collate_fn=data_collator, batch_size=conf['batch_size'])
    config = AutoConfig.from_pretrained(conf['student'], num_labels=num_labels, finetuning_task=conf['task'])


    valid_losses = []
    for i in range(conf['num_of_ex']):
        accelerator = Accelerator(log_with='all', logging_dir=conf['outdir'])
        checkpoint = torch.load(conf['check_path'][i])
        model = KD_model.KD_model(conf, task, num_labels)
        model.student.load_state_dict(checkpoint['student'])

        model, valid_dataloader = accelerator.prepare(
            model, valid_dataloader
        )

        model.module.teacher.eval()
        model.module.student.eval()

        loss_sum = torch.tensor(0, dtype=torch.float, requires_grad=False).to(accelerator.device)
        for step, batch in enumerate(valid_dataloader):
            with torch.no_grad():
                outputs, loss, losses = model(**batch)
                loss_sum += losses[1].detach().clone() # only use lkd

        valid_loss = accelerator.gather(loss_sum)
        valid_loss = torch.sum(valid_loss)/len(valid_dataloader)
        valid_losses.append(valid_loss.item())
        del accelerator
        torch.cuda.empty_cache()

    with open(conf['outdir']+'/'+task+'.pickle', 'wb') as f:
        pickle.dump(valid_losses, f)

if __name__ == "__main__":
    args = mytools.parse_args()
    calc_kl(args)