import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from typing import List, Optional, Tuple, Union
import KD_loss

import time
import math
import copy

import itertools

import bart

class KD_model(nn.Module):
    def __init__(self, conf, task, num_labels):
        super(KD_model, self).__init__()
        sconfig = AutoConfig.from_pretrained(conf['student'], num_labels=num_labels, finetuning_task=task)
        tconfig = AutoConfig.from_pretrained(conf['teacher'], num_labels=num_labels, finetuning_task=task)
        self.conf = conf
        self.task = task
        self.model_name = conf['model_name'] if 'model_name' in conf else None

        if 'lambdas' in conf:
            self.lambdas = conf['lambdas']
        
        if self.model_name == 'bart':
            self.student = bart.BartForSequenceClassificationEOS.from_pretrained(
                conf['student'],
                config=sconfig
            )
            self.teacher = bart.BartForSequenceClassificationEOS.from_pretrained(
                conf['teacher'],
                config=tconfig
            )
        else:
            self.student = AutoModelForSequenceClassification.from_pretrained(
                conf['student'],
                config=sconfig
            )
            self.teacher = AutoModelForSequenceClassification.from_pretrained(
                conf['teacher'],
                config=tconfig
            )
        # freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

    def loss_kd(self, teacher_logit, student_logit):
        if self.task == 'stsb':
            loss = torch.nn.functional.mse_loss(teacher_logit, student_logit)
        else:
            tp = nn.functional.softmax(teacher_logit, dim=1)
            sp = nn.functional.softmax(student_logit, dim=1)
            loss = KD_loss.KL_divergence(tp, sp, self.conf['batch_size'])
        return loss

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if self.model_name == 'bart':
            soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        else:
            soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        lkd = self.loss_kd(soft_label.logits, outputs.logits)
        loss = self.lambdas[0]*outputs.loss+self.lambdas[1]*lkd
        losses = [outputs.loss.detach().clone(), lkd.detach().clone()]
        return outputs, loss, losses

class ProKD_model(KD_model):
    def __init__(self, conf, task, num_labels):
        super(ProKD_model, self).__init__(conf, task, num_labels)

class RAIL_l_model(KD_model):
    def __init__(self, conf, task, num_labels):
        super().__init__(conf, task, num_labels)
        # random
        self.W_t = [torch.nn.Linear(in_features=self.teacher.config.hidden_size, out_features=conf['linear']) for i in range(self.teacher.config.num_hidden_layers)]
        self.W_s = [torch.nn.Linear(in_features=self.student.config.hidden_size, out_features=conf['linear']) for i in range(self.student.config.num_hidden_layers)]
        if self.conf['linear_method'] == 'pretrained':
            # linear mapping is already trained
            # we can save and load list of tensor such as [tensor, tensor, tensor, ...]
            W = torch.load(conf['W'])
            self.W_t = W['t']
            self.W_s = W['s']

        if self.conf['linear_method'] != 'training' and self.conf['linear_method'] != 'pretraining':
            # training : linear mapping will be trained while student is training
            # pretraining : only linear mapping will be trained. 
            for W_t in self.W_t:
                for param in W_t.parameters():
                    param.requires_grad = False
            for W_s in self.W_s:
                for param in W_s.parameters():
                    param.requires_grad = False
        
        if self.conf['lienar_method'] == 'pretraining':
            for param in self.student.parameters():
                param.requires_grad = False

        # freeze teacher
        # for param in self.teacher.parameters():
        #     param.requires_grad = False
    
    def set_accelerator(self, accelerator):
        self.accelerator = accelerator

    def set_selected(self, selected):
        self.selected = selected

    def one_layer_loss(self, selected, hs_s, W_s):
        # shape is (batch_size, 768 or so)
        h_bar_t = torch.mean(self.hs_t[selected], 1)
        h_bar_s = torch.mean(hs_s, 1)

        # shape is (batch_size, conf['linear'])
        h_hat_t = self.W_t[selected](h_bar_t)
        h_hat_s = W_s(h_bar_s)

        # shape is (batch_size, 1)
        deno_t = torch.linalg.norm(h_hat_t, dim=1).reshape(-1,1)
        deno_s = torch.linalg.norm(h_hat_s, dim=1).reshape(-1,1)
        #shape is (batch_size) -> (1) by meaning
        return torch.linalg.norm(h_hat_t/deno_t - h_hat_s/deno_s, dim=1).mean()

    def RAIL_l_loss(self, soft_label, outputs):
        #start = time.time()
        self.hs_t = soft_label.hidden_states[1:]
        hs_s = outputs.hidden_states[1:]
        losses = list(map(self.one_layer_loss, self.selected, hs_s, self.W_s))
        #end = time.time()
        #print('rail_l ', end-start)
        del self.hs_t
        torch.cuda.empty_cache()
        return sum(losses)/len(self.selected)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, output_hidden_states=True)
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        loss = self.lambdas[0]*outputs.loss+self.lambdas[1]*self.loss_kd(soft_label.logits, outputs.logits)+self.lambdas[2]*self.RAIL_l_loss(soft_label, outputs)
        return outputs, loss

class RAIL_c_model(KD_model):
    def __init__(self, conf, task, num_labels):
        super().__init__(conf, task, num_labels)
        # random
        self.W_t = torch.nn.Linear(in_features=self.teacher.config.hidden_size*self.student.config.num_hidden_layers, out_features=conf['linear'])
        self.W_s = torch.nn.Linear(in_features=self.student.config.hidden_size*self.student.config.num_hidden_layers, out_features=conf['linear'])
        if self.conf['linear_method'] == 'pretrained':
            # linear mapping is already trained
            W = torch.load(conf['W'])
            self.W_t = W['t']
            self.W_s = W['s']
        if self.conf['linear_method'] != 'training' and self.conf['linear_method'] != 'pretraining':
            # training : linear mapping will be trained while student is training
            # pretraining : only linear mapping will be trained. 
            for param in self.W_t.parameters():
                param.requires_grad = False
            for param in self.W_s.parameters():
                param.requires_grad = False
        if self.conf['linear_method'] == 'pretraining':
            for param in self.student.parameters():
                param.requires_grad = False
    
    def set_accelerator(self, accelerator):
        self.accelerator = accelerator
    
    def set_selected(self, selected):
        self.selected = selected
    
    def RAIL_c_loss(self, soft_label, outputs):
        # tuple of tensor
        # hs_*[0] is embedding layer output
        hs_t = soft_label.hidden_states[1:]
        hs_s = outputs.hidden_states[1:]
        h_bar_t = list(map(self.mean_pooling, hs_t))
        h_bar_s = list(map(self.mean_pooling, hs_s))

        # shape is (batch size, len(selected)*hidden_size)
        h_bar_t_c = torch.empty(0, device=self.accelerator.device)
        h_bar_s_c = torch.empty(0, device=self.accelerator.device)
        for i, s in enumerate(self.selected):
            h_bar_t_c = torch.cat([h_bar_t_c, h_bar_t[s]], dim=1)
            h_bar_s_c = torch.cat([h_bar_s_c, h_bar_s[i]], dim=1)

        assert h_bar_t_c.shape == torch.Size([self.conf['batch_size'], len(self.selected)*self.teacher.config.hidden_size]), 'h_bar_t_c size is not correct'
        assert h_bar_s_c.shape == torch.Size([self.conf['batch_size'], len(self.selected)*self.student.config.hidden_size]), 'h_bar_s_c size is not correct'

        # shape is (batch_size, conf['linear'])
        h_hat_t = self.W_t(h_bar_t_c)
        h_hat_s = self.W_s(h_bar_s_c)

        # shape is (batch_size, 1)
        deno_t = torch.linalg.norm(h_hat_t, dim=1).reshape(-1,1)
        deno_s = torch.linalg.norm(h_hat_s, dim=1).reshape(-1,1)

        return torch.linalg.norm(h_hat_t/deno_t - h_hat_s/deno_s, dim=1).mean()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, output_hidden_states=True)
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        lkd = self.loss_kd(soft_label.logits, outputs.logits)
        lrail = self.RAIL_c_loss(soft_label, outputs)
        loss = self.lambdas[0]*outputs.loss+self.lambdas[1]*lkd+self.lambdas[2]*lrail
        losses = [outputs.loss.detach().clone(), lkd.detach().clone(), lrail.detach().clone()]
        return outputs, loss, losses

    def mean_pooling(self, hidden_states):
        # shape (batch_size, sequence_length, hidden_size) -> (batch_size, hidden_size)
        return torch.mean(hidden_states, 1)

    def ILD_loss(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # use when you need only intermadiate layer loss
        soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, output_hidden_states=True)
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        loss = self.RAIL_c_loss(soft_label, outputs)
        return loss

class Bart_RAIL_c_model(KD_model):
    def __init__(self, conf, task, num_labels):
        super().__init__(conf, task, num_labels)
        # only training is supported
        self.We_t = torch.nn.Linear(in_features=self.teacher.config.d_model*self.student.config.encoder_layers, out_features=conf['linear'])
        self.We_s = torch.nn.Linear(in_features=self.student.config.d_model*self.student.config.encoder_layers, out_features=conf['linear'])
        self.Wd_t = torch.nn.Linear(in_features=self.teacher.config.d_model*self.student.config.decoder_layers, out_features=conf['linear'])
        self.Wd_s = torch.nn.Linear(in_features=self.student.config.d_model*self.student.config.decoder_layers, out_features=conf['linear'])
    
    def set_accelerator(self, accelerator):
        self.accelerator = accelerator
    
    def set_selected(self, selected):
        self.selected = selected
    
    def RAIL_c_loss(self, hidden_t, hidden_s, selected, W_t, W_s):
        # tuple of tensor
        # hidden_* is output.*_hidden_states
        # selected : selected decoder or encoder layers index (self.selected has 2 list. one is for encoder, and the other is for encoder)
        h_bar_t = list(map(self.mean_pooling, hidden_t))
        h_bar_s = list(map(self.mean_pooling, hidden_s))

        # shape is (batch size, len(selected)*hidden_size)
        h_bar_t_c = torch.empty(0, device=self.accelerator.device)
        h_bar_s_c = torch.empty(0, device=self.accelerator.device)
        for i, s in enumerate(selected):
            h_bar_t_c = torch.cat([h_bar_t_c, h_bar_t[s]], dim=1)
            h_bar_s_c = torch.cat([h_bar_s_c, h_bar_s[i]], dim=1)

        assert h_bar_t_c.shape == torch.Size([self.conf['batch_size'], len(selected)*self.teacher.config.d_model]), 'h_bar_t_c size is not correct'
        assert h_bar_s_c.shape == torch.Size([self.conf['batch_size'], len(selected)*self.student.config.d_model]), 'h_bar_s_c size is not correct'

        # shape is (batch_size, conf['linear'])
        h_hat_t = W_t(h_bar_t_c)
        h_hat_s = W_s(h_bar_s_c)

        # shape is (batch_size, 1)
        deno_t = torch.linalg.norm(h_hat_t, dim=1).reshape(-1,1)
        deno_s = torch.linalg.norm(h_hat_s, dim=1).reshape(-1,1)

        return torch.linalg.norm(h_hat_t/deno_t - h_hat_s/deno_s, dim=1).mean()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        lkd = self.loss_kd(soft_label.logits, outputs.logits)
        lrail_e = self.RAIL_c_loss(soft_label.encoder_hidden_states, outputs.encoder_hidden_states, self.selected[0], self.We_t, self.We_s)
        lrail_d = self.RAIL_c_loss(soft_label.decoder_hidden_states, outputs.decoder_hidden_states, self.selected[1], self.Wd_t, self.Wd_s)
        loss = self.lambdas[0]*outputs.loss+self.lambdas[1]*lkd+self.lambdas[2]*(lrail_d+lrail_e)
        losses = [outputs.loss.detach().clone(), lkd.detach().clone(), (lrail_d+lrail_e).detach().clone()]
        return outputs, loss, losses

    def mean_pooling(self, hidden_states):
        # shape (batch_size, sequence_length, hidden_size) -> (batch_size, hidden_size)
        return torch.mean(hidden_states, 1)

class CatILD_model(KD_model):
    # only support training
    def __init__(self, conf, task, num_labels):
        super().__init__(conf, task, num_labels)
        self.W_t = torch.nn.Linear(in_features=self.teacher.config.hidden_size*self.teacher.config.num_hidden_layers, out_features=conf['linear'])
        self.W_s = torch.nn.Linear(in_features=self.student.config.hidden_size*self.student.config.num_hidden_layers, out_features=conf['linear'])

    def set_accelerator(self, accelerator):
        self.accelerator = accelerator
    
    def CatILD_loss(self, soft_label, outputs):
        # tuple of tensor
        # hs_*[0] is embedding layer output
        hs_t = soft_label.hidden_states[1:]
        hs_s = outputs.hidden_states[1:]
        h_bar_t = list(map(self.mean_pooling, hs_t))
        h_bar_s = list(map(self.mean_pooling, hs_s))

        # shape is (batch size, len(selected)*hidden_size)
        h_bar_t_c = torch.empty(0, device=self.accelerator.device)
        h_bar_s_c = torch.empty(0, device=self.accelerator.device)
        for i in range(self.teacher.config.num_hidden_layers):
            h_bar_t_c = torch.cat([h_bar_t_c, h_bar_t[i]], dim=1)
        for i in range(self.student.config.num_hidden_layers):
            h_bar_s_c = torch.cat([h_bar_s_c, h_bar_s[i]], dim=1)

        assert h_bar_t_c.shape == torch.Size([self.conf['batch_size'], self.teacher.config.hidden_size*self.teacher.config.num_hidden_layers]), 'h_bar_t_c size is not correct'
        assert h_bar_s_c.shape == torch.Size([self.conf['batch_size'], self.student.config.hidden_size*self.student.config.num_hidden_layers]), 'h_bar_s_c size is not correct'

        # shape is (batch_size, conf['linear'])
        h_hat_t = self.W_t(h_bar_t_c)
        h_hat_s = self.W_s(h_bar_s_c)

        # shape is (batch_size, 1)
        deno_t = torch.linalg.norm(h_hat_t, dim=1).reshape(-1,1)
        deno_s = torch.linalg.norm(h_hat_s, dim=1).reshape(-1,1)

        return torch.linalg.norm(h_hat_t/deno_t - h_hat_s/deno_s, dim=1).mean()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, output_hidden_states=True)
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        lkd = self.loss_kd(soft_label.logits, outputs.logits)
        lild = self.CatILD_loss(soft_label, outputs)
        loss = self.lambdas[0]*outputs.loss+self.lambdas[1]*lkd+self.lambdas[2]*lild
        losses = [outputs.loss.detach().clone(), lkd.detach().clone(), lild.detach().clone()]
        return outputs, loss, losses

    def mean_pooling(self, hidden_states):
        # shape (batch_size, sequence_length, hidden_size) -> (batch_size, hidden_size)
        return torch.mean(hidden_states, 1)

class RKD_model(KD_model):
    def __init__(self, conf, task, num_labels):
        super().__init__(conf, task, num_labels)
        self.method = conf['rkd'] if 'rkd' in conf['rkd'] else 'd'
        self.mid = conf['mid'] if 'mid' in conf['mid'] else False
    
    def RKD_D_loss(self, rept, reps):
        # rep* : last layer's hidden states (t:teacher, s:student)
        # shape is Size(batch_size, sequence_length, hidden_size)
        rept_cls = rept[:, 0] # shape is Size(batch_size, hidden_size)
        reps_cls = reps[:, 0] # shape is Size(batch_size, hidden_size)

        def dist_func(rep_cls):
            # reshape(1, ) is for torch.stack
            return lambda ind:torch.linalg.norm(rep_cls[ind[0]]-rep_cls[ind[1]]).reshape(1,)

        bs = rept_cls.shape[0]
        ind = list(itertools.combinations(list(range(bs)), 2))
        rest = list(map(dist_func(rept_cls), ind))
        ress = list(map(dist_func(reps_cls), ind))

        rest = torch.stack(rest)
        ress = torch.stack(ress)

        mut = sum(rest)/len(rest)
        mus = sum(ress)/len(ress)
        rest = rest/mut
        ress = ress/mus

        loss = torch.nn.functional.huber_loss(ress, rest)

        return loss

    # spare
    def RKD_D_for_loss(self, rept, reps):
        # rep* : last layer's hidden states (t:teacher, s:student)
        # shape is Size(batch_size, sequence_length, hidden_size)
        rept_cls = rept[:, 0] # shape is Size(batch_size, hidden_size)
        reps_cls = reps[:, 0] # shape is Size(batch_size, hidden_size)

        bs = rept_cls.shape[0]
        ind = list(itertools.combinations(list(range(bs)), 2))
        rest = []
        ress = []
        for i in ind:
            rest.append(torch.linalg.norm(rept_cls[i[0]]-rept_cls[i[1]]).reshape(1,))
            ress.append(torch.linalg.norm(reps_cls[i[0]]-reps_cls[i[1]]).reshape(1,))

        rest = torch.stack(rest)
        ress = torch.stack(ress)

        mut = sum(rest)/len(rest)
        mus = sum(ress)/len(ress)
        rest = rest/mut
        ress = ress/mus

        loss = torch.nn.functional.huber_loss(ress, rest)

        return loss

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, output_hidden_states=True)
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        lkd = self.loss_kd(soft_label.logits, outputs.logits)
        if self.method=='d':
            lrkd = self.RKD_D_loss(soft_label.hidden_states[-1], outputs.hidden_states[-1])
            if self.mid:
                # if techer has 24 layer, hidden_states is (emb, 0, 1, 2, ..., 23). therefore, num_hidden_layers//2 is appropreate
                lrkd_mid = self.RKD_D_loss(soft_label.hidden_states[self.teacher.config.num_hidden_layers//2], outputs.hidden_states[self.student.config.num_hidden_layers//2])
        loss = self.lambdas[0]*outputs.loss+self.lambdas[1]*lkd+self.lambdas[2]*lrkd
        losses = [outputs.loss.detach().clone(), lkd.detach().clone(), lrkd.detach().clone()]
        if self.mid:
            loss += self.lambdas[3]*lrkd_mid
            losses += [lrkd_mid.detach().clone()]
        return outputs, loss, losses

class MATE_model(KD_model):
    def __init__(self, conf, task, num_labels):
        super().__init__(conf, task, num_labels)
        self.model_name = conf['model_name']
        self.generator = AutoModelForMaskedLM.from_pretrained(conf['generator'])
        self.adv = True

    def gen_output(
        self,
        input_ids_permuted: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        if self.model_name == 'bart':
            outputs = self.generator(input_ids=input_ids_permuted, attention_mask=attention_mask)
        else:
            outputs = self.generator(input_ids=input_ids_permuted, attention_mask=attention_mask, token_type_ids=token_type_ids)
        prediction_scores = outputs[0] #logit
        prediction_scores = torch.nn.functional.gumbel_softmax(prediction_scores, hard=True)
        return prediction_scores

    def bert_inps(
        self, 
        prediction_scores, 
        input_ids: Optional[torch.Tensor] = None, 
        mask_permuted: Optional[torch.Tensor] = None, 
    ):
        # only word embedding that was located in mask.
        teacher_inp = torch.matmul(prediction_scores, self.teacher.bert.embeddings.word_embeddings.weight) * mask_permuted.unsqueeze(-1)
        student_inp = torch.matmul(prediction_scores, self.student.distilbert.embeddings.word_embeddings.weight) * mask_permuted.unsqueeze(-1)

        # add word embedding that was not mask
        teacher_inp = teacher_inp + (self.teacher.bert.embeddings.word_embeddings(input_ids) * (1 - mask_permuted.unsqueeze(-1)))
        student_inp = student_inp + (self.student.distilbert.embeddings.word_embeddings(input_ids) * (1 - mask_permuted.unsqueeze(-1)))

        # ???
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        student_inp = student_inp + self.student.distilbert.embeddings.position_embeddings(position_ids)
        student_inp = self.student.distilbert.embeddings.LayerNorm(student_inp)
        student_inp = self.student.distilbert.embeddings.dropout(student_inp)

        return teacher_inp, student_inp

    def roberta_inps(
        self, 
        prediction_scores, 
        input_ids: Optional[torch.Tensor] = None, 
        mask_permuted: Optional[torch.Tensor] = None, 
    ):
        teacher_inp = torch.matmul(prediction_scores, self.teacher.roberta.embeddings.word_embeddings.weight) * mask_permuted.unsqueeze(-1)
        student_inp = torch.matmul(prediction_scores, self.student.roberta.embeddings.word_embeddings.weight) * mask_permuted.unsqueeze(-1)
        teacher_inp = teacher_inp + (self.teacher.roberta.embeddings.word_embeddings(input_ids) * (1 - mask_permuted.unsqueeze(-1)))
        student_inp = student_inp + (self.student.roberta.embeddings.word_embeddings(input_ids) * (1 - mask_permuted.unsqueeze(-1)))
        return teacher_inp, student_inp

    def bart_inps(
        self, 
        prediction_scores, 
        input_ids: Optional[torch.Tensor] = None, 
        mask_permuted: Optional[torch.Tensor] = None, 
    ):
        teacher_inp = torch.matmul(prediction_scores, self.teacher.model.shared.weight) * mask_permuted.unsqueeze(-1)
        student_inp = torch.matmul(prediction_scores, self.student.model.shared.weight) * mask_permuted.unsqueeze(-1)
        teacher_inp = teacher_inp + (self.teacher.model.shared(input_ids) * (1 - mask_permuted.unsqueeze(-1)))
        student_inp = student_inp + (self.student.model.shared(input_ids) * (1 - mask_permuted.unsqueeze(-1)))

        d_teacher_inp = self.shift_embeds(teacher_inp, self.teacher)
        d_student_inp = self.shift_embeds(student_inp, self.student)

        return (teacher_inp,  d_teacher_inp), (student_inp, d_student_inp)

    def shift_embeds(self, embeds, model):
        # model : teacher or student
        # embeds : shape is (batch_size, seqence_length, dim)
        shifted = embeds.new_zeros(embeds.shape)
        shifted[:, 1:] = embeds[:, :-1] # shift one token
        shifted[:, 0] = model.model.shared(torch.tensor([model.config.decoder_start_token_id], device=torch.cuda.current_device()))

        return shifted

    def adv_loss(
        self,
        teacher_inp,
        student_inp,
        eos_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None, 
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        # do not use labels in this
        # prediction_scores : one hot vector
        # self.***.***.embeddings.word_embeddings.weight : word embedding of all words in dictionary
        # input_ids_permuted : input_ids but it contains masked tokens

        if self.model_name == 'bart':
            teacher_logits = self.teacher(attention_mask=attention_mask, inputs_embeds=teacher_inp[0], decoder_inputs_embeds=teacher_inp[1], eos_mask=eos_mask)[0]
            student_logits = self.student(attention_mask=attention_mask, inputs_embeds=student_inp[0], decoder_inputs_embeds=student_inp[1], eos_mask=eos_mask)[0]
        else:
            teacher_logits = self.teacher(attention_mask=attention_mask, inputs_embeds=teacher_inp, token_type_ids=token_type_ids)[0]
            student_logits = self.student(attention_mask=attention_mask, inputs_embeds=student_inp)[0]
        loss = self.loss_kd(teacher_logits, student_logits)
        return loss

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None, 
        input_ids_permuted: Optional[torch.Tensor] = None, 
        mask_permuted: Optional[torch.Tensor] = None, 
        labels_permuted: Optional[torch.Tensor] = None, 
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        eos_mask = None
        if self.model_name == 'bart':
            soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            prediction_scores = self.gen_output(input_ids_permuted=input_ids_permuted, attention_mask=attention_mask)
            teacher_inp, student_inp = self.bart_inps(prediction_scores=prediction_scores, input_ids=input_ids, mask_permuted=mask_permuted)
            # teacher and student is same eos_mask and there is no differense between input_ids and input_ids_permuted
            eos_mask = input_ids.eq(self.teacher.config.eos_token_id).to(input_ids.device)
        else : 
            soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            prediction_scores = self.gen_output(input_ids_permuted=input_ids_permuted, attention_mask=attention_mask, token_type_ids=token_type_ids)
            
            if self.model_name == 'bert':
                teacher_inp, student_inp = self.bert_inps(prediction_scores=prediction_scores, input_ids=input_ids, mask_permuted=mask_permuted)
            elif self.model_name == 'roberta':
                teacher_inp, student_inp = self.roberta_inps(prediction_scores=prediction_scores, input_ids=input_ids, mask_permuted=mask_permuted)
            
        
        ladv = self.adv_loss(teacher_inp, student_inp, eos_mask, attention_mask, token_type_ids)
        loss = self.lambdas[2]*ladv
        if not self.adv:
            lkd = self.loss_kd(soft_label.logits, outputs.logits)
            loss += self.lambdas[0]*outputs.loss+self.lambdas[1]*lkd
            losses = [outputs.loss.detach().clone(), lkd.detach().clone(), ladv.detach().clone()]
        else:
            losses = [ladv.detach().clone()]
        return outputs, loss, losses

class MATEILD_model(MATE_model):
    def __init__(self, conf, task, num_labels):
        # lambda : hyperparameter in min step
        # alpha  : hyperparameter in max step
        super().__init__(conf, task, num_labels)
        self.W_t = torch.nn.Linear(in_features=self.teacher.config.hidden_size*self.teacher.config.num_hidden_layers, out_features=conf['linear'])
        self.W_s = torch.nn.Linear(in_features=self.student.config.hidden_size*self.student.config.num_hidden_layers, out_features=conf['linear'])
        self.maxILD = conf['maxILD'] if 'maxILD' in conf else False
        self.minILD = conf['minILD'] if 'minILD' in conf else False
        self.alphas = conf['alphas'] if 'alphas' in conf else [1.0]
    
    def set_accelerator(self, accelerator):
        self.accelerator = accelerator

    def mean_pooling(self, hidden_states):
        # shape (batch_size, sequence_length, hidden_size) -> (batch_size, hidden_size)
        return torch.mean(hidden_states, 1)

    def CatILD_loss(self, soft_label, outputs):
        # tuple of tensor
        # hs_*[0] is embedding layer output
        hs_t = soft_label.hidden_states[1:]
        hs_s = outputs.hidden_states[1:]
        h_bar_t = list(map(self.mean_pooling, hs_t))
        h_bar_s = list(map(self.mean_pooling, hs_s))

        # shape is (batch size, len(selected)*hidden_size)
        h_bar_t_c = torch.empty(0, device=self.accelerator.device)
        h_bar_s_c = torch.empty(0, device=self.accelerator.device)
        for i in range(self.teacher.config.num_hidden_layers):
            h_bar_t_c = torch.cat([h_bar_t_c, h_bar_t[i]], dim=1)
        for i in range(self.student.config.num_hidden_layers):
            h_bar_s_c = torch.cat([h_bar_s_c, h_bar_s[i]], dim=1)

        assert h_bar_t_c.shape == torch.Size([self.conf['batch_size'], self.teacher.config.hidden_size*self.teacher.config.num_hidden_layers]), 'h_bar_t_c size is not correct'
        assert h_bar_s_c.shape == torch.Size([self.conf['batch_size'], self.student.config.hidden_size*self.student.config.num_hidden_layers]), 'h_bar_s_c size is not correct'

        # shape is (batch_size, conf['linear'])
        h_hat_t = self.W_t(h_bar_t_c)
        h_hat_s = self.W_s(h_bar_s_c)

        # shape is (batch_size, 1)
        deno_t = torch.linalg.norm(h_hat_t, dim=1).reshape(-1,1)
        deno_s = torch.linalg.norm(h_hat_s, dim=1).reshape(-1,1)

        return torch.linalg.norm(h_hat_t/deno_t - h_hat_s/deno_s, dim=1).mean()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None, 
        input_ids_permuted: Optional[torch.Tensor] = None, 
        mask_permuted: Optional[torch.Tensor] = None, 
        labels_permuted: Optional[torch.Tensor] = None, 
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, output_hidden_states=True)
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        prediction_scores = self.gen_output(input_ids_permuted=input_ids_permuted, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.conf['model_name'] == 'bert':
            teacher_inp, student_inp = self.bert_inps(prediction_scores=prediction_scores, input_ids=input_ids, mask_permuted=mask_permuted)
        elif self.conf['model_name'] == 'roberta':
            teacher_inp, student_inp = self.roberta_inps(prediction_scores=prediction_scores, input_ids=input_ids, mask_permuted=mask_permuted)

        outputs_t_p = self.teacher(attention_mask=attention_mask, inputs_embeds=teacher_inp, token_type_ids=token_type_ids, output_hidden_states=True)
        outputs_s_p = self.student(attention_mask=attention_mask, inputs_embeds=student_inp, output_hidden_states=True, labels=labels)

        lkdp = self.loss_kd(outputs_t_p.logits, outputs_s_p.logits)
        if not self.adv:
            # minimization step
            lkd = self.loss_kd(soft_label.logits, outputs.logits)
            loss = self.lambdas[0]*outputs.loss+self.lambdas[1]*lkd*self.lambdas[2]*lkdp
            losses = [outputs.loss.detach().clone(), lkd.detach().clone(), lkdp.detach().clone()]

            if self.minILD:
                lild = self.CatILD_loss(soft_label, outputs)
                loss += self.lambdas[3]*lild
                losses += [lild]
            
        else:
            # maximization step
            loss = self.alphas[0]*lkdp
            losses = [lkdp.detach().clone()]

            if self.maxILD:
                lildp = self.CatILD_loss(outputs_t_p, outputs_s_p)
                loss += self.alphas[1]*lildp
                losses += [lildp]

        return outputs, loss, losses

class CILDA_model(MATE_model):
    def __init__(self, conf, task, num_labels):
        super().__init__(conf, task, num_labels)
        self.W_t = torch.nn.Linear(in_features=self.teacher.config.hidden_size*self.teacher.config.num_hidden_layers, out_features=conf['linear'])
        self.W_s = torch.nn.Linear(in_features=self.student.config.hidden_size*self.student.config.num_hidden_layers, out_features=conf['linear'])
        self.h_bar_t = None
        self.h_bar_s = None
        if self.conf['linear_method'] == 'pretrained':
            # linear mapping is already trained
            # we can save and load list of tensor such as [tensor, tensor, tensor, ...]
            W = torch.load(conf['W'])
            self.W_t = W['t']
            self.W_s = W['s']
        if self.conf['linear_method'] != 'training' and self.conf['linear_method'] != 'pretraining':
            # training : linear mapping will be trained while student is training
            # pretraining : only linear mapping will be trained. 
            for param in self.W_t.parameters():
                param.requires_grad = False
            for param in self.W_s.parameters():
                param.requires_grad = False
        if self.conf['linear_method'] == 'pretraining':
            for param in self.student.parameters():
                param.requires_grad = False
        self.alpha = copy.deepcopy(self.conf['alpha'])
        if 'alpha2' in self.conf and self.conf['alpha2'] == True:
            # normalize lcrd batch_size
            self.alpha[1] /= math.log(self.conf['batch_size'])
    # adv_loss is L_G

    def set_accelerator(self, accelerator):
        self.accelerator = accelerator

    def set_run(self, run):
        self.run = run

    def CRD_one_loss(self, batch_index):
        tau2 = 2
        # self.h_bar_t[batch_index].shape = (1, conf['linear']) and self.h_bar_s.shape = (batch_size, conf['linear'])
        deno = torch.exp(torch.nn.functional.cosine_similarity(self.h_bar_t[batch_index], self.h_bar_s, dim=1)/tau2).sum()
        # self.h_bar_t[batch_index].shape = (1, conf['linear']) and self.h_bar_s[batch_index].shape = (1, conf['linear'])
        nume = torch.exp(torch.nn.functional.cosine_similarity(self.h_bar_t[batch_index], self.h_bar_s[batch_index], dim=0)/tau2)
        return -torch.log(nume/deno)

    def CRD_loss(
        self,
        outputs_t,
        outputs_s,
    ):
        # shape is (batch size, num_hidden_layers*hidden_size)
        h_bar_t_c = torch.empty(0, device=self.accelerator.device)
        h_bar_s_c = torch.empty(0, device=self.accelerator.device)
        for h in outputs_t.hidden_states[1:]:
            # use CLS token representation
            h_bar_t_c = torch.cat([h_bar_t_c, h[:, 0]], dim=1)
        for h in outputs_s.hidden_states[1:]:
            h_bar_s_c = torch.cat([h_bar_s_c, h[:, 0]], dim=1)
        
        assert h_bar_t_c.shape[0] == h_bar_s_c.shape[0] and h_bar_t_c.shape[1] == self.teacher.config.hidden_size*self.teacher.config.num_hidden_layers and h_bar_s_c.shape[1] == self.student.config.hidden_size*self.student.config.num_hidden_layers, f'h_bar_t_c or h_bat_s_c size is not correct. h_bar_t_c size is ({h_bar_t_c.shape[0]}, {h_bar_t_c.shape[1]}), h_bar_s_c size is ({h_bar_s_c.shape[0]}, {h_bar_s_c.shape[1]})'

        # shape is (batch_size, conf['linear'])
        self.h_bar_t = self.W_t(h_bar_t_c)
        self.h_bar_s = self.W_s(h_bar_s_c)
        batch_indexes = list(range(self.conf['batch_size']))
        losses = list(map(self.CRD_one_loss, batch_indexes))
        return sum(losses)/len(losses)
        
    def forward(self,
        input_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None, 
        input_ids_permuted: Optional[torch.Tensor] = None, 
        mask_permuted: Optional[torch.Tensor] = None, 
        labels_permuted: Optional[torch.Tensor] = None, 
        token_type_ids: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None, 
        head_mask: Optional[torch.Tensor] = None, 
        inputs_embeds: Optional[torch.Tensor] = None, 
        output_attentions: Optional[bool] = None, 
        output_hidden_states: Optional[bool] = None, 
        return_dict: Optional[bool] = None
    ):
        soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        prediction_scores = self.gen_output(input_ids_permuted=input_ids_permuted, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.conf['model_name'] == 'bert':
            teacher_inp, student_inp = self.bert_inps(prediction_scores=prediction_scores, input_ids=input_ids, mask_permuted=mask_permuted)
        elif self.conf['model_name'] == 'roberta':
            teacher_inp, student_inp = self.roberta_inps(prediction_scores=prediction_scores, input_ids=input_ids, mask_permuted=mask_permuted)
        
        outputs_t_p = self.teacher(attention_mask=attention_mask, inputs_embeds=teacher_inp, token_type_ids=token_type_ids, output_hidden_states=True)
        outputs_s_p = self.student(attention_mask=attention_mask, inputs_embeds=student_inp, output_hidden_states=True, labels=labels)
        if self.adv:
            lkdp = self.loss_kd(outputs_t_p.logits, outputs_s_p.logits)
            lcrdp = self.CRD_loss(outputs_t_p, outputs_s_p)
            loss = self.alpha[0]*lkdp+self.alpha[1]*lcrdp
            losses = [lkdp.detach().clone(), lcrdp.detach().clone()]
        else:
            lkd = self.loss_kd(soft_label.logits, outputs.logits)
            lkdp = self.loss_kd(outputs_t_p.logits, outputs_s_p.logits)
            loss = self.conf['lambdas'][0]*outputs.loss + self.conf['lambdas'][1]*lkd + self.conf['lambdas'][2]*outputs_s_p.loss + self.conf['lambdas'][3]*lkdp
            losses = [outputs.loss.detach().clone(), lkd.detach().clone(), outputs_s_p.loss.detach().clone(), lkdp.detach().clone()]
        return outputs, loss, losses

class Bart_CILDA_model(MATE_model):
    def __init__(self, conf, task, num_labels):
        super().__init__(conf, task, num_labels)
        self.We_t = torch.nn.Linear(in_features=self.teacher.config.hidden_size*self.teacher.config.num_hidden_layers, out_features=conf['linear'])
        self.We_s = torch.nn.Linear(in_features=self.student.config.hidden_size*self.student.config.num_hidden_layers, out_features=conf['linear'])
        self.Wd_t = torch.nn.Linear(in_features=self.teacher.config.hidden_size*self.teacher.config.num_hidden_layers, out_features=conf['linear'])
        self.Wd_s = torch.nn.Linear(in_features=self.student.config.hidden_size*self.student.config.num_hidden_layers, out_features=conf['linear'])
        self.h_bar_t = None
        self.h_bar_s = None

        self.alpha = copy.deepcopy(self.conf['alpha'])
        if 'alpha2' in self.conf and self.conf['alpha2'] == True:
            # normalize lcrd batch_size
            self.alpha[1] /= math.log(self.conf['batch_size'])

    def set_accelerator(self, accelerator):
        self.accelerator = accelerator

    def set_run(self, run):
        self.run = run

    def CRD_one_loss(self, batch_index):
        tau2 = 2
        # self.h_bar_t[batch_index].shape = (1, conf['linear']) and self.h_bar_s.shape = (batch_size, conf['linear'])
        deno = torch.exp(torch.nn.functional.cosine_similarity(self.h_bar_t[batch_index], self.h_bar_s, dim=1)/tau2).sum()
        # self.h_bar_t[batch_index].shape = (1, conf['linear']) and self.h_bar_s[batch_index].shape = (1, conf['linear'])
        nume = torch.exp(torch.nn.functional.cosine_similarity(self.h_bar_t[batch_index], self.h_bar_s[batch_index], dim=0)/tau2)
        return -torch.log(nume/deno)

    def CRD_loss(
        self,
        hidden_t,
        hidden_s,
        W_t,
        W_s,
        num_layers
    ):
        # shape is (batch size, num_hidden_layers*hidden_size)
        h_bar_t_c = torch.empty(0, device=self.accelerator.device)
        h_bar_s_c = torch.empty(0, device=self.accelerator.device)
        for h in hidden_t:
            # use CLS token representation
            h_bar_t_c = torch.cat([h_bar_t_c, h[:, 0]], dim=1)
        for h in hidden_s:
            h_bar_s_c = torch.cat([h_bar_s_c, h[:, 0]], dim=1)
        
        assert h_bar_t_c.shape[0] == h_bar_s_c.shape[0] and h_bar_t_c.shape[1] == self.teacher.config.d_model*num_layers[0] and h_bar_s_c.shape[1] == self.student.config.d_model*num_layers[1], f'h_bar_t_c or h_bat_s_c size is not correct. h_bar_t_c size is ({h_bar_t_c.shape[0]}, {h_bar_t_c.shape[1]}), h_bar_s_c size is ({h_bar_s_c.shape[0]}, {h_bar_s_c.shape[1]})'

        # shape is (batch_size, conf['linear'])
        self.h_bar_t = W_t(h_bar_t_c)
        self.h_bar_s = W_s(h_bar_s_c)
        batch_indexes = list(range(self.conf['batch_size']))
        losses = list(map(self.CRD_one_loss, batch_indexes))
        return sum(losses)/len(losses)

    def forward(self,
        input_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None, 
        input_ids_permuted: Optional[torch.Tensor] = None, 
        mask_permuted: Optional[torch.Tensor] = None, 
        labels_permuted: Optional[torch.Tensor] = None, 
        token_type_ids: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None, 
        head_mask: Optional[torch.Tensor] = None, 
        inputs_embeds: Optional[torch.Tensor] = None, 
        output_attentions: Optional[bool] = None, 
        output_hidden_states: Optional[bool] = None, 
        return_dict: Optional[bool] = None
    ):
        soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        prediction_scores = self.gen_output(input_ids_permuted=input_ids_permuted, attention_mask=attention_mask, token_type_ids=token_type_ids)
        teacher_inp, student_inp = self.bart_inps(prediction_scores=prediction_scores, input_ids=input_ids, mask_permuted=mask_permuted)
        eos_mask = input_ids.eq(self.teacher.config.eos_token_id).to(input_ids.device)

        outputs_t_p = self.teacher(attention_mask=attention_mask, inputs_embeds=teacher_inp[0], decoder_inputs_embeds=teacher_inp[1], eos_mask=eos_mask, output_hidden_states=True)
        outputs_s_p = self.student(attention_mask=attention_mask, inputs_embeds=student_inp[0], decoder_inputs_embeds=student_inp[1], eos_mask=eos_mask, output_hidden_states=True, labels=labels)

        if self.adv:
            lkdp = self.loss_kd(outputs_t_p.logits, outputs_s_p.logits)

            lcrdp_e = self.CRD_loss(outputs_t_p.encoder_hidden_states[1:], outputs_s_p.encoder_hidden_states[1:], self.We_t, self.We_s, [self.teacher.config.encoder_layers, self.student.config.encoder_layers])
            lcrdp_d = self.CRD_loss(outputs_t_p.decoder_hidden_states[1:], outputs_s_p.decoder_hidden_states[1:], self.Wd_t, self.Wd_s, [self.teacher.config.decoder_layers, self.student.config.decoder_layers])

            lcrdp = lcrdp_e+lcrdp_d

            loss = self.alpha[0]*lkdp+self.alpha[1]*lcrdp
            losses = [lkdp.detach().clone(), lcrdp.detach().clone()]
        else:
            lkd = self.loss_kd(soft_label.logits, outputs.logits)
            lkdp = self.loss_kd(outputs_t_p.logits, outputs_s_p.logits)
            loss = self.conf['lambdas'][0]*outputs.loss + self.conf['lambdas'][1]*lkd + self.conf['lambdas'][2]*outputs_s_p.loss + self.conf['lambdas'][3]*lkdp
            losses = [outputs.loss.detach().clone(), lkd.detach().clone(), outputs_s_p.loss.detach().clone(), lkdp.detach().clone()]
        return outputs, loss, losses

class CILDA_minILD_model(CILDA_model):
    def __init__(self, conf, task, num_labels):
        super().__init__(conf, task, num_labels)
        self.lambdas = []
        for nume, deno in zip(self.conf['lambda_nume'], self.conf['lambda_deno']):
            self.lambdas.append(nume/deno)
        if 'alpha2' in self.conf and self.conf['alpha2']:
            self.lambdas[2] /= math.log(self.conf['batch_size'])
            self.lambdas[5] /= math.log(self.conf['batch_size'])

    
    def forward(self,
        input_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None, 
        input_ids_permuted: Optional[torch.Tensor] = None, 
        mask_permuted: Optional[torch.Tensor] = None, 
        labels_permuted: Optional[torch.Tensor] = None, 
        token_type_ids: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None, 
        head_mask: Optional[torch.Tensor] = None, 
        inputs_embeds: Optional[torch.Tensor] = None, 
        output_attentions: Optional[bool] = None, 
        output_hidden_states: Optional[bool] = None, 
        return_dict: Optional[bool] = None
    ):
        soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, output_hidden_states=True)
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)

        prediction_scores = self.gen_output(input_ids_permuted=input_ids_permuted, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.conf['model_name'] == 'bert':
            teacher_inp, student_inp = self.bert_inps(prediction_scores=prediction_scores, input_ids=input_ids, mask_permuted=mask_permuted)
        elif self.conf['model_name'] == 'roberta':
            teacher_inp, student_inp = self.roberta_inps(prediction_scores=prediction_scores, input_ids=input_ids, mask_permuted=mask_permuted)
        
        outputs_t_p = self.teacher(attention_mask=attention_mask, inputs_embeds=teacher_inp, token_type_ids=token_type_ids, output_hidden_states=True)
        outputs_s_p = self.student(attention_mask=attention_mask, inputs_embeds=student_inp, labels=labels, output_hidden_states=True)

        lkdp = self.loss_kd(outputs_t_p.logits, outputs_s_p.logits)
        lcrdp = self.CRD_loss(outputs_t_p, outputs_s_p)
        if self.adv:
            loss = self.alpha[0]*lkdp+self.alpha[1]*lcrdp
            losses = [lkdp.detach().clone(), lcrdp.detach().clone()]

        else:
            lkd = self.loss_kd(soft_label.logits, outputs.logits)
            lcrd = self.CRD_loss(soft_label, outputs)
            loss = self.lambdas[0]*outputs.loss + self.lambdas[1]*lkd + self.lambdas[2]*lcrd + self.lambdas[3]*outputs_s_p.loss + self.lambdas[4]*lkdp + self.lambdas[5]*lcrdp
            losses = [outputs.loss.detach().clone(), lkd.detach().clone(), lcrd.detach().clone(), outputs_s_p.loss.detach().clone(), lkdp.detach().clone(), lcrdp.detach().clone()]

        return outputs, loss, losses