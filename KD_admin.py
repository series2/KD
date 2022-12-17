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
import KD_model

import time
import random

no_decay = ["bias", "LayerNorm.weight"]

class normal_admin():
    def __init__(self, conf):
        self.conf = conf
    
    def opt_parameter(self, model):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters
    
    def log_losses(self, run, loss, mode):
        run[mode+'losses/lce'].log(loss.item())

    def train_mode(self, model):
        # model : accelerate prepared
        model.module.train()
    
    def eval_mode(self, model):
        model.module.eval()
    
    def checkpoint(self, model):
        # model is model.module
        return {'model': model.state_dict()}
    
    def save(self, model, dir, accelerator):
        model.save_pretrained(
            dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )

class KD_admin():
    def __init__(self, conf):
        self.conf = conf
    
    def opt_parameter(self, model):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.student.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            },
            {
                "params": [p for n, p in model.student.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters
     
    def log_losses(self, run, losses, mode):
        run[mode+'losses/lce'].log(losses[0].item())
        run[mode+'losses/lkd'].log(losses[1].item())

    def train_mode(self, model):
        # model : accelerate prepared
        model.module.teacher.eval()
        model.module.student.train()
    
    def eval_mode(self, model):
        model.module.teacher.eval()
        model.module.student.eval()

    def checkpoint(self, model):
        # model is model.module
        return {'student': model.student.state_dict()}
    
    def save(self, model, dir, accelerator):
        model.student.save_pretrained(
            dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )

class RKD_admin(KD_admin):
    def __init__(self, conf):
        self.conf = conf
    
    def log_losses(self, run, losses, mode):
        run[mode+'losses/lce'].log(losses[0].item())
        run[mode+'losses/lkd'].log(losses[1].item())
        run[mode+'losses/lrkd'].log(losses[2].item())

class ProKD_admin(KD_admin):
    def __init__(self, conf):
        self.conf = conf
    
    def t_opt_parameter(self, model):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.teacher.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            },
            {
                "params": [p for n, p in model.teacher.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def t_train_mode(self, model):
        model.module.teacher.train()

    def checkpoint(self, model):
        # model is model.module
        return {'student': model.student.state_dict(), 'teacher': model.teacher.state_dict()}

class MATE_admin(KD_admin):
    # opt_parameter and checkpoint and save is same
    def __init__(self, conf):
        super().__init__(conf)

    def set_adv(self, model, adv):
        model.adv = adv

    def checkpoint(self, model):
        # model is model.module
        return {'student': model.student.state_dict(), 'generator': model.generator.state_dict()}

    def log_losses(self, run, losses, mode):
        if len(losses) == 1:
            # adv
            run[mode+'losses/lkdp'].log(losses[0].item())
        else:
            run[mode+'losses/lce'].log(losses[0].item())
            run[mode+'losses/lkd'].log(losses[1].item())
            run[mode+'losses/lkdp'].log(losses[2].item())


    def s_train_mode(self, model):
        model.module.teacher.eval()
        model.module.student.train()
        model.module.generator.eval()
    
    def g_train_mode(self, model):
        model.module.teacher.eval()
        model.module.student.eval()
        model.module.generator.train()

    def eval_mode(self, model):
        model.module.teacher.eval()
        model.module.student.eval()
        model.module.generator.eval()

    def g_opt_parameter(self, model):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.generator.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            },
            {
                "params": [p for n, p in model.generator.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

class CILDA_admin(MATE_admin):
    def __init__(self, conf):
        super().__init__(conf)
        lm = self.conf['linear_method']
        self.train_W = lm == 'pretraining' or lm == 'training'
        self.train_student = lm != 'pretraining'

    def opt_parameter(self, model):
        optimizer_grouped_parameters = []
        if self.train_student:
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.student.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.conf['wd'],
                },
                {
                    "params": [p for n, p in model.student.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                }
            ]
        return optimizer_grouped_parameters

    def g_opt_parameter(self, model):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.generator.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            },
            {
                "params": [p for n, p in model.generator.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.train_W:
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.W_t.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.conf['wd'],
                }
            ]
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.W_t.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                }
            ]
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.W_s.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.conf['wd'],
                }
            ]
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.W_s.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                }
            ]
        return optimizer_grouped_parameters

    def log_losses(self, run, losses, mode):
        if len(losses) == 2:
            # adv
            run[mode+'losses/lkdp'].log(losses[0].item())
            run[mode+'losses/lcrdp'].log(losses[1].item())

        else:
            run[mode+'losses/lce'].log(losses[0].item())
            run[mode+'losses/lkd'].log(losses[1].item())
            run[mode+'losses/lcep'].log(losses[2].item())
            run[mode+'losses/lkdp'].log(losses[3].item())

    def checkpoint(self, model):
        # model is model.module
        return {
            'student': model.student.state_dict(), 
            'generator': model.generator.state_dict(), 
            'W_t': model.W_t.state_dict(), 
            'W_s': model.W_s.state_dict()
            }

    def save(self, model, dir, accelerator):
        model.student.save_pretrained(
            dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process and self.train_W:
            W = {'t' : model.W_t, 's' : model.W_s}
            torch.save(W, dir+'/linear_mapping.pt')
            del W

class Bart_CILDA_admin(CILDA_admin):
    def __init__(self, conf):
        super().__init__(conf)

    def g_opt_parameter(self, model):
        optimizer_grouped_parameters = []
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.We_t.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.We_t.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.We_s.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.We_s.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.Wd_t.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.Wd_t.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.Wd_s.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.Wd_s.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.generator.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            },
            {
                "params": [p for n, p in model.generator.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        return optimizer_grouped_parameters
    
    def checkpoint(self, model):
        # model is model.module
        return {
            'student': model.student.state_dict(), 
            'generator': model.generator.state_dict(), 
            'We_t': model.We_t.state_dict(), 
            'We_s': model.We_s.state_dict(),
            'Wd_t': model.Wd_t.state_dict(), 
            'Wd_s': model.Wd_s.state_dict()
            }

    def save(self, model, dir, accelerator):
        model.student.save_pretrained(
            dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process and self.train_W:
            W = {'et' : model.We_t, 'es' : model.We_s, 'dt' : model.Wd_t, 'ds' : model.Wd_s}
            torch.save(W, dir+'/linear_mapping.pt')
            del W

class CILDA_minILD_admin(CILDA_admin):
    def __init__(self, conf):
        super().__init__(conf)

    def opt_parameter(self, model):
        optimizer_grouped_parameters = super().opt_parameter(model)
        if self.train_W:
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.W_t.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.conf['wd'],
                }
            ]
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.W_t.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                }
            ]
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.W_s.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.conf['wd'],
                }
            ]
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.W_s.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                }
            ]
        return optimizer_grouped_parameters

    def log_losses(self, run, losses, mode):
        if len(losses) == 2:
            # adv
            run[mode+'losses/lkdp'].log(losses[0].item())
            run[mode+'losses/lcrdp'].log(losses[1].item())

        else:
            run[mode+'losses/lce'].log(losses[0].item())
            run[mode+'losses/lkd'].log(losses[1].item())
            run[mode+'losses/lcrd'].log(losses[2].item())
            run[mode+'losses/lcep'].log(losses[3].item())
            run[mode+'losses/lkdp'].log(losses[4].item())
            run[mode+'losses/lcrdp'].log(losses[5].item())

class RAIL_l_admin(KD_admin):
    #train_mode and eval_mode is same
    def __init__(self, conf):
        super().__init__(conf)
        lm = self.conf['linear_method']
        self.train_W = lm == 'pretraining' or lm == 'training'
        self.train_student = lm != 'pretraining'

    def opt_parameter(self, model):
        optimizer_grouped_parameters = []
        if self.train_W:
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in W.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.conf['wd'],
                } for W in model.W_t
            ]
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in W.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                } for W in model.W_t
            ]
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in W.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.conf['wd'],
                } for W in model.W_s
            ]
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in W.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                } for W in model.W_s
            ]
        if self.train_student:
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.student.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.conf['wd'],
                },
                {
                    "params": [p for n, p in model.student.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                }
            ]
        return optimizer_grouped_parameters

    def select_layers(self, model):
        layers = list(range(model.module.teacher.config.num_hidden_layers))
        selected = sorted(random.sample(layers, model.module.student.config.num_hidden_layers))
        model.module.set_selected(selected)


    def log_losses(self, run, losses, mode):
        run[mode+'losses/lce'].log(losses[0].item())
        run[mode+'losses/lkd'].log(losses[1].item())
        run[mode+'losses/lrail'].log(losses[2].item())

    def checkpoint(self, model):
        checkpoint={'student' : model.student.state_dict()}
        if self.train_W:
            checkpoint['W_t'] = model.W_t.state_dict()
            checkpoint['W_s'] = model.W_s.state_dict()            
        return checkpoint

    def save(self, model, dir, accelerator):
        model.student.save_pretrained(
            dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process and self.train_W:
            W = {'t' : model.W_t, 's' : model.W_s}
            torch.save(W, dir+'/linear_mapping.pt')
            del W
    
class RAIL_c_admin(KD_admin):
    #train_mode and eval_mode is same
    def __init__(self, conf):
        super().__init__(conf)
        lm = self.conf['linear_method']
        self.train_W = lm == 'pretraining' or lm == 'training'
        self.train_student = lm != 'pretraining'

    def opt_parameter(self, model):
        optimizer_grouped_parameters = []
        if self.train_W:
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.W_t.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.conf['wd'],
                }
            ]
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.W_t.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                }
            ]
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.W_s.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.conf['wd'],
                }
            ]
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.W_s.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                }
            ]
        if self.train_student:
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.student.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.conf['wd'],
                },
                {
                    "params": [p for n, p in model.student.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                }
            ]
        return optimizer_grouped_parameters

    def select_layers(self, model):
        layers = list(range(model.module.teacher.config.num_hidden_layers))
        selected = sorted(random.sample(layers, model.module.student.config.num_hidden_layers))
        model.module.set_selected(selected)

    def log_losses(self, run, losses, mode):
        run[mode+'losses/lce'].log(losses[0].item())
        run[mode+'losses/lkd'].log(losses[1].item())
        run[mode+'losses/lrail'].log(losses[2].item())

    def checkpoint(self, model):
        checkpoint={'student' : model.student.state_dict()}
        if self.train_W:
            checkpoint['W_t'] = model.W_t.state_dict()
            checkpoint['W_s'] = model.W_s.state_dict()            
        return checkpoint

    def save(self, model, dir, accelerator):
        model.student.save_pretrained(
            dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process and self.train_W:
            W = {'t' : model.W_t, 's' : model.W_s}
            torch.save(W, dir+'/linear_mapping.pt')
            del W

class Bart_RAIL_c_admin(RAIL_c_admin):
    def __init__(self, conf):
        super().__init__(conf)

    def opt_parameter(self, model):
        optimizer_grouped_parameters = []
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.We_t.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.We_t.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.We_s.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.We_s.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.Wd_t.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.Wd_t.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.Wd_s.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.Wd_s.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.student.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            },
            {
                "params": [p for n, p in model.student.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        return optimizer_grouped_parameters

    def select_layers(self, model):
        layers_e = list(range(model.module.teacher.config.encoder_layers))
        layers_d = list(range(model.module.teacher.config.decoder_layers))
        selected_e = sorted(random.sample(layers_e, model.module.student.config.encoder_layers))
        selected_d = sorted(random.sample(layers_d, model.module.student.config.decoder_layers))
        model.module.set_selected([selected_e, selected_d])

    def checkpoint(self, model):
        checkpoint={'student' : model.student.state_dict()}
        checkpoint['We_t'] = model.We_t.state_dict()
        checkpoint['We_s'] = model.We_s.state_dict()
        checkpoint['Wd_t'] = model.Wd_t.state_dict()
        checkpoint['Wd_s'] = model.Wd_s.state_dict()   
        return checkpoint

    def save(self, model, dir, accelerator):
        model.student.save_pretrained(
            dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process and self.train_W:
            W = {'et' : model.We_t, 'es' : model.We_s, 'dt' : model.Wd_t, 'ds' : model.Wd_s}
            torch.save(W, dir+'/linear_mapping.pt')
            del W

class CatILD_admin(KD_admin):
    def __init__(self, conf):
        super().__init__(conf)

    def opt_parameter(self, model):
        optimizer_grouped_parameters = []
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.W_t.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.W_t.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.W_s.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.W_s.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.student.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            },
            {
                "params": [p for n, p in model.student.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        return optimizer_grouped_parameters

    def log_losses(self, run, losses, mode):
        run[mode+'losses/lce'].log(losses[0].item())
        run[mode+'losses/lkd'].log(losses[1].item())
        run[mode+'losses/lild'].log(losses[2].item())

    def checkpoint(self, model):
        checkpoint={'student' : model.student.state_dict()}
        checkpoint['W_t'] = model.W_t.state_dict()
        checkpoint['W_s'] = model.W_s.state_dict()            
        return checkpoint

    def save(self, model, dir, accelerator):
        model.student.save_pretrained(
            dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            W = {'t' : model.W_t, 's' : model.W_s}
            torch.save(W, dir+'/linear_mapping.pt')
            del W

class MATEILD_admin(MATE_admin):
    def __init__(self, conf):
        super().__init__(conf)
    
    def opt_parameter(self, model):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.student.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            },
            {
                "params": [p for n, p in model.student.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.W_t.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.W_t.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.W_s.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.W_s.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        return optimizer_grouped_parameters


    def g_opt_parameter(self, model):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.generator.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            },
            {
                "params": [p for n, p in model.generator.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.W_t.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.W_t.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.W_s.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.W_s.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        return optimizer_grouped_parameters

    def log_losses(self, run, losses, mode):
        # adv
        if len(losses) == 1:
            run[mode+'losses/lkdp'].log(losses[0].item())
        elif len(losses) == 2:
            run[mode+'losses/lkdp'].log(losses[0].item())
            run[mode+'losses/lildp'].log(losses[1].item())
        # min
        elif len(losses) == 3:
            run[mode+'losses/lce'].log(losses[0].item())
            run[mode+'losses/lkd'].log(losses[1].item())
            run[mode+'losses/lkdp'].log(losses[2].item())
        elif len(losses) == 4:
            run[mode+'losses/lce'].log(losses[0].item())
            run[mode+'losses/lkd'].log(losses[1].item())
            run[mode+'losses/lkdp'].log(losses[2].item())
            run[mode+'losses/lild'].log(losses[3].item())

    def checkpoint(self, model):
        # model is model.module
        return {
            'student': model.student.state_dict(), 
            'generator': model.generator.state_dict(), 
            'W_t': model.W_t.state_dict(), 
            'W_s': model.W_s.state_dict()
            }

    def save(self, model, dir, accelerator):
        model.student.save_pretrained(
            dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            W = {'t' : model.W_t, 's' : model.W_s}
            torch.save(W, dir+'/linear_mapping.pt')
            del W