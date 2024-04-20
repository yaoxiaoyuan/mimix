# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 21:41:58 2024

@author: Xiaoyuan Yao
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from argparse import ArgumentParser
import numpy as np
import torch
from mimix.models import build_model, load_model_weights
from mimix.optimizer import build_optimizer
from mimix.scheduler import build_scheduler
from mimix.loss import seq_cross_entropy
from mimix.utils import real_path, load_config, load_model_config, SimpleDataset
import deepspeed
from mimix.ds import train
import re
import random
import json

import mimix.tokenization as tokenization

deepspeed.init_distributed()
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
rank = int(os.environ['RANK'])

def loop_all_texts():
    pass

import math
class Scheduler():
    """
    """
    def __init__(self, optimizer, train_config):
        """
        """
        self.optimizer = optimizer
        self.steps = 0


    def step(self):
        """
        """
        self.steps += 1

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr


def main(model_config, train_config, ds_config):
    """
    """
    model = build_model(model_config)
    if train_config.get("reload_model", None) is not None:
        model = load_model_weights(model, real_path(train_config["reload_model"]))
    PAD = model.PAD
    eps = train_config.get("eps", 0)
    model.loss_fn = lambda x,y:seq_cross_entropy(x["logits"], y["y_target"], eps, PAD)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
 
    model_engine, optimizer, _, __ = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        config=ds_config,
    )
    lr_scheduler = Scheduler(optimizer, train_config)
    train(model_engine, optimizer, train_config, loop_all_texts, lr_scheduler)

def run_train():
    """
    """
    model_conf = "conf/xxx_lm_conf" 
    train_conf = "conf/xxx_train_conf"
    model_config = load_model_config(real_path(model_conf))
    train_config = load_config(real_path(train_conf))
    ds_config = json.loads(open("conf/ds_config_zero2.json", "rb").read()) 

    main(model_config, train_config, ds_config)


if __name__ == "__main__":
    run_train()
    
