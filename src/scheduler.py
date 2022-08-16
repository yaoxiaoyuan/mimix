# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 11:53:07 2021

@author: Xiaoyuan Yao
"""
import math

class ConstantScheduler():
    """
    """
    def __init__(self, train_config, optimizer):
        """
        """
        self.lr = train_config["lr"]
        self.optimizer = optimizer
    

    def step(self):
        """
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr


def build_lr_scheduler(train_config, optimizer):
    """
    """
    if "scheduler" not in train_config:
        return ConstantScheduler(train_config, optimizer)
    else:
        raise ValueError("scheduler not correct!")