# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:54:15 2021

@author: Xiaoyuan Yao
"""
from torch import optim

def build_optimizer(model, optimizer, lr):
    """
    """
    optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr, amsgrad=True)
    return optimizer