# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:38:54 2021

@author: Xiaoyuan Yao
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from predictor import EncDecGenerator,LMGenerator
from utils import parse_test_args,load_model_config,real_path

def enc_dec_score(config):
    """
    """
    enc_dec_gen = EncDecGenerator(config)
                        
    start = time.time()

    print("INPUT TEXT:")
    for line in sys.stdin:
        line = line.strip()
        
        if len(line) == 0:
            continue
        
        arr = line.strip().split("\t")
        src = arr[0]
        trg_list = arr[1:]
            
        pairs_list = [[src, trg_list]]
            
        res = enc_dec_gen.scoring(pairs_list)
        
        for src, trg_list in res:
            for trg,score in trg_list:
                print(src, trg, score)

    end = time.time()
    cost = end - start
    print("#cost time: %s s" % cost)


def lm_score(config):
    """
    """
    lm_gen = LMGenerator(config)
                        
    start = time.time()

    print("INPUT TEXT:")
    for line in sys.stdin:
        line = line.strip()
        
        if len(line) == 0:
            continue
        
        trg_list = line.strip().split("\t")
            
        res = lm_gen.scoring(trg_list)
        
        for trg, score in res:
            print(trg, score)

    end = time.time()
    cost = end - start
    print("#cost time: %s s" % cost)


def run_scoring():
    """
    """
    usage = "usage: scoring.py --model_conf <file>"
    options = parse_test_args(usage)
    conf_file = options.model_config
    config = load_model_config(real_path(conf_file))
        
    if config["task"] == "enc_dec":
        enc_dec_score(config)
    elif config["task"] == "lm":
        lm_score(config)
        
if __name__ == "__main__":
    run_scoring()
