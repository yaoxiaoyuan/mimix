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
from predictor import EncDecGenerator
from utils import parse_test_args,load_model_config,real_path

def enc_dec_score(config):
    """
    """
    enc_dec_gen = EncDecGenerator(config)
    test_batch_size = 1
    
    def _process_batch(src_list, trg_list):
        """
        """
        res = scoring_pairs(enc_dec_gen, src_list, trg_list)
        
        for src,trg,score in zip(src_list, trg_list, res):
            outstream.write(src + "\t" + trg + "\t" + str(score) + "\n")
                    
    start = time.time()
    src_list = []
    trg_list = []
    
    print("INPUT TEXT:")
    for line in sys.stdin:
        line = line.strip()
        
        if len(line) == 0:
            continue
        
        src,trg = line.split("\t")[:2]
            
        src_list.append(src)
        trg_list.append(trg)
            
        if len(src_list) >= test_batch_size:
            _process_batch(src_list, trg_list)
            src_list = []
            trg_list = []

    if len(src_list) > 0:
        _process_batch(src_list, trg_list)

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

        
if __name__ == "__main__":
    run_scoring()
