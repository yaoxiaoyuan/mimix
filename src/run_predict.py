# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:39:21 2019

@author: Xiaoyuan Yao
"""
import os
import sys
import time
from predictor import EncDecGenerator, TextClassifier, LMGenerator
from utils import parse_test_args, real_path, load_config

def predict_enc_dec(config):
    """
    """
    enc_dec_gen = EncDecGenerator(config)
    test_batch_size = 1
    instream = sys.stdin
    outstream = sys.stdout
    if config.get("test_in", "stdin") != "stdin":
        instream = open(real_path(config["test_in"]), 
                        "r", encoding="utf-8", errors="ignore")
        test_batch_size = config.get("test_batch_size", 1)
        
    if config.get("test_out", "stdout") != "stdout":
        outstream = open(real_path(config["test_out"]), 
                         "w", encoding="utf-8")
        
    
    def _process_batch(src_list, trg_list=None):
        """
        """
        search_res = enc_dec_gen.predict(src_list, prefix_list=trg_list)
        
        for raw_src, (src, res) in zip(src_list, search_res):
            for trg, score in res:
                outstream.write(raw_src + "\t" + trg + "\n")
        outstream.flush()
                    
    start = time.time()
    src_list = []
    trg_list = None
    if "prefix" in enc_dec_gen.strategy:
        trg_list = []
        
    for line in instream:
        line = line.strip()
        
        if len(line) == 0:
            continue
        
        src = line.split("\t")[0]
        if "prefix" in enc_dec_gen.strategy:
            trg = line.split("\t")[1]
            
        change_prefix = False
        if "prefix" in enc_dec_gen.strategy:
            if len(trg_list) > 0 and trg != trg_list[-1]:
                change_prefix = True
            
        if len(src_list) < test_batch_size:
            if change_prefix == True:
                src_list.append(src)
                trg_list.append(trg)
            else:
                src_list.append(src)
            
        if len(src_list) >= test_batch_size or change_prefix == True:
            _process_batch(src_list, trg_list)
            src_list = []
            if "prefix" in enc_dec_gen.strategy:
                trg_list = []
                
        if change_prefix == True:
            src_list.append(src)
            trg_list.append(trg)

    if len(src_list) > 0:
        _process_batch(src_list)

    end = time.time()
    cost = end - start
    print("#cost time: %s s" % cost)

    outstream.close()


def predict_lm(config):
    """
    """
    lm_gen = LMGenerator(config)

    outstream = sys.stdout
    if config.get("test_out", "stdout") != "stdout":
        outstream = open(real_path(config["test_out"]), 
                          "w", encoding="utf-8")
    test_batch_size = config.get("test_batch_size", 1)
    
    if "prefix" not in lm_gen.strategy:
        while True:
            search_res = lm_gen.sample(batch_size=test_batch_size)
            for y,score in search_res[0]:
                outstream.write(y + "\n")
    else:
        instream = open(real_path(config["test_in"]), 
                        "r", encoding="utf-8", errors="ignore")
        test_batch_size = 1
        prefix_list = []
        for line in instream:
            line = line.strip()
            prefix_list.append(line)
            if len(prefix_list) >= test_batch_size:
                search_res = lm_gen.sample(prefix_list)
                for li in search_res:
                    for y,score in li:
                        outstream.write(y + "\n")
                prefix_list = []

    if len(prefix_list) > 0:
        search_res = lm_gen.sample(prefix_list)
        for li in search_res:
            for y,score in li:
                outstream.write(y + "\n")
                
    outstream.close()
    

def predict_classify(config):
    """
    """
    cls = TextClassifier(config)

    test_batch_size = 1
    instream = sys.stdin
    outstream = sys.stdout
    if config.get("test_in", "stdin") != "stdin":
        instream = open(real_path(config["test_in"]), 
                        "r", encoding="utf-8", errors="ignore")
        if config.get("test_out", "stdout") != "stdout":
            outstream = open(real_path(config["test_out"]), 
                             "w", encoding="utf-8")
        test_batch_size = config.get("test_batch_size", 1)

    def _process_batch(src_list):
        """
        """
        texts = [src.split("\t")[0] for src in src_list]
        res = cls.predict(texts)
        
        for line, (src, labels) in zip(src_list, res):
            outstream.write(line + "\t" + labels[0][0] + "\n")
    
    start = time.time()
    src_list = []
    for line in instream:
        line = line.strip()
        
        if len(line) == 0:
            continue
        
        src_list.append(line)
        
        if len(src_list) % test_batch_size == 0:
            _process_batch(src_list)
            src_list = []
        
    if len(src_list) > 0:
        _process_batch(src_list)
    
    end = time.time()
    cost = end - start
    print("#cost time: %s s" % cost)

    outstream.close()


def run_predict():
    """
    """
    usage = "usage: run_predict.py --model_conf <file>"
    options = parse_test_args(usage)
    conf_file = options.model_config
    config = load_config(real_path(conf_file), add_symbol=True) 
        
    if config["task"] == "enc_dec":
        predict_enc_dec(config)
    elif config["task"] == "lm":
        predict_lm(config)
    elif config["task"] == "classify":
        predict_classify(config)


if __name__ == "__main__":
    run_predict()
