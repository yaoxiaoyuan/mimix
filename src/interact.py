# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 12:20:14 2018

@author: Xiaoyuan Yao
"""
import os
import sys
import time
from predictor import EncDecGenerator
from predictor import LMGenerator
from predictor import BiLMGenerator
from predictor import TextClassifier
from predictor import SequenceLabeler
from predictor import TextMatcher
from utils import pretty_print, parse_test_args, real_path, load_config

def enc_dec_demo(config):
    """
    """    
    enc_dec_gen = EncDecGenerator(config)
    
    print("INPUT TEXT:")
    
    for line in sys.stdin:
        line = line.strip()

        if len(line) == 0:
            continue
    
        src_list = [line.strip().split("\t")[0]]
            
        prefix_list = None
        if "prefix" in enc_dec_gen.strategy:
            if "\t" not in line:
                print("prefix can't be empty!")
                continue
            prefix_list = [line.split("\t")[1]]
        
        start = time.time()
        search_res = enc_dec_gen.predict(src_list, prefix_list=prefix_list)
            
        search_res = [{"src":x, "predict":y} for x,y in search_res]
        pretty_print(search_res)

        end = time.time()
        cost = end - start
        print("-----cost time: %s s-----" % cost)


def lm_demo(config):
    """
    """
    lm_gen = LMGenerator(config)
    
    print("INPUT TEXT:")
    
    for line in sys.stdin:
        line = line.strip()
        prefix_list = None
        if "prefix" in lm_gen.strategy:
            if len(line) > 0:
                prefix_list = [line] 
        
        start = time.time()
        search_res = lm_gen.sample(prefix_list=prefix_list) 
        
        search_res = [{"predict":y} for y in search_res]
        pretty_print(search_res)

        end = time.time()
        cost = end - start
        print("-----cost time: %s s-----" % cost)
        

def bi_lm_demo(config):
    """
    """
    lm_gen =  BiLMGenerator(config)
    
    print("INPUT TEXT:")
    
    for line in sys.stdin:
        line = line.strip()

        if len(line) == 0:
            continue
        
        start = time.time()
        
        res = lm_gen.predict([line])
        for src,pred in res:
            print("src:", src)
            for li in pred:
                print(" ".join(["%s:%s" % (w,s) for w,s in li]))
        end = time.time()
        cost = end - start
        print("-----cost time: %s s-----" % cost)


def match_text_demo(config):
    """
    """
    text_matcher = TextMatcher(config)
    
    print("INPUT TEXT:")
    
    for line in sys.stdin:
        line = line.strip()

        if len(line) == 0:
            continue
        
        start = time.time()
        
        texts = line.split("\t")
        res = text_matcher.predict(texts)
        for i,text_1 in enumerate(texts):
            for j, text_2 in enumerate(texts):
                if j <= i:
                    continue
                print(text_1, text_2, res[i][j])
        end = time.time()
        cost = end - start
        print("-----cost time: %s s-----" % cost)


def classify_demo(config):
    """
    """
    classifier = TextClassifier(config)
    
    print("INPUT TEXT:")
    
    for line in sys.stdin:        

        line = line.strip()

        if len(line) == 0:
            continue
        
        src_list = [line]
        
        start = time.time()
        
        res = classifier.predict(src_list)
        res = [{"src":src, "labels":li} for src,li in res]
        pretty_print(res)

        end = time.time()
        cost = end - start
        print("-----cost time: %s s-----" % cost)


def sequene_labeling_demo(config):
    """
    """
    labeler = SequenceLabeler(config)
    
    print("INPUT TEXT:")
    
    for line in sys.stdin: 
        
        line = line.strip()

        if len(line) == 0:
            continue
        
        src_list = [line]
        
        start = time.time()
        
        res = labeler.predict(src_list)
        res = [{"src":src, "labels":li} for src,li in res]
        pretty_print(res)

        end = time.time()
        cost = end - start
        print("-----cost time: %s s-----" % cost)


def run_interactive():
    """
    """
    usage = "usage: interact.py --model_conf <file>"
    options = parse_test_args(usage)
    conf_file = options.model_config
    config = load_config(real_path(conf_file), add_symbol=True)
    
    if config["task"] == "enc_dec":
        enc_dec_demo(config)
    elif config["task"] == "classify":
        classify_demo(config)
    elif config["task"] == "lm":
        lm_demo(config)
    elif config["task"] == "bi_lm":
        bi_lm_demo(config)
    elif config["task"] == "sequence_labeling":
        sequene_labeling_demo(config)
    elif config["task"] == "match":
        match_text_demo(config)
        
if __name__ == "__main__":
    run_interactive()
