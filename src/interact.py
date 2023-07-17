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
from utils import pretty_print, parse_test_args, real_path, load_model_config

def enc_dec_demo(config):
    """
    """    
    enc_dec_gen = EncDecGenerator(config)
    #src_list = ["综合美国《华盛顿时报》、彭博社等多家外媒报道，美国企业家埃隆·马斯克当地时间12日参加“推特空间”（Twitter Spaces）活动时，谈及人工智能（AI）、中国等话题。他表示，中国有意愿且已准备好与国际社会合作、一同制定AI规则。期间他称自己“有点亲华”，并称“中国人民真的很棒”。",
    #            "参考消息网7月12日报道据俄罗斯《莫斯科共青团员报》网站7月12日报道，乌克兰总统泽连斯基在北约峰会期间的一场工作会谈中，拒绝与北约秘书长斯托尔滕贝格握手，引发外界的猜想和疑虑。"]
    #prefix_list = ["美国",
    #               "乌克兰"]
    src_list = ["what is the next number of 1 2 3 4 5",
                "what is the english for 1 2 3 4 5"]
    prefix_list = ["the next number",
                   "it is"]
    search_res = enc_dec_gen.predict(src_list, prefix_list=prefix_list)
    search_res = [{"src":x, "predict":y} for x,y in search_res]
    pretty_print(search_res)      
    print("INPUT TEXT:")
    for line in sys.stdin:
        line = line.strip()

        if len(line) == 0:
            continue
    
        src_list = [line.strip().split("\t")[0]]
            
        prefix_list = None
        if "\t" in line:
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
    prefix_list = ["_shi_ _xinyun_ _7jue_ _title_ 梅花",
                   "_shi_ _xinyun_ _5jue_ _title_ 寒江雪"]
    #prefix_list = ["李", "张 晓"]
    search_res = lm_gen.predict(prefix_list=prefix_list) 
    search_res = [{"src":x, "predict":y} for x,y in search_res]
    pretty_print(search_res)        
    print("INPUT TEXT:")
    for line in sys.stdin:
        line = line.strip()
        prefix_list = None
        if len(line) > 0:
            prefix_list = [line] 
        
        start = time.time()
        search_res = lm_gen.predict(prefix_list=prefix_list) 
        
        search_res = [{"src":x, "predict":y} for x,y in search_res]
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
    config = load_model_config(real_path(conf_file))
    
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
