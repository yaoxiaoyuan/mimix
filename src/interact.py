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
    
    print("INPUT TEXT:")
    #src_list = ["人民网北京7月10日电（欧阳易佳）据中国气象局消息，在刚刚过去的周末，京津冀和河南、山东、浙江、福建、江西等地出现高温天气，多地超过40℃。今天，高温天气将继续发力，中央气象台继续发布高温橙色预警，西北地区东部、华北大部、黄淮、江汉、江南、华南大部以及四川盆地等地有35℃以上的高温天气，其中，华北东部、黄淮中北部、江南中北部和东部及四川盆地中南部等地部分地区最高气温37至39℃，北京东南部、河北南部、河南北部、浙江东部、福建东部等局地可达40℃以上。11日，北方地区高温范围和强度将有所减小，12日受降水影响，此轮高温过程基本结束。",
    #            "根据北京市住建委官网数据统计，自今年4月起，北京二手房市场成交量已连续三个月下跌，另据机构数据，截至6月底，北京二手房挂牌量已高达近19万套。三个月以来市场的“活跃度不高”，让买卖双方之间的“堰塞湖”逐步形成：一方面，部分千万级别的房源开始被中介通知“降价200才有信心卖掉”；另一方面，卖房的不顺畅，直接影响到置换的第二步，部分刚改新房项目已经开始收到“因卖不掉二手房”为理由退掉的预定。"]
    #search_res = enc_dec_gen.predict(src_list, prefix_list=None)
            
    #search_res = [{"src":x, "predict":y} for x,y in search_res]
    #pretty_print(search_res)
        
    for line in sys.stdin:
        line = line.strip()

        if len(line) == 0:
            continue
    
        src_list = [line.strip().split("\t")[0]]
            
        prefix_list = None
        if "\t" in line:
            prefix_list = [line.split("\t")[1]]
        print(src_list, prefix_list)
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
