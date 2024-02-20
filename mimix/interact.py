# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 12:20:14 2018

@author: Xiaoyuan Yao
"""
from argparse import ArgumentParser
import sys
import time
from PIL import Image
from mimix.predictor import EncDecGenerator,LMGenerator,TextEncoder
from mimix.predictor import ImageEncoder, ClipMatcher
from mimix.utils import real_path, load_model_config, convert_tokens_to_midi


def pretty_print(res):
    """
    Assume res = [{k1:v1,k2:v2,...,kn:[[s1, score1], [s2, score2]]}, {}, ... ]
    """
    for dic in res:
        info = [[k,dic[k]] for k in dic if not isinstance(dic[k], list)]
        info = " ".join("%s:%s" % (k,v) for k,v in info)
        if len(info) > 0:
            print(info)
        print("--------------------")
        for k in dic:
            if isinstance(dic[k], list):
                for a in dic[k]:
                    info = " ".join([str(x) for x in a])
                    print(info)


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
     
    print("INPUT TEXT:")
    for line in sys.stdin:
        line = line.strip()
        prefix_list = None
        if len(line) > 0:
            prefix_list = [line] 
        
        start = time.time()
        search_res = lm_gen.predict(prefix_list=prefix_list) 

        if config.get("convert2midi", False) == True:
            tokens = search_res[0][1][0][0].split()
            convert_tokens_to_midi(tokens, "test.mid")
        
        search_res = [{"src":x, "predict":y} for x,y in search_res]
        pretty_print(search_res)
        
        end = time.time()
        cost = end - start
        print("-----cost time: %s s-----" % cost)
        

def mlm_demo(config):
    """
    """
    lm_gen = TextEncoder(config)
    
    print("INPUT TEXT:")
    
    for line in sys.stdin:
        line = line.strip()

        if len(line) == 0:
            continue
        
        start = time.time()
        
        res = lm_gen.predict_mlm([line])
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
    text_matcher = TextEncoder(config)
    
    print("INPUT TEXT:")
    
    for line in sys.stdin:
        line = line.strip()

        if len(line) == 0:
            continue
        
        start = time.time()
        
        texts = line.split("\t")
        res = text_matcher.predict_sim(texts)
        for i,text_1 in enumerate(texts):
            for j, text_2 in enumerate(texts):
                if j <= i:
                    continue
                print(text_1, text_2, res[i][j])
        end = time.time()
        cost = end - start
        print("-----cost time: %s s-----" % cost)


def classification_demo(config):
    """
    """
    classifier = TextEncoder(config)
    
    print("INPUT TEXT:")
    
    for line in sys.stdin:        

        line = line.strip()

        if len(line) == 0:
            continue
        
        src_list = [line]
        
        start = time.time()
        
        res = classifier.predict_cls(src_list)
        res = [{"src":src, "labels":li} for src,li in res]
        pretty_print(res)

        end = time.time()
        cost = end - start
        print("-----cost time: %s s-----" % cost)


def sequene_labeling_demo(config):
    """
    """
    labeler = TextEncoder(config)
    
    print("INPUT TEXT:")
    
    for line in sys.stdin: 
        
        line = line.strip()

        if len(line) == 0:
            continue
        
        src_list = [line]
        
        start = time.time()
        
        res = labeler.predict_seq(src_list)
        res = [{"src":src, "labels":li} for src,li in res]
        pretty_print(res)

        end = time.time()
        cost = end - start
        print("-----cost time: %s s-----" % cost)


def enc_dec_debug(config):
    """
    """
    enc_dec_gen = EncDecGenerator(config)
    
    print("INPUT TEXT:")
    src_list = []
    trg_list = []
    for line in sys.stdin:
        line = line.strip()
        
        src,trg = line.split("\t")[:2]
            
        src_list.append(src)
        trg_list.append(trg)
            
        res = enc_dec_gen.get_topk_pred(src_list, trg_list, topk=10)[0]
        words, topk_pairs, history, sum_log_probs, entropy = res
        print("src: %s" % src)
        print("trg: %s" % trg)
        print("sum_log_probs: %.2f" % sum_log_probs)
        print("avg_log_probs: %.2f" % (sum_log_probs / len(words)))
        for i,word in enumerate(words):
            info = word
            info = info + " prob: %.2f entropy: %.2f" % (history[i], entropy[i])
            info = info + " topk:" + " ".join(["%s:%.2f" % (w,s) for w,s in topk_pairs[i]])
            print(info)


def lm_debug(config):
    """
    """
    lm_gen = LMGenerator(config)
    
    print("INPUT TEXT:")
    for line in sys.stdin:
        line = line.strip()
        
        trg = line.strip()

        res = lm_gen.get_topk_pred([trg], topk=10)[0]
        words, topk_pairs, history, sum_log_probs, entropy = res
        print("trg: %s" % trg)
        print("sum_log_probs: %.2f" % sum_log_probs)
        print("avg_log_probs: %.2f" % (sum_log_probs / len(words)))
        for i,word in enumerate(words):
            info = word
            info = info + " prob: %.2f entropy: %.2f" % (history[i], entropy[i])
            info = info + " topk:" + " ".join(["%s:%.2f" % (w,s) for w,s in topk_pairs[i]])
            print(info)


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


def image_classification_demo(config):
    """
    """
    classifier = ImageEncoder(config)

    print("INPUT IMAGE PATH:")

    for line in sys.stdin:

        line = line.strip()

        if len(line) == 0:
            continue

        image_path = line
        images = [Image.open(image_path)]

        start = time.time()

        res = classifier.predict_cls(images)
        images[0].close()
        res = [{"labels":li} for src,li in res]
        pretty_print(res)
        
        end = time.time()
        cost = end - start
        print("-----cost time: %s s-----" % cost)


def image2text_demo(config):
    """
    """
    if config["model"] == "transformer":
        enc_dec_gen = EncDecGenerator(config)
    
    print("INPUT IMAGE PATH:")

    for line in sys.stdin:

        line = line.strip()

        if len(line) == 0:
            continue

        image_path = line
        images = [Image.open(image_path)]

        start = time.time()

        search_res = enc_dec_gen.predict(images)
        images[0].close()
        search_res = [{"predict":y} for x,y in search_res]
        pretty_print(search_res)

        end = time.time()
        cost = end - start
        print("-----cost time: %s s-----" % cost)


def text_image_match_demo(config):
    """
    """
    clip = ClipMatcher(config)
    
    print("INPUT IMAGE PATH AND TEXT:")

    for line in sys.stdin:

        line = line.strip()

        if len(line) == 0:
            continue

        image_path = line.split("\t")[0]
        images = [Image.open(image_path)]
        texts = line.split("\t")[1:]

        start = time.time()

        res = clip.predict_sim(images,texts)
        images[0].close()
        for text,score,prob in zip(texts, res[0][0], res[1][0]):
            print(text, score, prob)

        end = time.time()
        cost = end - start
        print("-----cost time: %s s-----" % cost)


def run_interactive():
    """
    """
    parser = ArgumentParser()

    parser.add_argument("--model_conf", type=str)
    parser.add_argument("--mode", type=str, default="pred")
    
    args = parser.parse_args(sys.argv[1:])

    conf_file = args.model_conf
    config = load_model_config(real_path(conf_file))
    
    if args.mode == "pred":
        if config["task"] == "enc_dec":
            enc_dec_demo(config)
        elif config["task"] == "classification":
            classification_demo(config)
        elif config["task"] == "lm":
            lm_demo(config)
        elif config["task"] == "mlm":
            mlm_demo(config)
        elif config["task"] == "seqcls":
            sequene_labeling_demo(config)
        elif config["task"] == "match":
            match_text_demo(config)
        elif config["task"] == "image_classification":
            image_classification_demo(config)
        elif config["task"] == "image2text":
            image2text_demo(config)
        elif config["task"] == "image_text_match":
            text_image_match_demo(config)
    elif args.mode == "debug":
        if config["task"] == "enc_dec":
            enc_dec_debug(config)  
        if config["task"] == "lm":
            lm_debug(config) 
    elif args.mode == "scoring":
        if config["task"] == "enc_dec":
            enc_dec_score(config) 
        elif config["task"] == "lm":
            lm_score(config)
            
if __name__ == "__main__":
    run_interactive()
