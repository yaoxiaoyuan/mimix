# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 09:49:33 2021

@author: Xiaoyuan Yao
"""
import os
import numpy as np
import sys
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from predictor import EncDecGenerator
from utils import parse_test_args,load_config,real_path

def debug(enc_dec_gen, src_list, trg_list, topk=20):
    """
    """
    x, y = enc_dec_gen.encode_inputs(src_list, trg_list, add_bos=True)

    with torch.no_grad():
        outputs = enc_dec_gen.model([x,y], return_states=True)
            
        logits = outputs[0]
        logits = logits.view(-1, enc_dec_gen.trg_vocab_size)
        
        probs = F.softmax(logits, -1)
        probs = probs.view(y.size(0), y.size(1), -1)
        
        top_probs, top_words = probs.topk(topk, -1)
    
        y_len = torch.sum(y.ne(enc_dec_gen.model.PAD), -1)
        
        entropy = Categorical(probs=probs).entropy()
        
        log_probs = F.log_softmax(logits, -1).view(y.size(0), y.size(1), -1)
        history = torch.gather(log_probs[:,:-1,:], -1, y[:,1:].unsqueeze(-1))
    
    top_words = top_words.cpu().numpy()
    top_probs = np.round(top_probs.cpu().numpy(), 3)
    y_len = y_len.cpu().numpy()
    history = np.round(history.squeeze(-1).cpu().numpy(), 3)
    entropy = np.round(entropy.cpu().numpy(), 3)

    res = []
    for i,(src,trg) in enumerate(zip(src_list, trg_list)):
        word_ids = top_words[i][y_len[i]-1]
        words = [enc_dec_gen.trg_id2word.get(w, "_unk_") for w in word_ids]
        word_prob = top_probs[i][y_len[i]-1]
        pairs = tuple(zip(words, word_prob))
        
        history_probs = list(history[i][:y_len[i]])
        sum_log_probs = sum(history[i][:y_len[i]])
        
        res.append([pairs, history_probs, sum_log_probs, list(entropy[i])])
        
    return res
        

def enc_dec_debug(config):
    """
    """
    enc_dec_gen = EncDecGenerator(config)
    print("INPUT TEXT:")
    def _process_batch(src_list, trg_list):
        """
        """
        res = debug(enc_dec_gen, src_list, trg_list, topk=20)
        
        for i,(src,trg) in enumerate(zip(src_list, trg_list)):
            print("src: %s" % src)
            print("trg: %s" % trg)
            pairs, history, sum_log_probs, entropy = res[i]
            print("scores: ", history)
            print("sum log probs: ", sum_log_probs)
            print("entropy: ", entropy)
            for word,prob in pairs:
                print(word, prob)
                    
    src_list = []
    trg_list = []
        
    for line in sys.stdin:
        line = line.strip()
        
        if len(line) == 0:
            continue
        
        try:
            src,trg = line.split("\t")[:2]
        except:
            src,trg = line, ""
            
        src_list.append(src)
        trg_list.append(trg)
            
        _process_batch(src_list, trg_list)
        src_list = []
        trg_list = []


def run_debug():
    """
    """
    usage = "usage: debug.py --model_conf <file>"
    options = parse_test_args(usage)
    conf_file = options.model_config
    config = load_config(real_path(conf_file), add_symbol=True)
        
    if config["task"] == "enc_dec":
        enc_dec_debug(config)
        
if __name__ == "__main__":
    run_debug()
