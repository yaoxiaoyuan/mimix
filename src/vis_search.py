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
from utils import parse_test_args,load_model_config,real_path

def get_topk_pred(enc_dec_gen, src_list, trg_list, topk=10):
    """
    """
    x, y = enc_dec_gen.encode_inputs(src_list, trg_list, add_bos=True, add_eos=True)
    enc_dec_gen.model.eval()
    with torch.no_grad():
        outputs = enc_dec_gen.model([x,y])
            
        logits = outputs[0]
        logits = logits.view(-1, enc_dec_gen.trg_vocab_size)
        
        probs = F.softmax(logits, -1)
        probs = probs.view(y.size(0), y.size(1), -1)
        
        top_probs, top_ids = probs.topk(topk, -1)
        
        y_len = torch.sum(y.ne(enc_dec_gen.model.PAD), -1)
        
        entropy = Categorical(probs=probs).entropy()
        
        history = torch.gather(probs[:,:-1,:], -1, y[:,1:].unsqueeze(-1))
        
    y = y.cpu().numpy()
    top_ids = top_ids.cpu().numpy()
    top_probs = np.round(top_probs.cpu().numpy(), 3)
    y_len = y_len.cpu().numpy()
    history = np.round(history.squeeze(-1).cpu().numpy(), 3)
    entropy = np.round(entropy.cpu().numpy(), 3)

    res = []
    for i,(src,trg) in enumerate(zip(src_list, trg_list)):
        words = [enc_dec_gen.trg_id2word.get(w, "_unk_") for w in  y[i][1:]]
        topk_pairs = []
        for j in range(y_len[i].item() - 1):

            top_words = [enc_dec_gen.trg_id2word.get(w, "_unk_") for w in top_ids[i][j]]
            pairs = tuple(zip(top_words, top_probs[i][j]))
            topk_pairs.append(pairs)
        
        history_probs = list(history[i][:y_len[i]])
        sum_log_probs = sum(np.log(history[i][:y_len[i]]))
        
        res.append([words, topk_pairs, history_probs, sum_log_probs, list(entropy[i])])
        
    return res
        

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
            
        res = get_topk_pred(enc_dec_gen, src_list, trg_list, topk=10)[0]
        words, topk_pairs, history, sum_log_probs, entropy = res
        print("src: %s" % src)
        print("trg: %s" % trg)
        print("sum_log_probs: %.2f" % sum_log_probs)
        for i,word in enumerate(words):
            info = word
            info = info + " prob: %.2f entropy: %.2f" % (history[i], entropy[i])
            info = info + " topk:" + " ".join(["%s:%.2f" % (w,s) for w,s in topk_pairs[i]])
            print(info)

def run_debug():
    """
    """
    usage = "usage: debug.py --model_conf <file>"
    options = parse_test_args(usage)
    conf_file = options.model_config
    config = load_model_config(real_path(conf_file))
        
    if config["task"] == "enc_dec":
        enc_dec_debug(config)
        
if __name__ == "__main__":
    run_debug()
