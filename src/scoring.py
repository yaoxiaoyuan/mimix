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
from utils import parse_args,load_config,real_path

def scoring_pairs(enc_dec_gen, src_list, trg_list):
    """
    """
    batch_size = len(src_list)
    x, y = enc_dec_gen.encode_inputs(src_list, 
                                     trg_list, 
                                     add_bos=True, 
                                     add_eos=True)
    
    y_len = torch.sum(y.ne(enc_dec_gen.model.PAD), -1)
        
    with torch.no_grad():
        y_target = y[:, 1:]
        y = y[:, :-1]
        outputs = enc_dec_gen.model([x,y], return_states=True)
            
        logits = outputs[0]
        logits = logits.view(-1, enc_dec_gen.trg_vocab_size)
            
        log_probs = -F.nll_loss(F.log_softmax(logits, -1), 
                                y_target.contiguous().view(-1), 
                                ignore_index=0, 
                                reduction='none')
        log_probs = torch.sum(log_probs.view(batch_size, -1), -1)    

    norm = 1
    if enc_dec_gen.normalize == "gnmt":
        norm = torch.pow(5. + y_len, enc_dec_gen.gamma) / np.power(6., enc_dec_gen.gamma)
    elif enc_dec_gen.normalize == "linear":
        norm = y_len

    log_probs = log_probs / norm
        
    log_probs = log_probs.cpu().numpy()

    return log_probs
        

def rank_src_trgs(enc_dec_gen, src_list, trg_list):
    """
    """
    batch_size = len(trg_list)
    x, y = enc_dec_gen.encode_inputs(src_list, 
                                     trg_list, 
                                     add_bos=True, 
                                     add_eos=True)
    
    y_len = torch.sum(y.ne(enc_dec_gen.model.PAD), -1)
        
    with torch.no_grad():
        y_target = y[:, 1:]
        y = y[:, :-1]
        
        enc_self_attn_mask = enc_dec_gen.model.get_attn_mask(x, x)
        enc_outputs = enc_dec_gen.model.encoder(x, 
                                                enc_self_attn_mask)
        enc_output = enc_outputs[0]
        
        n = y.size(0)//x.size(0)
        x = x.repeat([1,n]).view(y.size(0), -1)
        enc_output = enc_output.repeat([1, n, 1]).view(x.size(0), x.size(1), -1)
        
        dec_self_attn_mask = enc_dec_gen.model.get_subsequent_mask(y)
    

        dec_self_attn_mask = dec_self_attn_mask | enc_dec_gen.model.get_attn_mask(y, y)
        dec_enc_attn_mask = enc_dec_gen.model.get_attn_mask(y, x)
    
        trg_embedding = None
        if enc_dec_gen.model.share_src_trg_emb == True:
            trg_embedding = enc_dec_gen.model.encoder.src_embedding

        dec_outputs = enc_dec_gen.model.decoder(y, 
                                                enc_output, 
                                                dec_self_attn_mask, 
                                                dec_enc_attn_mask,
                                                trg_embedding=trg_embedding)
            
        logits = dec_outputs[0]
        logits = logits.view(-1, enc_dec_gen.trg_vocab_size)
            
        log_probs = -F.nll_loss(F.log_softmax(logits, -1), 
                                y_target.contiguous().view(-1), 
                                ignore_index=enc_dec_gen.model.PAD, 
                                reduction='none')

        log_probs = torch.sum(log_probs.view(batch_size, -1), -1)    
        
    norm = 1
    if enc_dec_gen.normalize == "gnmt":
        norm = torch.pow(5. + y_len, enc_dec_gen.gamma) / np.power(6., enc_dec_gen.gamma)
    elif enc_dec_gen.normalize == "linear":
        norm = y_len

    log_probs = log_probs / norm
        
    log_probs = log_probs.cpu().numpy()

    return log_probs


def enc_dec_score(config):
    """
    """
    enc_dec_gen = EncDecGenerator(config)
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
    for line in instream:
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

    outstream.close()


def run_scoring():
    """
    """
    usage = "usage: scoring.py --model_conf <file>"
    options = parse_test_args(usage)
    conf_file = options.model_config
    config = load_config(real_path(conf_file), add_symbol=True) 
        
    if config["task"] == "enc_dec":
        enc_dec_score(config)

        
if __name__ == "__main__":
    run_scoring()
