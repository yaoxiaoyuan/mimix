# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:16:15 2021

@author: Xiaoyuan Yao
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import TransformerLM

def load_gpt2_weights(gpt2, weight_path):
    """
    """
    gpt2.eval()
    state_dict = torch.load(weight_path,
                            map_location=lambda storage, loc: storage)
    
    model_state_dict = {}
    model_state_dict["decoder.trg_embedding.W"] = state_dict["transformer.wte.weight"]
    model_state_dict["decoder.pos_embedding.W"] = state_dict["transformer.wpe.weight"]
    
    for i in range(gpt2.n_dec_layers):
        n = state_dict["transformer.h.%d.attn.c_attn.weight" % i].shape[1]
        model_state_dict["decoder.layers.%d.self_attention.W_q" % i] = state_dict["transformer.h.%d.attn.c_attn.weight" % i][:,:n//3].T
        model_state_dict["decoder.layers.%d.self_attention.b_q" % i] = state_dict["transformer.h.%d.attn.c_attn.bias" % i][:n//3]
        model_state_dict["decoder.layers.%d.self_attention.W_k" % i] = state_dict["transformer.h.%d.attn.c_attn.weight" % i][:,n//3:2*n//3].T
        model_state_dict["decoder.layers.%d.self_attention.b_k" % i] = state_dict["transformer.h.%d.attn.c_attn.bias" % i][n//3:2*n//3]
        model_state_dict["decoder.layers.%d.self_attention.W_v" % i] = state_dict["transformer.h.%d.attn.c_attn.weight" % i][:,-n//3:].T
        model_state_dict["decoder.layers.%d.self_attention.b_v" % i] = state_dict["transformer.h.%d.attn.c_attn.bias" % i][-n//3:]
        model_state_dict["decoder.layers.%d.self_attention.W_o" % i] = state_dict["transformer.h.%d.attn.c_proj.weight" % i].T
        model_state_dict["decoder.layers.%d.self_attention.b_o" % i] = state_dict["transformer.h.%d.attn.c_proj.bias" % i]
        model_state_dict["decoder.layers.%d.norm_1.alpha" % i] = state_dict["transformer.h.%d.ln_1.weight" % i]
        model_state_dict["decoder.layers.%d.norm_1.bias" % i] = state_dict["transformer.h.%d.ln_1.bias" % i]
        model_state_dict["decoder.layers.%d.ffn.W1" % i] = state_dict["transformer.h.%d.mlp.c_fc.weight" % i].T
        model_state_dict["decoder.layers.%d.ffn.b1" % i] = state_dict["transformer.h.%d.mlp.c_fc.bias" % i]
        model_state_dict["decoder.layers.%d.ffn.W2" % i] = state_dict["transformer.h.%d.mlp.c_proj.weight" % i].T
        model_state_dict["decoder.layers.%d.ffn.b2" % i] = state_dict["transformer.h.%d.mlp.c_proj.bias" % i]
        model_state_dict["decoder.layers.%d.norm_3.alpha" % i] = state_dict["transformer.h.%d.ln_2.weight" % i]
        model_state_dict["decoder.layers.%d.norm_3.bias" % i] = state_dict["transformer.h.%d.ln_2.bias" % i]
    
    if gpt2.share_emb_out_proj == False:
        model_state_dict["decoder.W"] = state_dict["lm_head.weight"]
    if "lm_head.bias" in state_dict:
        model_state_dict["decoder.b"] = state_dict["lm_head.bias"]
    
    model_state_dict["decoder.norm.alpha"] = state_dict["transformer.ln_f.weight"]
    model_state_dict["decoder.norm.bias"] = state_dict["transformer.ln_f.bias"]

    
    gpt2.load_state_dict(model_state_dict, True)
    
    return gpt2

if __name__ == "__main__":
    
    from decoding import lm_sample
    from tokenization import build_tokenizer
    gpt2 = TransformerLM({'_pad_':0, '_bos_':3297, '_eos_':0, '_unk_':100, "_cls_":101, "_sep_":102, "_mask_":103}, 
                         21128,
                         1024, 
                         12, 
                         768, 
                         3072,
                         64, 
                         64, 
                         12, 
                         dropout=0., 
                         share_emb_out_proj=True, 
                         activation="gelu_new",
                         norm_before_pred=True,
                         pos_need_train=True,
                         use_proj_bias=False)
    
    gpt2 = load_gpt2_weights(gpt2, "../../pretrain/gpt2_lyric/pytorch_model.bin")
    
    hypothesis,scores = lm_sample(gpt2, 
                                  100, 
                                  False,
                                  torch.device("cpu"),
                                  normalize="none",
                                  gamma=1,
                                  temp=1) 
    
    trg_tokenizer = build_tokenizer(tokenizer="bert", 
                                    vocab_file="../../pretrain/gpt2_lyric/vocab.txt",
                                    pre_tokenized=False, 
                                    pre_vectorized=False)
    for hyp,score in zip(hypothesis, scores):
        trg = trg_tokenizer.detokenize_ids(hyp.numpy())
    
        print(trg)
    