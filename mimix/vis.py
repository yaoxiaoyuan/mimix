# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:37:34 2020

@author: Xiaoyuan Yao
"""
import os
import sys
import torch
import numpy as np
from argparse import ArgumentParser
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pylab import mpl
from mimix.predictor import EncDecGenerator
from mimix.utils import load_model_config,real_path

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def draw_heatmap(x, y, scores, fig_path):
    """
    """    
    #mpl.rcconfig['font.sans-serif'] = ['STZhongsong']    
    #mpl.rcconfig['axes.unicode_minus'] = False 
    
    scores = np.round(scores, 2)

    fig, ax = plt.subplots()
    im = ax.imshow(scores, cmap='hot_r')
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(y)))
    ax.set_yticks(np.arange(len(x)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(y)
    ax.set_yticklabels(x)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(x)):
        for j in range(len(y)):
            text = ax.text(j, i, scores[i, j],
                           ha="center", va="center", color="w")
    
    plt.savefig(fig_path, dpi=300, bbox_inches = 'tight')
    #plt.show()
    plt.close()


def analysis_search(enc_dec_gen, src, trg):
    """
    """
    src_list, trg_list = [src], [trg]
    batch_size = 1
    
    x,y = enc_dec_gen.encode_inputs(src_list, trg_list, add_bos=True, add_eos=True)
    enc_dec_gen.model.eval()
    with torch.no_grad():
        outputs = enc_dec_gen.model([x,y[:, :-1]])
        
        dec_enc_attn_weights_list = outputs["enc_attn_weights_list"]
        
        attn_score_list = []
        
        for i in range(enc_dec_gen.model.n_dec_layers):
            
            attn_weights = dec_enc_attn_weights_list[i]
            
            attn_weights = attn_weights.mean(1).cpu().numpy()
            
            attn_score_list.append(attn_weights)
    
    res_list = []
    
    x,y = x.cpu().numpy(), y.cpu().numpy()
    for i in range(batch_size):
        src = [enc_dec_gen.src_id2word.get(w, "_unk_") for w in x[i]]
        trg = [enc_dec_gen.trg_id2word.get(w, "_unk_") for w in y[i][1:]]
        
        tmp = []
        for j in range(enc_dec_gen.model.n_dec_layers):
            tmp.append(attn_score_list[j][i][:len(trg), :len(src)].T)

        res = [src, trg, tmp]
        
        res_list.append(res)

    return res_list


def visualize_enc_dec(config):
    """
    """ 
    enc_dec_gen = EncDecGenerator(config)
    
    print("INPUT TEXT:")
    for line in sys.stdin:
        line = line.strip()
        
        if len(line) == 0:
            continue
        
        src,trg = line.split("\t")[:2]
        
        res = analysis_search(enc_dec_gen, src, trg)
        src, trg, attn_score_list = res[0]

        for i in range(enc_dec_gen.model.n_dec_layers):
            
            draw_heatmap(src, trg, attn_score_list[i], "logger/dec_enc_%d.png" % i)


def run_visualize():
    """
    """
    parser = ArgumentParser()

    parser.add_argument("--model_conf", type=str)
    args = parser.parse_args(sys.argv[1:])
    
    conf_file = args.model_conf
    config = load_model_config(real_path(conf_file))
        
    if config["task"] == "enc_dec":
        visualize_enc_dec(config)
        
        
if __name__ == "__main__":
    run_visualize()
