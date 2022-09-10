# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:37:34 2020

@author: Xiaoyuan Yao
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pylab import mpl
from predictor import EncDecGenerator
from utils import parse_test_args,load_config,real_path

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


def analysis_search(enc_dec_gen, src, trg, topk=3):
    """
    """
    src_list, trg_list = [src], [trg]
    batch_size = 1
    
    x,y = enc_dec_gen.encode_inputs(src_list, trg_list)
    
    with torch.no_grad():
        outputs = enc_dec_gen.model([x,y[:, :-1]], return_states=True)

        dec_enc_dec_gen_attn_scores_list = outputs[3]
        
        dec_enc_attn_scores_list = outputs[4]
        enc_enc_dec_gen_attn_scores_list = outputs[8]

        attn_score_list = [[], [], []]
        
        for i in range(enc_dec_gen.model.n_enc_layers):
            
            attn_scores = enc_enc_dec_gen_attn_scores_list[i]
            
            #attn_scores = torch.mean(attn_scores, 1)
            #print(attn_scores.shape)
            attn_scores = attn_scores.cpu().numpy()
        
            attn_score_list[0].append(attn_scores)
        
        for i in range(enc_dec_gen.model.n_dec_layers):
            
            attn_scores = dec_enc_dec_gen_attn_scores_list[i]
            
            #attn_scores = torch.mean(attn_scores, 1)
            #print(attn_scores.shape)
            attn_scores = attn_scores.cpu().numpy()
            
            attn_score_list[1].append(attn_scores)

            attn_scores = dec_enc_attn_scores_list[i]
            
            #attn_scores = torch.mean(attn_scores, 1)
            #print(attn_scores.shape)
            attn_scores = attn_scores.cpu().numpy()

            attn_score_list[2].append(attn_scores)
    
    res_list = []
    
    x,y = x.cpu().numpy(), y.cpu().numpy()
    for i in range(batch_size):
        src = [enc_dec_gen.src_id2word.get(w, "_unk_") for w in x[i]]
        trg = [enc_dec_gen.trg_id2word.get(w, "_unk_") for w in y[i][1:]]
        
        attn_score = [[], [], []]
        for j in range(enc_dec_gen.model.n_enc_layers):
            attn_score[0].append(attn_score_list[0][j][i,:,:,:])
        
        for j in range(enc_dec_gen.model.n_dec_layers):
            attn_score[1].append(attn_score_list[1][j][i,:,:,:])
            attn_score[2].append(attn_score_list[2][j][i,:,:,:])
        
        res = [src, trg, attn_score]
        
        res_list.append(res)

    return res_list


def visualize_enc_dec(config):
    """
    """ 
    enc_dec_gen = EncDecGenerator(config)
    
    src_list = []
    trg_list = []
    print("INPUT TEXT:")
    for line in sys.stdin:
        line = line.strip()
        
        if len(line) == 0:
            continue
        
        src,trg = line.split("\t")[:2]
        
        res = analysis_search(enc_dec_gen, src, trg)
        src, trg, attn_score = res[0]
        print("src: %s, trg: %s" % (" ".join(src), " ".join(trg)))

        #for i in range(enc_dec_gen.model.n_enc_layers):
        #    for j in range(enc_dec_gen.model.n_heads):
        #        draw_heatmap(src, src, attn_score[0][i][j,:,:].T, 
        #                     "../logger/enc_%d_%d"%(i, j))
        
        #for i in range(enc_dec_gen.model.n_dec_layers):
        #    for j in range(enc_dec_gen.model.n_heads):
        #        
        #        #draw_heatmap(trg, trg, attn_score[1][i][j,:,:].T, 
        #        #             "../logger/dec_%d_%d"%(i, j))
        #        draw_heatmap(src, trg, attn_score[2][i][j,:,:].T, 
        #                     "../logger/dec_enc_%d_%d"%(i, j))

        dec_enc_attn_scores = []
        for i in range(enc_dec_gen.model.n_dec_layers):
            for j in range(enc_dec_gen.model.n_heads):
                dec_enc_attn_scores.append(attn_score[2][i][j,:,:].T)
        dec_enc_attn_score = np.mean(dec_enc_attn_scores, 0)
        print(dec_enc_attn_score.shape)
        draw_heatmap(src, trg, dec_enc_attn_score, "../logger/dec_enc")


def run_visualize():
    """
    """
    usage = "usage: visualize_attn.py --model_conf <file>"
    options = parse_test_args(usage)
    conf_file = options.model_config
    config = load_config(real_path(conf_file), add_symbol=True)
        
    if config["task"] == "enc_dec":
        visualize_enc_dec(config)
        
        
if __name__ == "__main__":
    run_visualize()
