# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:16:15 2021

@author: Xiaoyuan Yao
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Encoder,Dropout,LayerNorm

class Bert(nn.Module):
    """
    """
    def __init__(self,                 
                 symbol2id, 
                 vocab_size, 
                 max_len, 
                 n_heads, 
                 d_model, 
                 d_ff, 
                 d_qk,
                 d_v, 
                 n_enc_layers,
                 n_types,
                 dropout, 
                 embedding_size=None, 
                 share_layer_params=False, 
                 n_share_across_layers=0,
                 use_pre_norm=False, 
                 activation="gelu",
                 output_mlm=True,
                 output_cls=True,
                 share_emb_out_proj=True):
        """
        """
        super(Bert, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model 
        self.d_ff = d_ff 
        self.max_len = max_len
        self.n_enc_layers = n_enc_layers 
        self.n_types = n_types
        self.dropout = Dropout(dropout)
        self.use_pre_norm = use_pre_norm
        self.activation = activation
        self.vocab_size = vocab_size
        
        self.encoder = Encoder(vocab_size, 
                               max_len, 
                               n_heads,
                               d_model, 
                               d_ff, 
                               d_qk, 
                               d_v, 
                               n_enc_layers,
                               dropout, 
                               embedding_size,
                               share_layer_params, 
                               n_share_across_layers,
                               use_pre_norm, 
                               activation, 
                               scale_embedding=False,
                               norm_before_pred=False,
                               norm_after_embedding=True,
                               pos_need_train=True,
                               add_segment_embedding=True,
                               n_types=n_types)
        
        self.output_cls = output_cls
        if output_cls == True:
            self.W_pool = nn.Parameter(torch.Tensor(d_model, d_model))
            self.b_pool = nn.Parameter(torch.Tensor(d_model))
        
            self.W_cls = nn.Parameter(torch.Tensor(2, d_model))
            self.b_cls = nn.Parameter(torch.Tensor(2))
        
        self.output_mlm = output_mlm
        if output_mlm == True:
            self.W_mlm = nn.Parameter(torch.Tensor(d_model, d_model))
            self.b_mlm = nn.Parameter(torch.Tensor(d_model))
            self.norm_mlm = LayerNorm(self.d_model)
        
            self.share_emb_out_proj = share_emb_out_proj
            self.W_out_mlm = None
            if share_emb_out_proj == False:
                self.W_out_mlm = nn.Parameter(torch.Tensor(self.vocab_size, d_model))
            self.b_out_mlm = nn.Parameter(torch.Tensor(self.vocab_size))
        
        self.reset_parameters()
        

    def reset_parameters(self):
        """
        """
        weights = []
        bias = []
        if self.output_cls == True:
            weights.append(self.W_pool)
            weights.append(self.W_cls)
            bias.append(self.b_cls)
            bias.append(self.b_pool)
        if self.output_mlm == True:
            weights.append(self.W_mlm)
            bias.append(self.b_mlm)
        
            if self.share_emb_out_proj == False:
                weights.append(self.W_out_mlm)
            bias.append(self.b_out_mlm)        
        
        stdv = 1.0 / np.sqrt(self.d_model)
        for weight in weights:
            weight.data.uniform_(-stdv, stdv)
        for weight in bias:
            weight.data.zero_()
            
        
    def forward(self, x, types=None, attn_mask=None, return_states=False):
        """
        """
        outputs = []
        enc_self_attn_list = [] 

        if types is None:
            types = torch.zeros_like(x)
        
        enc_output, embeded, enc_states, enc_self_attn_list = self.encoder(x, attn_mask, types, return_states=True)
        
        outputs = []
        if self.output_cls == True:
            pool = enc_output[:,0,:]
            pool = torch.tanh(F.linear(pool, self.W_pool) + self.b_pool)
            cls = F.linear(pool, self.W_cls) + self.b_cls
            outputs.append(cls)
            
        if self.output_mlm == True:
            mlm = F.linear(enc_output, self.W_mlm) + self.b_mlm
            if self.activation == "relu":
                mlm = F.relu(mlm)
            elif self.activation == "gelu":
                mlm = F.gelu(mlm)
            mlm = self.norm_mlm(mlm)
            w = self.W_out_mlm
            if self.share_emb_out_proj == True:
                w = self.encoder.src_embedding.W
            logits = F.linear(mlm, w) + self.b_out_mlm
            outputs.append(logits)
            
        if return_states == True:
            outputs = outputs + [embeded, enc_states, enc_self_attn_list]
        
        return outputs
        

def load_bert_weights(bert, weight_path):
    """
    """
    bert.eval()
    state_dict = torch.load(weight_path)
    
    map_key_dict = {"encoder.src_embedding.W": "bert.embeddings.word_embeddings.weight",
                    "encoder.pos_embedding.W": "bert.embeddings.position_embeddings.weight",
                    "encoder.type_embedding.W":"bert.embeddings.token_type_embeddings.weight",
                    "encoder.norm_emb.alpha": "bert.embeddings.LayerNorm.gamma",
                    "encoder.norm_emb.bias": "bert.embeddings.LayerNorm.beta",
                    "W_pool": "bert.pooler.dense.weight",
                    "b_pool": "bert.pooler.dense.bias",
                    "W_cls": "cls.seq_relationship.weight",
                    "b_cls": "cls.seq_relationship.bias",        
                    "W_mlm": "cls.predictions.transform.dense.weight",
                    "b_mlm": "cls.predictions.transform.dense.bias",
                    "norm_mlm.alpha": "cls.predictions.transform.LayerNorm.gamma",
                    "norm_mlm.bias": "cls.predictions.transform.LayerNorm.beta",
                    "b_out_mlm": "cls.predictions.bias"}
    
    for i in range(bert.n_enc_layers):
        map_key_dict["encoder.layers.%d.attention.W_q" % i] = "bert.encoder.layer.%d.attention.self.query.weight" % i
        map_key_dict["encoder.layers.%d.attention.b_q" % i] = "bert.encoder.layer.%d.attention.self.query.bias" % i
        map_key_dict["encoder.layers.%d.attention.W_k" % i] = "bert.encoder.layer.%d.attention.self.key.weight" % i
        map_key_dict["encoder.layers.%d.attention.b_k" % i] = "bert.encoder.layer.%d.attention.self.key.bias" % i
        map_key_dict["encoder.layers.%d.attention.W_v" % i] = "bert.encoder.layer.%d.attention.self.value.weight" % i
        map_key_dict["encoder.layers.%d.attention.b_v" % i] = "bert.encoder.layer.%d.attention.self.value.bias" % i
        map_key_dict["encoder.layers.%d.attention.W_o" % i] = "bert.encoder.layer.%d.attention.output.dense.weight" % i
        map_key_dict["encoder.layers.%d.attention.b_o" % i] = "bert.encoder.layer.%d.attention.output.dense.bias" % i
        map_key_dict["encoder.layers.%d.norm_1.alpha" % i] = "bert.encoder.layer.%d.attention.output.LayerNorm.gamma" % i
        map_key_dict["encoder.layers.%d.norm_1.bias" % i] = "bert.encoder.layer.%d.attention.output.LayerNorm.beta" % i
        map_key_dict["encoder.layers.%d.ffn.W1" % i] = "bert.encoder.layer.%d.intermediate.dense.weight" % i
        map_key_dict["encoder.layers.%d.ffn.b1" % i] = "bert.encoder.layer.%d.intermediate.dense.bias" % i
        map_key_dict["encoder.layers.%d.ffn.W2" % i] = "bert.encoder.layer.%d.output.dense.weight" % i
        map_key_dict["encoder.layers.%d.ffn.b2" % i] = "bert.encoder.layer.%d.output.dense.bias" % i
        map_key_dict["encoder.layers.%d.norm_2.alpha" % i] = "bert.encoder.layer.%d.output.LayerNorm.gamma" % i
        map_key_dict["encoder.layers.%d.norm_2.bias" % i] = "bert.encoder.layer.%d.output.LayerNorm.beta" % i
    
    if bert.share_emb_out_proj == False:
        map_key_dict["W_out_mlm"] = "cls.predictions.decoder.weight"
    
    model_state_dict = {}
    for key,param in bert.named_parameters():
        model_state_dict[key] = state_dict[map_key_dict[key]]
    
    bert.load_state_dict(model_state_dict, False)
    
    return bert

if __name__ == "__main__":
    bert  = Bert({}, 
                 21128, 
                 512, 
                 12, 
                 768, 
                 3072, 
                 768//12,
                 768//12, 
                 12,
                 2,
                 0)
    load_bert_weights(bert, "C:/Users/39502/Desktop/bert-base-chinese/pytorch_model.bin")
