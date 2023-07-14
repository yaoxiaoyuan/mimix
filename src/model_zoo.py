# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 21:43:09 2023

@author: 1
"""
import torch

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
