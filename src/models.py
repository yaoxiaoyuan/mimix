# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:15:01 2019

@author: Xiaoyuan Yao
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Encoder,Decoder,CRF,LayerNorm,gelu_new

class Transformer(nn.Module):
    """
    """
    def __init__(self, 
                 symbol2id,
                 src_vocab_size,
                 src_max_len, 
                 trg_vocab_size, 
                 trg_max_len,
                 n_heads, 
                 d_model, 
                 d_ff, 
                 d_qk, 
                 d_v, 
                 n_enc_layers, 
                 n_dec_layers,
                 dropout=0, 
                 attn_dropout=0,
                 emb_dropout=0,
                 ln_eps=1e-5,
                 use_rms_norm=False,
                 use_attention_bias=True,
                 use_ffn_bias=True,
                 max_relative_len=-1,
                 use_rel_pos_value=False,
                 rel_pos_need_train=True,
                 share_src_trg_emb=False, 
                 share_emb_out_proj=False, 
                 embedding_size=None, 
                 share_layer_params=False, 
                 n_share_across_layers=1,
                 use_pre_norm=True, 
                 activation="relu", 
                 scale_embedding=False,
                 norm_before_pred=False,
                 norm_after_embedding=False,
                 pos_need_train=False,
                 use_output_bias=True):
        """
        """
        super(Transformer, self).__init__()
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        
        self.src_vocab_size = src_vocab_size
        self.src_max_len = src_max_len
        self.trg_vocab_size = trg_vocab_size
        self.trg_max_len = trg_max_len
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.dropout = dropout
        self.share_src_trg_emb = share_src_trg_emb
        self.share_emb_out_proj = share_emb_out_proj
        self.MAX_LOGITS = 10000.
        self.MIN_LOGITS = -10000.
        
        self.d = self.d_model // self.n_heads
        
        self.encoder = Encoder(src_vocab_size, 
                               src_max_len, 
                               n_heads,
                               d_model, 
                               d_ff, 
                               d_qk, 
                               d_v, 
                               n_enc_layers,
                               dropout, 
                               attn_dropout,
                               emb_dropout,
                               ln_eps,
                               use_rms_norm,
                               use_attention_bias,
                               use_ffn_bias,
                               max_relative_len,
                               use_rel_pos_value,
                               rel_pos_need_train,
                               embedding_size,
                               share_layer_params, 
                               n_share_across_layers,
                               use_pre_norm, 
                               activation, 
                               scale_embedding,
                               norm_before_pred,
                               norm_after_embedding,
                               pos_need_train)
        
        self.decoder = Decoder(trg_vocab_size,
                               trg_max_len, 
                               n_heads, 
                               d_model, 
                               d_ff, 
                               d_qk, 
                               d_v, 
                               n_dec_layers,
                               True,
                               dropout, 
                               attn_dropout,
                               emb_dropout,
                               ln_eps,
                               use_rms_norm,
                               use_attention_bias,
                               use_ffn_bias,
                               max_relative_len,
                               use_rel_pos_value,
                               rel_pos_need_train,
                               embedding_size,
                               share_layer_params, 
                               n_share_across_layers,
                               share_src_trg_emb,
                               share_emb_out_proj,
                               use_pre_norm, 
                               activation, 
                               scale_embedding,
                               norm_before_pred,
                               norm_after_embedding,
                               pos_need_train,
                               use_output_bias)
        
        
    def get_attn_mask(self, seq_query, seq_key):
        """
        """
        len_query = seq_query.size(1)
        mask = seq_key.eq(self.PAD).byte()
        mask = mask.unsqueeze(1).repeat(1, len_query, 1)
        
        return mask
    
    
    def get_subsequent_mask(self, seq):
        """
        """
        len_seq = seq.size(1)
        mask = torch.triu(torch.ones(len_seq, len_seq, 
                                     device=seq.device, dtype=torch.uint8), 
                          diagonal=1)

        return mask
        
    
    def forward(self, inputs, return_states=False, targets=None, compute_loss=False):
        """
        """
        x, y = inputs
        enc_self_attn_mask = self.get_attn_mask(x, x)
        dec_self_attn_mask = self.get_subsequent_mask(y)
        

        dec_self_attn_mask = dec_self_attn_mask | self.get_attn_mask(y, y)
        dec_enc_attn_mask = self.get_attn_mask(y, x)
        
        enc_outputs = self.encoder(x, enc_self_attn_mask,
                                   return_states=return_states)
        enc_output = enc_outputs[0]

        trg_embedding = None
        if self.share_src_trg_emb == True:
            trg_embedding = self.encoder.src_embedding

        dec_outputs = self.decoder(y, enc_output, 
                                   dec_self_attn_mask, dec_enc_attn_mask,
                                   return_states=return_states,
                                   trg_embedding=trg_embedding)
        
        outputs = dec_outputs + enc_outputs
        
        if return_states == False:
            outputs = outputs[:1]

        if compute_loss == True:
            loss = self.loss_fn(outputs, targets)
            outputs = [loss] + outputs
                    
        return outputs


    def init_search(self, x):
        """
        """
        enc_states = self.get_enc_states(x)
        
        dec_kv_list = []
        for i in range(self.n_dec_layers):
            dec_kv_list.append([None, None])
        
        dec_enc_attn_mask = x.eq(self.PAD).unsqueeze(1).byte()
        
        return enc_states, dec_kv_list, dec_enc_attn_mask


    def get_enc_states(self, x):
        """
        """
        enc_attn_mask = self.get_attn_mask(x, x)
        enc_outputs = self.encoder(x, enc_attn_mask)

        enc_output = enc_outputs[0]
        enc_states = self.decoder.cache_enc_kv(enc_output)

        return enc_states


    def gather_beam_states(self, 
                           dec_kv_list, 
                           dec_enc_attn_mask,
                           beam_id,
                           enc_kv_list=None):
        """
        """
        for i in range(self.n_dec_layers):
            if enc_kv_list is not None:
                enc_kv_list[i][0] = enc_kv_list[i][0][beam_id]
                enc_kv_list[i][1] = enc_kv_list[i][1][beam_id]
            
            dec_kv_list[i][0] = dec_kv_list[i][0][beam_id]
            dec_kv_list[i][1] = dec_kv_list[i][1][beam_id]
            
        dec_enc_attn_mask = dec_enc_attn_mask[beam_id]
        
        return dec_kv_list, dec_enc_attn_mask, enc_kv_list

    
    def step(self,
             steps, 
             enc_states, 
             dec_enc_attn_mask, 
             dec_states, 
             y):
        """
        """
        trg_embedding = None
        if self.share_src_trg_emb == True:
            trg_embedding = self.encoder.src_embedding
        return self.decoder.step(steps, 
                                 dec_states, 
                                 y,
                                 None,
                                 enc_states, 
                                 dec_enc_attn_mask, 
                                 trg_embedding)


class TransformerLM(nn.Module):
    """
    """
    def __init__(self, 
                 symbol2id, 
                 trg_vocab_size,
                 trg_max_len, 
                 n_heads, 
                 d_model, 
                 d_ff,
                 d_qk, 
                 d_v, 
                 n_dec_layers, 
                 dropout=0, 
                 attn_dropout=0,
                 emb_dropout=0,
                 ln_eps=1e-5,
                 use_rms_norm=False,
                 use_attention_bias=True,
                 use_ffn_bias=True,
                 max_relative_len=-1,
                 use_rel_pos_value=False,
                 rel_pos_need_train=True,
                 share_emb_out_proj=False, 
                 embedding_size=None, 
                 share_layer_params=False, 
                 n_share_across_layers=1,
                 use_pre_norm=True, 
                 activation="relu", 
                 scale_embedding=False,
                 norm_before_pred=False,
                 norm_after_embedding=False,
                 pos_need_train=False,
                 use_output_bias=False):
        """
        """
        super(TransformerLM, self).__init__()
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        
        self.trg_vocab_size = trg_vocab_size
        self.trg_max_len = trg_max_len
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_dec_layers = n_dec_layers
        self.dropout = dropout
        self.share_emb_out_proj = share_emb_out_proj
        
        self.d = self.d_model // self.n_heads
        
        self.MAX_LOGITS = 10000.
        self.MIN_LOGITS = -10000.
        
        self.decoder = Decoder(trg_vocab_size, 
                               trg_max_len, 
                               n_heads, 
                               d_model, 
                               d_ff, 
                               d_qk, 
                               d_v,
                               n_dec_layers,
                               False,
                               dropout, 
                               attn_dropout,
                               emb_dropout,
                               ln_eps,
                               use_rms_norm,
                               use_attention_bias,
                               use_ffn_bias,
                               max_relative_len,
                               use_rel_pos_value,
                               rel_pos_need_train,
                               embedding_size,
                               share_layer_params, 
                               n_share_across_layers,
                               False, 
                               share_emb_out_proj,
                               use_pre_norm, 
                               activation, 
                               scale_embedding,
                               norm_before_pred,
                               norm_after_embedding,
                               pos_need_train,
                               use_output_bias)
    
    def get_attn_mask(self, seq_query, seq_key):
        """
        """
        len_query = seq_query.size(1)
        mask = seq_key.eq(self.PAD).byte()
        mask = mask.unsqueeze(1).repeat(1, len_query, 1)
        
        return mask
    
    
    def get_subsequent_mask(self, seq):
        """
        """
        len_seq = seq.size(1)
        mask = torch.triu(torch.ones(len_seq, len_seq, 
                                     device=seq.device, dtype=torch.uint8), 
                          diagonal=1)
        
        return mask
        
    
    def forward(self, inputs, return_states=False, targets=None, compute_loss=False):
        """
        """
        y = inputs[0]
        dec_self_attn_mask = self.get_subsequent_mask(y)

        dec_self_attn_mask = dec_self_attn_mask | self.get_attn_mask(y, y)

        dec_outputs = self.decoder(y, dec_self_attn_mask, 
                                   return_states=return_states)

        outputs = dec_outputs
        
        if return_states == False:
            outputs = outputs[:1]

        if compute_loss == True:
            loss = self.loss_fn(outputs, targets)
            outputs = [loss] + outputs
            
        return outputs


    def step(self, steps, dec_kv_list, y):
        """
        """
        return self.decoder.step(steps, dec_kv_list, y)


    def init_search(self):
        """
        """
        dec_kv_list = []
        for i in range(self.n_dec_layers):
            dec_kv_list.append([None, None])
            
        return dec_kv_list
    
    
class TransformerEncoder(nn.Module):
    """
    """
    def __init__(self,
                 symbol2id,
                 src_vocab_size, 
                 src_max_len, 
                 n_heads, 
                 d_model,
                 d_ff, 
                 d_qk,
                 d_v, 
                 n_layers, 
                 dropout=0, 
                 attn_dropout=0,
                 emb_dropout=0,
                 ln_eps=1e-5,
                 use_rms_norm=False,
                 use_attention_bias=True,
                 use_ffn_bias=True,
                 max_relative_len=-1,
                 use_rel_pos_value=False,
                 rel_pos_need_train=True,
                 embedding_size=None,
                 share_layer_params=False, 
                 n_share_across_layers=1,
                 use_pre_norm=True, 
                 activation="relu", 
                 scale_embedding=False,
                 norm_before_pred=False,
                 norm_after_embedding=False,
                 pos_need_train=False,
                 n_types=None,
                 use_pooling=False,
                 out_dim=None,
                 n_class=None,
                 crf=False,
                 with_mlm=False,
                 share_emb_out_proj=False):

        super(TransformerEncoder, self).__init__()
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        
        self.src_vocab_size = src_vocab_size
        self.src_max_len = src_max_len
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout = dropout
        self.MAX_LOGITS = 10000.
        self.MIN_LOGITS = -10000.
        
        self.encoder = Encoder(src_vocab_size, 
                               src_max_len, 
                               n_heads,
                               d_model, 
                               d_ff, 
                               d_qk, 
                               d_v, 
                               n_layers,
                               dropout, 
                               attn_dropout,
                               emb_dropout,
                               ln_eps,
                               use_rms_norm,
                               use_attention_bias,
                               use_ffn_bias,
                               max_relative_len,
                               use_rel_pos_value,
                               rel_pos_need_train,
                               embedding_size,
                               share_layer_params, 
                               n_share_across_layers,
                               use_pre_norm, 
                               activation, 
                               scale_embedding,
                               norm_before_pred,
                               norm_after_embedding,
                               pos_need_train,
                               n_types,
                               use_pooling,
                               out_dim,
                               n_class,
                               crf,
                               with_mlm,
                               share_emb_out_proj)

        self.n_class = n_class
        self.share_emb_out_proj = share_emb_out_proj
        
        self.activation = activation
        
        self.W_pool = None
        self.b_pool = None
        self.W_out = None
        self.b_out = None
        self.use_pooling = use_pooling
        if use_pooling == True:
            self.W_pool = nn.Parameter(torch.Tensor(d_model, d_model))
            self.b_pool = nn.Parameter(torch.zeros(d_model))
        
        self.W_out = None
        self.b_out = None
        if self.n_class is not None:
            self.W_out = nn.Parameter(torch.Tensor(d_model, self.n_class))
            self.b_out = nn.Parameter(torch.zeros(self.n_class))
        
        self.out_dim = out_dim
        if out_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(d_model, self.out_dim))
            self.b_out = nn.Parameter(torch.zeros(self.out_dim))
        
        self.crf = None
        if self.n_class is not None and crf == True:
            self.crf = CRF(self.n_labels)
        
        self.W_mlm = None
        self.b_mlm = None
        self.W_mlm = None
        self.b_mlm = None
        self.with_mlm = with_mlm
        if self.with_mlm == True:
            self.b_mlm = nn.Parameter(torch.Tensor(d_model))
            self.norm_mlm = LayerNorm(self.d_model)
        
            self.share_emb_out_proj = share_emb_out_proj
            if share_emb_out_proj == False:
                self.W_mlm = nn.Parameter(torch.Tensor(d_model, self.src_vocab_size))
            self.b_mlm = nn.Parameter(torch.Tensor(self.src_vocab_size))

        self.reset_parameters()
    
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.d_model)
        for weight in [self.W_pool, self.W_out, self.W_mlm]:
            if weight is not None:
                weight.data.uniform_(-stdv, stdv)
        for weight in [self.b_pool, self.b_out, self.b_mlm]:
            if weight is not None:
                weight.data.zero_()
                
                
    def get_attn_mask(self, seq_query, seq_key):
        """
        """
        len_query = seq_query.size(1)
        mask = seq_key.eq(self.PAD)
        mask = mask.unsqueeze(1).repeat(1, len_query, 1)
        
        return mask
    
    
    def forward(self, inputs, return_states=False, targets=None, compute_loss=False):
        """
        """
        x = inputs[0]

        enc_self_attn_mask = self.get_attn_mask(x, x)

        enc_outputs = self.encoder(x, 
                                   enc_self_attn_mask, 
                                   return_states=return_states)
        enc_output = enc_outputs[0]
        
        if self.use_pooling == True:
            
            enc_output = torch.tanh(torch.matmul(enc_output, self.W_pool) + self.b_pool)
            
            outputs = [enc_output]
            if self.out_dim is not None or self.n_class is not None:
                enc_output = torch.matmul(enc_output, self.W_out) + self.b_out

                outputs = [enc_output] + outputs
            
            if self.crf is not None:
                mask = x.ne(self.PAD).float()
                nlg = self.crf(enc_output, x, mask)
                outputs = [nlg] + outputs 
        
        if self.with_mlm == True:
            if self.activation == "relu":
                enc_output = F.relu(enc_output)
            elif self.activation == "gelu":
                enc_output = gelu_new(enc_output)
            
            enc_output = self.norm_mlm(enc_output)

            if self.share_emb_out_proj == False:
                W = self.W_mlm
            else:
                W = self.encoder.src_embedding.get_embedding().T
            
            logits = torch.matmul(enc_output, W) + self.b_mlm
            
            outputs = [logits]
        
        outputs = outputs + enc_outputs
        
        if return_states == False:
            outputs = outputs[:1]

        if compute_loss == True:
            loss = self.loss_fn(outputs, targets)
            outputs = [loss] + outputs
                    
        return outputs        


def build_transformer_model(config):
    """
    """
    src_vocab_size = config["src_vocab_size"]
    src_max_len = config["src_max_len"]
    trg_vocab_size = config["trg_vocab_size"]
    trg_max_len = config["trg_max_len"]
    n_heads = config["n_heads"]
    d_model = config["d_model"]
    d_ff = config["d_ff"]
    d_qk = config.get("d_qk", d_model//n_heads) 
    d_v = config.get("d_v", d_model//n_heads) 
    n_enc_layers = config["n_enc_layers"]
    n_dec_layers = config["n_dec_layers"]
    dropout = config.get("dropout", 0)
    attn_dropout = config.get("attn_dropout", 0)
    emb_dropout = config.get("emb_dropout", 0)
    ln_eps = config.get("ln_eps", 1e-5)
    use_rms_norm = config.get("use_rms_norm", False)
    use_attention_bias = config.get("use_attention_bias", True)
    use_ffn_bias = config.get("use_ffn_bias", True)
    max_relative_len = config.get("max_relative_len", -1)
    use_rel_pos_value = config.get("use_rel_pos_value", False)
    rel_pos_need_train = config.get("rel_pos_need_train", True)
    share_src_trg_emb = config["share_src_trg_emb"]
    share_emb_out_proj = config.get("share_emb_out_proj", False)
    embedding_size = config.get("embedding_size", None)
    share_layer_params = config.get("share_layer_params", False)
    n_share_across_layers = config.get("n_share_across_layers", 1)
    use_pre_norm = config.get("use_pre_norm", True)
    activation = config.get("activation", "relu")
    scale_embedding = config.get("scale_embedding", False)
    norm_before_pred = config.get("norm_before_pred", False)
    norm_after_embedding = config.get("norm_after_embedding", False)
    pos_need_train = config.get("pos_need_train", False)
    use_output_bias = config.get("use_output_bias", True)
    transformer = Transformer(config["symbol2id"],
                              src_vocab_size, 
                              src_max_len, 
                              trg_vocab_size, 
                              trg_max_len, 
                              n_heads, 
                              d_model, 
                              d_ff, 
                              d_qk, 
                              d_v,
                              n_enc_layers,
                              n_dec_layers, 
                              dropout, 
                              attn_dropout, 
                              emb_dropout,
                              ln_eps,
                              use_rms_norm,
                              use_attention_bias,
                              use_ffn_bias,
                              max_relative_len,
                              use_rel_pos_value,
                              rel_pos_need_train,
                              share_src_trg_emb, 
                              share_emb_out_proj,
                              embedding_size,
                              share_layer_params, 
                              n_share_across_layers,
                              use_pre_norm,
                              activation,
                              scale_embedding,
                              norm_before_pred,
                              norm_after_embedding,
                              pos_need_train,
                              use_output_bias)
    
    return transformer


def build_enc_dec_model(config):
    """
    """
    if config["model"] == "transformer":
        model = build_transformer_model(config)
    else:
        raise ValueError("model not correct!")
        
    return model


def build_transformer_lm_model(config):
    """
    """
    trg_vocab_size = config["trg_vocab_size"]
    trg_max_len = config["trg_max_len"]
    n_heads = config["n_heads"]
    d_model = config["d_model"]
    d_ff = config["d_ff"]
    d_qk = config.get("d_qk", d_model//n_heads) 
    d_v = config.get("d_v", d_model//n_heads) 
    n_dec_layers = config["n_dec_layers"]
    dropout = config.get("dropout", 0)
    attn_dropout = config.get("attn_dropout", 0)
    emb_dropout = config.get("emb_dropout", 0)
    ln_eps = config.get("ln_eps", 1e-5)
    use_rms_norm = config.get("use_rms_norm", False)
    use_attention_bias = config.get("use_attention_bias", True)
    use_ffn_bias = config.get("use_ffn_bias", True)
    max_relative_len = config.get("max_relative_len", -1)
    use_rel_pos_value = config.get("use_rel_pos_value", False)
    rel_pos_need_train = config.get("rel_pos_need_train", True)
    share_emb_out_proj = config.get("share_emb_out_proj", False)
    share_layer_params = config.get("share_layer_params", False)
    embedding_size = config.get("embedding_size", None)
    n_share_across_layers = config.get("n_share_across_layers", 1)
    use_pre_norm = config.get("use_pre_norm", True)
    activation = config.get("activation", "relu")
    scale_embedding = config.get("scale_embedding", False)
    norm_before_pred = config.get("norm_before_pred", False)
    norm_after_embedding = config.get("norm_after_embedding", False)
    pos_need_train = config.get("pos_need_train", False)
    use_output_bias = config.get("use_output_bias", True)
    
    transformer = TransformerLM(config["symbol2id"],
                                trg_vocab_size, 
                                trg_max_len, 
                                n_heads, 
                                d_model, 
                                d_ff,
                                d_qk,
                                d_v,
                                n_dec_layers, 
                                dropout, 
                                attn_dropout, 
                                emb_dropout, 
                                ln_eps,
                                use_rms_norm,
                                use_attention_bias,
                                use_ffn_bias,
                                max_relative_len,
                                use_rel_pos_value,
                                rel_pos_need_train,
                                share_emb_out_proj,
                                embedding_size,
                                share_layer_params,
                                n_share_across_layers,
                                use_pre_norm,
                                activation,
                                scale_embedding,
                                norm_before_pred,
                                norm_after_embedding,
                                pos_need_train,
                                use_output_bias)
    
    return transformer


def build_lm_model(config):
    """
    """
    if config["model"] == "transformer":
        model = build_transformer_lm_model(config)
    else:
        raise ValueError("model not correct!")
        
    return model


def build_transformer_encoder_model(config):
    """
    """
    src_vocab_size = config["src_vocab_size"]
    src_max_len = config["src_max_len"]
    n_heads = config["n_heads"]
    d_model = config["d_model"]
    d_ff = config["d_ff"]
    d_qk = config.get("d_qk", d_model//n_heads) 
    d_v = config.get("d_v", d_model//n_heads) 
    n_enc_layers = config["n_enc_layers"]
    n_types = config.get("n_types", None)
    use_pooling = config.get("use_pooling", True)
    out_dim = config.get("out_dim", None)
    n_class = config.get("n_class", None)
    crf = config.get("crf", False)
    with_mlm = config.get("with_mlm", False)
    share_emb_out_proj = config.get("share_emb_out_proj", False)
    dropout = config.get("dropout", 0)
    attn_dropout = config.get("attn_dropout", 0)
    emb_dropout = config.get("emb_dropout", 0)
    ln_eps = config.get("ln_eps", 1e-5)
    use_rms_norm = config.get("use_rms_norm", False)
    use_attention_bias = config.get("use_attention_bias", True)
    use_ffn_bias = config.get("use_ffn_bias", True)
    max_relative_len = config.get("max_relative_len", -1)
    use_rel_pos_value = config.get("use_rel_pos_value", False)
    rel_pos_need_train = config.get("rel_pos_need_train", True)
    share_layer_params = config.get("share_layer_params", False)
    embedding_size = config.get("embedding_size", None)
    n_share_across_layers = config.get("n_share_across_layers", 1)
    use_pre_norm = config.get("use_pre_norm", True)
    activation = config.get("activation", "relu")
    scale_embedding = config.get("scale_embedding", False)
    norm_before_pred = config.get("norm_before_pred", False)
    norm_after_embedding = config.get("norm_after_embedding", False)
    pos_need_train = config.get("pos_need_train", False)
    
    transformer = TransformerEncoder(config["symbol2id"],
                                     src_vocab_size, 
                                     src_max_len, 
                                     n_heads,
                                     d_model, 
                                     d_ff, 
                                     d_qk, 
                                     d_v,
                                     n_enc_layers,
                                     dropout,
                                     attn_dropout,
                                     emb_dropout,
                                     ln_eps,
                                     use_rms_norm,
                                     use_attention_bias,
                                     use_ffn_bias,
                                     max_relative_len,
                                     use_rel_pos_value,
                                     rel_pos_need_train,
                                     embedding_size,
                                     share_layer_params,
                                     n_share_across_layers,
                                     use_pre_norm,
                                     activation,
                                     scale_embedding,
                                     norm_before_pred,
                                     norm_after_embedding,
                                     pos_need_train,
                                     n_types,
                                     use_pooling,
                                     out_dim,
                                     n_class,
                                     crf,
                                     with_mlm,
                                     share_emb_out_proj)
    
    return transformer


def build_encoder_model(config):
    """
    """
    if config["model"] == "transformer":
        model = build_transformer_encoder_model(config)
    else:
        raise ValueError("model not correct!")
        
    return model

 
model_builder_dict = {
            "enc_dec": build_enc_dec_model,
            "lm": build_lm_model,
            "classify": build_transformer_encoder_model,
            "bi_lm": build_transformer_encoder_model,
            "sequence_labeling": build_transformer_encoder_model,
            "match": build_transformer_encoder_model,
        }


def build_model(config):
    """
    """
    if config["task"] in model_builder_dict:
        return model_builder_dict[config["task"]](config)
    else:
        raise ValueError("model not correct!")
