# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:15:01 2019

@author: Xiaoyuan Yao
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Embedding, GRU,LSTM
from layers import Encoder,Decoder,LMDecoder
from layers import RNNEncoder,AttentionDecoder
from layers import CRF
from layers import gelu_new,LayerNorm
from bert import Bert

class TransformerClassifer(nn.Module):
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
                 n_enc_layers,    
                 n_class,
                 dropout=0, 
                 attn_dropout=0,
                 emb_dropout=0,
                 ln_eps=1e-5,
                 embedding_size=None, 
                 share_layer_params=False, 
                 n_share_across_layers=0,
                 use_pre_norm=True, 
                 activation="relu", 
                 scale_embedding=False,
                 norm_before_pred=False,
                 norm_after_embedding=False,
                 pos_need_train=False):
        """
        """
        super(TransformerClassifer, self).__init__()
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
        self.n_enc_layers = n_enc_layers
        self.dropout = dropout
        self.n_class = n_class
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
                               embedding_size,
                               share_layer_params, 
                               n_share_across_layers,
                               use_pre_norm,
                               activation, 
                               scale_embedding,
                               norm_before_pred,
                               norm_after_embedding,
                               pos_need_train)
        
        self.W_pool = nn.Parameter(torch.Tensor(d_model, d_model))
        self.b_pool = nn.Parameter(torch.zeros(d_model))
        self.W = nn.Parameter(torch.Tensor(d_model, n_class))
        self.b = nn.Parameter(torch.zeros(n_class))
        
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.d_model)
        for weight in [self.W_pool, self.W]:
            weight.data.uniform_(-stdv, stdv)
    
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
        
        enc_output = enc_outputs[0][:,0,:]
        
        enc_output = torch.tanh(F.linear(enc_output, self.W_pool) + self.b_pool)
        
        logits = torch.matmul(enc_output, self.W) + self.b

        outputs = [logits]
        
        if return_states == True:
            outputs = outputs + enc_outputs 

        if compute_loss == True:
            loss = self.loss_fn(outputs, targets)
            outputs = [loss] + outputs
        
        return outputs


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
                 n_enc_layers,    
                 dropout=0, 
                 attn_dropout=0,
                 emb_dropout=0,
                 ln_eps=1e-5,
                 embedding_size=None, 
                 share_layer_params=False, 
                 n_share_across_layers=0,
                 use_pre_norm=True, 
                 activation="relu", 
                 scale_embedding=False,
                 norm_before_pred=False,
                 norm_after_embedding=False,
                 pos_need_train=False):
        """
        """
        super(TransformerEncoder, self).__init__()
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        
        self.MIN_LOGITS = -10000.
        
        self.src_vocab_size = src_vocab_size
        self.src_max_len = src_max_len
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_enc_layers = n_enc_layers
        self.dropout = dropout
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
                               embedding_size,
                               share_layer_params, 
                               n_share_across_layers,
                               use_pre_norm,
                               activation, 
                               scale_embedding,
                               norm_before_pred,
                               norm_after_embedding,
                               pos_need_train)
        
        self.W2 = nn.Parameter(torch.Tensor(d_model, d_model))
        self.b2 = nn.Parameter(torch.zeros(d_model))
        self.W3 = nn.Parameter(torch.Tensor(d_model, d_model))
        self.b3 = nn.Parameter(torch.zeros(d_model))
        
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.d_model)
        for weight in [self.W2, self.W3]:
            weight.data.uniform_(-stdv, stdv)
    
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

        enc_output = enc_outputs[0][:,0,:]

        enc_output = torch.matmul(torch.tanh(torch.matmul(enc_output, self.W2) + self.b2), self.W3) + self.b3
        
        norm_vec = F.normalize(enc_output, p=2, dim=1)
        sim = torch.mm(norm_vec, norm_vec.T)
        sim = sim + self.MIN_LOGITS * torch.eye(sim.shape[0], device=sim.device)

        outputs = [sim, enc_output]
        
        if return_states == True:
            outputs = outputs + enc_outputs 

        if compute_loss == True:
            loss = self.loss_fn(outputs, targets)
            outputs = [loss] + outputs
        
        return outputs


class TransformerCRF(nn.Module):
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
                 n_enc_layers,
                 n_labels,
                 dropout=0, 
                 attn_dropout=0,
                 emb_dropout=0,
                 ln_eps=1e-5,
                 embedding_size=None, 
                 share_layer_params=False,
                 n_share_across_layers=0,
                 use_pre_norm=True, 
                 activation="relu", 
                 scale_embedding=False,
                 norm_before_pred=False,
                 norm_after_embedding=False,
                 pos_need_train=False):
        """
        """
        super(TransformerCRF, self).__init__()
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        
        self.use_crf = True
        
        self.src_vocab_size = src_vocab_size
        self.src_max_len = src_max_len
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_enc_layers = n_enc_layers
        self.dropout = dropout
        self.n_labels = n_labels
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
                               embedding_size,
                               share_layer_params, 
                               n_share_across_layers,
                               use_pre_norm,
                               activation, 
                               scale_embedding,
                               norm_before_pred,
                               norm_after_embedding,
                               pos_need_train)
        self.W = nn.Parameter(torch.Tensor(d_model, n_labels))

        self.crf = CRF(n_labels)
        
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.d_model)
        for weight in [self.W]:
            weight.data.uniform_(-stdv, stdv)        
    
    
    def get_attn_mask(self, seq_query, seq_key):
        """
        """
        len_query = seq_query.size(1)
        mask = seq_key.eq(self.PAD)
        mask = mask.unsqueeze(1).repeat(1, len_query, 1)
        
        return mask
        

    def get_emission(self, x):
        """
        """
        enc_self_attn_mask = self.get_attn_mask(x, x)

        enc_outputs = self.encoder(x, enc_self_attn_mask)

        emission = torch.matmul(enc_outputs[0], self.W)
        
        return emission

    
    def forward(self, inputs, return_states=False, targets=None, compute_loss=False):
        """
        """
        x,y = inputs[:2]
        enc_self_attn_mask = self.get_attn_mask(x, x)

        enc_outputs = self.encoder(x, enc_self_attn_mask, 
                                   return_states=return_states)

        emission = torch.matmul(enc_outputs[0], self.W)

        mask = x.ne(self.PAD).float()
        nlg = self.crf(emission, y, mask)
        
        outputs = [nlg]
        
        if return_states == True:
            outputs = outputs + enc_outputs 

        if compute_loss == True:
            loss = self.loss_fn(outputs, targets)
            outputs = [loss] + outputs
                    
        return outputs


class TransformerSeqCls(nn.Module):
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
                 n_enc_layers, 
                 n_labels,
                 dropout=0, 
                 attn_dropout=0,
                 emb_dropout=0,
                 ln_eps=1e-5,
                 embedding_size=None, 
                 share_layer_params=False,
                 n_share_across_layers=0,
                 use_pre_norm=True, 
                 activation="relu", 
                 scale_embedding=False,
                 norm_before_pred=False,
                 norm_after_embedding=False,
                 pos_need_train=False):
        """
        """
        super(TransformerSeqCls, self).__init__()
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        
        self.use_crf = False
        
        self.src_vocab_size = src_vocab_size
        self.src_max_len = src_max_len
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_enc_layers = n_enc_layers
        self.dropout = dropout
        self.n_labels = n_labels
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
                               embedding_size,
                               share_layer_params, 
                               n_share_across_layers,
                               use_pre_norm,
                               activation, 
                               scale_embedding,
                               norm_before_pred,
                               norm_after_embedding,
                               pos_need_train)
        self.W = nn.Parameter(torch.Tensor(d_model, n_labels))
        
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.d_model)
        for weight in [self.W]:
            weight.data.uniform_(-stdv, stdv)        
    
    
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

        logits = torch.matmul(enc_outputs[0], self.W)
        
        outputs = [logits]
        
        if return_states == True:
            outputs = outputs + enc_outputs 

        if compute_loss == True:
            loss = self.loss_fn(outputs, targets)
            outputs = [loss] + outputs
                    
        return outputs
    
    
    def compute_logits(self, inputs):
        """
        """
        return self.forward(inputs)[0].argmax(-1)


class CnnTextClassifier(nn.Module):
    """
    """
    def __init__(self, symbol2id, vocab_size, n_classes, emb_size, 
                 num_filters, window_sizes=(3, 4, 5)):
        """
        """
        super(CnnTextClassifier, self).__init__()
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        
        self.embedding = nn.Embedding(vocab_size, emb_size)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, [window_size, emb_size],
                      padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])

        self.fc = nn.Linear(num_filters * len(window_sizes), n_classes)
    
        self.reset_parameters()
        
    
    def reset_parameters(self):
        """
        """
        pass
            
            
    def forward(self, inputs, targets=None, compute_loss=False):
        """
        """
        x = inputs[0]
        x = self.embedding(x)           # [B, T, E]

        # Apply a convolution + max pool layer for each window size
        x = torch.unsqueeze(x, 1)       # [B, C, T, E] Add a channel dim.
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))        # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)            # [B, F, window]

        # FC
        x = x.view(x.size(0), -1)       # [B, F * window]
        logits = self.fc(x)             # [B, class]

        outputs = [logits]

        if compute_loss == True:
            loss = self.loss_fn(outputs, targets)
            outputs = [loss] + outputs
                
        return outputs


class RNNLM(nn.Module):
    """
    """
    def __init__(self, symbol2id, 
                 trg_vocab_size, trg_emb_size, trg_max_len,
                 dec_hidden_size, num_layers, cell_type, 
                 share_emb_out_proj=True, dropout=0):
        """
        """
        super(RNNLM, self).__init__()
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        self.trg_embedding = Embedding(trg_vocab_size, trg_emb_size)
        self.trg_emb_size = trg_emb_size
        self.trg_max_len = trg_max_len
        self.trg_vocab_size = trg_vocab_size
        self.dec_hidden_size = dec_hidden_size
        self.cell_type = cell_type
        self.num_layers = num_layers
        
        self.MAX_LOGITS = 10000.
        self.MIN_LOGITS = -10000.
        
        layers = []
        for i in range(self.num_layers):
            if self.cell_type == "gru":
                if i == 0:
                    layers.append(GRU(trg_emb_size, 
                                      dec_hidden_size, 
                                      dropout=dropout))
                else:
                    layers.append(GRU(self.dec_hidden_size, 
                                      dec_hidden_size, 
                                      dropout=dropout))
            elif self.cell_type == "lstm":
                if i == 0:
                    layers.append(LSTM(trg_emb_size, 
                                       dec_hidden_size, 
                                       dropout=dropout))
                else:
                    layers.append(LSTM(self.dec_hidden_size, 
                                       dec_hidden_size, 
                                       dropout=dropout))
            else:
                raise ValueError("rnn cell type not correct!")
        
        self.layers = nn.ModuleList(layers)
        
        self.share_emb_out_proj = share_emb_out_proj
        if share_emb_out_proj == False:
            self.W = nn.Parameter(torch.Tensor(self.trg_emb_size, 
                                               self.dec_hidden_size))
        
        self.reset_parameters()


    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.dec_hidden_size)
        if self.share_emb_out_proj == False:
            for weight in [self.W]:
                weight.data.uniform_(-stdv, stdv)


    def get_non_pad_mask(self, seq):
        """
        """
        return seq.gt(self.PAD).type(torch.float).unsqueeze(2)


    def _step(self, steps, last_dec_states, y):
        """
        """
        y_embedding = self.trg_embedding(y)
        y_mask = self.get_non_pad_mask(y)
        
        dec_states_list = []
        for i in range(self.num_layers):
            if self.cell_type == "gru":
                if i == 0:
                    dec_states = self.layers[i](y_embedding, 
                                                y_mask, 
                                                last_dec_states[i])
                else:
                    dec_states = self.layers[i](dec_states, 
                                                y_mask, 
                                                last_dec_states[i])
            elif self.cell_type == "lstm":
                if i == 0:
                    dec_states = self.layers[i](y_embedding, 
                                                y_mask, 
                                                last_dec_states[i])
                else:
                    dec_states = self.layers[i](dec_states, 
                                                y_mask, 
                                                last_dec_states[i])
            
            dec_states_list.append(dec_states)
            
            if self.cell_type == "lstm":
                dec_states = dec_states[0]
            
        
        if self.share_emb_out_proj == True:
            W = self.trg_embedding.W
        else:
            W = self.W
        
        logits = F.linear(dec_states, W)
        logits = logits.view(-1, self.trg_vocab_size)
        outputs = [dec_states_list, logits]
        
        return outputs
    

    def forward(self, inputs, return_states=True, targets=None, compute_loss=False):
        """
        """
        y = inputs[0]
        y_mask = self.get_non_pad_mask(y)
        
        y_embedding = self.trg_embedding(y)
        
        for i in range(self.num_layers):
            if self.cell_type == "gru":
                if i == 0:
                    dec_states = self.layers[i](y_embedding, 
                                                y_mask, 
                                                None)
                else:
                    dec_states = self.layers[i](dec_states, 
                                                y_mask, 
                                                None)
            elif self.cell_type == "lstm":
                if i == 0:
                    dec_states = self.layers[i](y_embedding, 
                                                y_mask, 
                                                [None, None])
                else:
                    dec_states = self.layers[i](dec_states, 
                                                y_mask, 
                                                [None, None])
                dec_states = dec_states[0]
        
        if self.share_emb_out_proj == True:
            W = self.trg_embedding.W
        else:
            W = self.W
        
        logits = F.linear(dec_states, W)
        outputs = [logits]
        
        if return_states == True:
            outputs.append(dec_states)

        if compute_loss == True:
            loss = self.loss_fn(outputs, targets)
            outputs = [loss] + outputs
                
        return outputs


    def init_search(self):
        """
        """        
        dec_states = []
        for i in range(self.num_layers):
            if self.cell_type == "gru":
                dec_states.append(None)
            elif self.cell_type == "lstm":
                dec_states.append([None, None])
        return dec_states


class Seq2seq(nn.Module):
    """
    """
    def __init__(self, 
                 symbol2id,
                 src_max_len,
                 trg_max_len,
                 src_vocab_size,
                 src_emb_size,
                 trg_vocab_size, 
                 trg_emb_size,
                 enc_hidden_size, 
                 dec_hidden_size,
                 attention, 
                 num_enc_layers, 
                 num_dec_layers,
                 cell_type, 
                 dropout, 
                 share_src_trg_emb):
        """
        """
        super(Seq2seq, self).__init__()
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        
        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len
        self.src_vocab_size = src_vocab_size
        self.src_emb_size = src_emb_size
        self.trg_vocab_size = trg_vocab_size
        self.trg_emb_size = trg_emb_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.attention = attention
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.cell_type = cell_type
        self.dropout = dropout
        self.share_src_trg_emb = share_src_trg_emb

        self.MAX_LOGITS = 10000.
        self.MIN_LOGITS = -10000.
        
        #To facilitate the Luong attention
        query_size = self.dec_hidden_size
        key_size = query_size
        value_size = 2 * self.enc_hidden_size

        self.encoder = RNNEncoder(self.src_vocab_size,
                                   self.src_emb_size,
                                   self.enc_hidden_size,
                                   key_size,
                                   self.num_enc_layers,
                                   self.cell_type,
                                   self.dropout)
            
        self.decoder = AttentionDecoder(self.trg_vocab_size,
                                         self.trg_emb_size, 
                                         self.dec_hidden_size, 
                                         key_size,
                                         query_size,
                                         value_size,
                                         self.attention,
                                         self.num_dec_layers,
                                         self.cell_type,
                                         self.dropout,
                                         share_src_trg_emb=share_src_trg_emb)

    
    def get_non_pad_mask(self, seq):
        """
        """
        return seq.gt(self.PAD).type(torch.float).unsqueeze(2)
    
    
    def forward(self, inputs, targets=None, compute_loss=False):
        """
        """
        x, y = inputs
        x_mask = self.get_non_pad_mask(x)
        enc_states, enc_keys, enc_values = self.encoder(x, x_mask)
        
        trg_embedding = None
        if self.share_src_trg_emb == True:
            trg_embedding = self.encoder.src_embedding
        
        outputs = self.decoder(enc_keys, enc_values, x_mask, y, trg_embedding)
        
        if compute_loss == True:
            loss = self.loss_fn(outputs, targets)
            outputs = [loss] + outputs
                
        return outputs


    def init_search(self, x):
        """
        """
        enc_states = self.get_enc_states(x)
        
        if self.cell_type == "gru":
            dec_states = [None for i in range(self.num_dec_layers)]
        elif self.cell_type == "lstm":
            dec_states = [[None, None] for i in range(self.num_dec_layers)]
            
        dec_enc_attn_mask = self.get_non_pad_mask(x)
        
        return enc_states, dec_states, dec_enc_attn_mask


    def get_enc_states(self, x):
        """
        """
        x_mask = self.get_non_pad_mask(x)
        enc_states, enc_keys, enc_values = self.encoder(x, x_mask)
        cache_enc_states = [enc_keys, enc_values]
        return cache_enc_states
    
    
    def get_attn_mask(self, y, x):
        """
        """
        x_mask = self.get_non_pad_mask(x)
        return x_mask


    def gather_beam_states(self, 
                           dec_states, 
                           dec_enc_attn_mask,
                           beam_id,
                           enc_kv_list=None):
        """
        """
        if self.cell_type == "gru":
            for i,s in enumerate(dec_states):
                dec_states[i] = dec_states[i][beam_id]
        elif self.cell_type == "lstm":
            for i,s in enumerate(dec_states):
                dec_states[i][0] = dec_states[i][0][beam_id]
                dec_states[i][1] = dec_states[i][1][beam_id]
        
        if enc_kv_list is not None:
            enc_keys, enc_values = enc_kv_list
            enc_keys = enc_keys[beam_id] 
            enc_values = enc_values[beam_id]
            enc_kv_list = [enc_keys, enc_values]
            
        dec_enc_attn_mask = dec_enc_attn_mask[beam_id]
        
        return dec_states, dec_enc_attn_mask, enc_kv_list
    
    
    def step(self,
             steps, 
             enc_states, 
             dec_enc_attn_mask, 
             dec_states, 
             y):
        """
        """
        trg_embedding = False
        if self.share_src_trg_emb == True:
            trg_embedding = self.encoder.src_embedding
        return self.decoder._step(steps, 
                                  enc_states, 
                                  dec_enc_attn_mask, 
                                  dec_states, 
                                  y,
                                  trg_embedding)


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
                 use_proj_bias=True):
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
                               dropout, 
                               attn_dropout,
                               emb_dropout,
                               ln_eps,
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
                               use_proj_bias)
        
        
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
        return self.decoder._step(steps, 
                                  enc_states, 
                                  dec_enc_attn_mask, 
                                  dec_states, 
                                  y,
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
                 use_proj_bias=False):
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
        
        self.decoder = LMDecoder(trg_vocab_size, 
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
                                 use_proj_bias)
    
    
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


    def _step(self, steps, dec_kv_list, y):
        """
        """
        return self.decoder._step(steps, dec_kv_list, y)


    def init_search(self):
        """
        """
        dec_kv_list = []
        for i in range(self.n_dec_layers):
            dec_kv_list.append([None, None])
            
        return dec_kv_list


class TransformerBiLM(nn.Module):
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
                 n_enc_layers,
                 dropout=0, 
                 attn_dropout=0,
                 emb_dropout=0,
                 ln_eps=1e-5,
                 share_emb_out_proj=False, 
                 embedding_size=None, 
                 share_layer_params=False, 
                 n_share_across_layers=0,
                 use_pre_norm=True,
                 activation="relu", 
                 scale_embedding=False,
                 norm_before_pred=False,
                 norm_after_embedding=False,
                 pos_need_train=False):
        """
        """
        super(TransformerBiLM, self).__init__()
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
        self.n_enc_layers = n_enc_layers
        self.dropout = dropout
        self.activation = activation
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
                               embedding_size,
                               share_layer_params, 
                               n_share_across_layers,
                               use_pre_norm,
                               activation, 
                               scale_embedding,
                               norm_before_pred,
                               norm_after_embedding,
                               pos_need_train)
        self.norm = LayerNorm(d_model)
        self.share_emb_out_proj = share_emb_out_proj
        if share_emb_out_proj == False:
            self.share_emb_out_proj = share_emb_out_proj
            self.W = nn.Parameter(torch.Tensor(d_model, src_vocab_size))
        
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.d_model)
        if self.share_emb_out_proj == False:
            for weight in [self.W]:
                weight.data.uniform_(-stdv, stdv)
    
    
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

        enc_outputs = self.encoder(x, enc_self_attn_mask, 
                                   return_states=return_states)

        enc_output = enc_outputs[0]

        if self.activation == "relu":
            enc_output = F.relu(enc_output)
        elif self.activation == "gelu":
            enc_output = gelu_new(enc_output)
        enc_output = self.norm(enc_output)
            
        if self.share_emb_out_proj == False:
            W = self.W
        else:
            W = self.encoder.src_embedding.get_embedding().T
        
        logits = torch.matmul(enc_output, W)
        
        outputs = [logits]
        
        if return_states == True:
            outputs = outputs + enc_outputs 
            
        if compute_loss == True:
            loss = self.loss_fn(outputs, targets)
            outputs = [loss] + outputs
            
        return outputs


def build_seq2seq_model(config):
    """
    """
    src_max_len = config["src_max_len"]
    trg_max_len = config["trg_max_len"]
    src_vocab_size = config["src_vocab_size"]
    src_emb_size = config["src_emb_size"]
    trg_vocab_size = config["trg_vocab_size"]
    trg_emb_size = config["trg_emb_size"]
    enc_hidden_size = config["enc_hidden_size"]
    dec_hidden_size = config["dec_hidden_size"]
    attention = config["attention"]
    num_enc_layers = config["n_enc_layers"]
    num_dec_layers = config["n_dec_layers"]
    cell_type = config["cell_type"]
    dropout = config.get("dropout", 0)
    share_emb = config["share_emb"]
    
    seq2seq = Seq2seq(config["symbol2id"],
                      src_max_len,
                      trg_max_len,
                      src_vocab_size,
                      src_emb_size,
                      trg_vocab_size, 
                      trg_emb_size,
                      enc_hidden_size, 
                      dec_hidden_size,
                      attention, 
                      num_enc_layers, 
                      num_dec_layers,
                      cell_type, 
                      dropout, 
                      share_emb)
    
    return seq2seq
    

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
    use_proj_bias = config.get("pos_need_train", True)
    
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
                              use_proj_bias)
    
    return transformer


def build_enc_dec_model(config):
    """
    """
    if config["model"] == "transformer":
        model = build_transformer_model(config)
    elif config["model"] == "seq2seq":
        model = build_seq2seq_model(config)
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
    use_proj_bias = config.get("pos_need_train", True)
    
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
                                use_proj_bias)
    
    return transformer


def build_rnn_lm_model(config):
    """
    """
    trg_vocab_size = config["trg_vocab_size"]
    trg_max_len = config["trg_max_len"]
    cell_type = config["cell_type"]
    n_dec_layers = config["n_dec_layers"]
    dec_hidden_size = config["dec_hidden_size"]
    n_dec_layers = config["n_dec_layers"]
    dropout = config.get("dropout", 0)
    share_emb_out_proj = config.get("share_emb_out_proj", False)
    embedding_size = config["embedding_size"]
    
    
    rnnlm = RNNLM(config["symbol2id"],
                  trg_vocab_size,
                  embedding_size,
                  trg_max_len,
                  dec_hidden_size, 
                  n_dec_layers, 
                  cell_type, 
                  share_emb_out_proj=share_emb_out_proj, 
                  dropout=dropout)
    
    return rnnlm


def build_lm_model(config):
    """
    """
    if config["model"] == "transformer":
        model = build_transformer_lm_model(config)
    elif config["model"] == "rnn":
        model = build_rnn_lm_model(config)
    else:
        raise ValueError("model not correct!")
        
    return model


def build_transformer_bi_lm_model(config):
    """
    """
    trg_vocab_size = config["trg_vocab_size"]
    trg_max_len = config["trg_max_len"]
    n_heads = config["n_heads"]
    d_model = config["d_model"]
    d_ff = config["d_ff"]
    d_qk = config.get("d_qk", d_model//n_heads) 
    d_v = config.get("d_v", d_model//n_heads) 
    n_enc_layers = config["n_enc_layers"]
    dropout = config.get("dropout", 0)
    attn_dropout = config.get("attn_dropout", 0)
    emb_dropout = config.get("emb_dropout", 0)
    ln_eps = config.get("ln_eps", 1e-5)
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
    
    transformer = TransformerBiLM(config["symbol2id"],
                                  trg_vocab_size, 
                                  trg_max_len, 
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
                                  share_emb_out_proj,
                                  embedding_size,
                                  share_layer_params,
                                  n_share_across_layers,
                                  use_pre_norm,
                                  activation,
                                  scale_embedding,
                                  norm_before_pred,
                                  norm_after_embedding,
                                  pos_need_train)
    
    return transformer


def build_bert_model(config):
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
    dropout = config.get("dropout", 0)
    ln_eps = config.get("ln_eps", 1e-5)
    n_types = config.get("n_types", 2)
    share_emb_out_proj = config.get("share_emb_out_proj", False)
    share_layer_params = config.get("share_layer_params", False)
    embedding_size = config.get("embedding_size", None)
    n_share_across_layers = config.get("n_share_across_layers", 1)
    use_pre_norm = config.get("use_pre_norm", False)
    activation = config.get("activation", "gelu")
    
    symbol2id = {}
    bert = Bert(symbol2id, 
                src_vocab_size, 
                src_max_len, 
                n_heads, 
                d_model, 
                d_ff, 
                d_qk,
                d_v, 
                n_enc_layers,
                n_types,
                dropout,
                ln_eps,
                embedding_size=embedding_size, 
                share_layer_params=share_layer_params, 
                n_share_across_layers=n_share_across_layers,
                use_pre_norm=use_pre_norm, 
                activation=activation,
                output_mlm=True,
                output_cls=False,
                share_emb_out_proj=share_emb_out_proj)

    return bert


def build_bi_lm_model(config):
    """
    """
    if config["model"] == "transformer":
        model = build_transformer_bi_lm_model(config)
    elif config["model"] == "bert":
        model = build_bert_model(config)
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
    dropout = config.get("dropout", 0)
    attn_dropout = config.get("attn_dropout", 0)
    emb_dropout = config.get("emb_dropout", 0)
    ln_eps = config.get("ln_eps", 1e-5)
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
                                     embedding_size,
                                     share_layer_params,
                                     n_share_across_layers,
                                     use_pre_norm,
                                     activation,
                                     scale_embedding,
                                     norm_before_pred,
                                     norm_after_embedding,
                                     pos_need_train)
    
    return transformer


def build_match_text_model(config):
    """
    """
    if config["model"] == "transformer":
        model = build_transformer_encoder_model(config)
    else:
        raise ValueError("model not correct!")
        
    return model


def build_transformer_classify_model(config):
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
    dropout = config.get("dropout", 0)
    attn_dropout = config.get("attn_dropout", 0)
    emb_dropout = config.get("emb_dropout", 0)
    ln_eps = config.get("ln_eps", 1e-5)
    n_class = config["n_class"]
    share_layer_params = config.get("share_layer_params", False)
    n_share_across_layers = config.get("n_share_across_layers", 1)
    embedding_size = config.get("embedding_size", None)
    use_pre_norm = config.get("use_pre_norm", True)
    activation = config.get("activation", "relu")
    scale_embedding = config.get("scale_embedding", False)
    norm_before_pred = config.get("norm_before_pred", False)
    norm_after_embedding = config.get("norm_after_embedding", False)
    pos_need_train = config.get("pos_need_train", False)
    
    transformer = TransformerClassifer(config["symbol2id"], 
                                       src_vocab_size, 
                                       src_max_len, 
                                       n_heads, 
                                       d_model, 
                                       d_ff, 
                                       d_qk, 
                                       d_v,
                                       n_enc_layers,                                        
                                       n_class,
                                       dropout, 
                                       attn_dropout,
                                       emb_dropout,
                                       ln_eps,
                                       embedding_size,
                                       share_layer_params, 
                                       n_share_across_layers,
                                       use_pre_norm,
                                       activation,
                                       scale_embedding,
                                       norm_before_pred,
                                       norm_after_embedding,
                                       pos_need_train)
    
    return transformer


def build_transformer_crf_model(config):
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
    dropout = config.get("dropout", 0)
    attn_dropout = config.get("attn_dropout", 0)
    emb_dropout = config.get("emb_dropout", 0)
    ln_eps = config.get("ln_eps", 1e-5)
    n_labels = config["n_labels"]
    share_layer_params = config.get("share_layer_params", False)
    n_share_across_layers = config.get("n_share_across_layers", 1)
    embedding_size = config.get("embedding_size", None)
    use_pre_norm = config.get("use_pre_norm", True)
    activation = config.get("activation", "relu")
    scale_embedding = config.get("scale_embedding", False)
    norm_before_pred = config.get("norm_before_pred", False)
    norm_after_embedding = config.get("norm_after_embedding", False)
    pos_need_train = config.get("pos_need_train", False)
    
    transformer = TransformerCRF(config["symbol2id"], 
                                 src_vocab_size, 
                                 src_max_len, 
                                 n_heads,
                                 d_model,
                                 d_ff, 
                                 d_qk,
                                 d_v,
                                 n_enc_layers,                                  
                                 n_labels,
                                 dropout, 
                                 attn_dropout,
                                 emb_dropout,
                                 ln_eps, 
                                 embedding_size,
                                 share_layer_params, 
                                 n_share_across_layers,
                                 use_pre_norm,
                                 activation,
                                 scale_embedding,
                                 norm_before_pred,
                                 norm_after_embedding,
                                 pos_need_train)
    
    return transformer


def build_transformer_seq_cls_model(config):
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
    dropout = config.get("dropout", 0)
    attn_dropout = config.get("attn_dropout", 0)
    emb_dropout = config.get("emb_dropout", 0)
    ln_eps = config.get("ln_eps", 1e-5)
    n_labels = config["n_labels"]
    share_layer_params = config.get("share_layer_params", False)
    n_share_across_layers = config.get("n_share_across_layers", 1)
    embedding_size = config.get("embedding_size", None)
    use_pre_norm = config.get("use_pre_norm", True)
    activation = config.get("activation", "relu")
    scale_embedding = config.get("scale_embedding", False)
    norm_before_pred = config.get("norm_before_pred", False)
    norm_after_embedding = config.get("norm_after_embedding", False)
    pos_need_train = config.get("pos_need_train", False)
    
    transformer = TransformerSeqCls(config["symbol2id"], 
                                    src_vocab_size, 
                                    src_max_len, 
                                    n_heads,
                                    d_model,
                                    d_ff, 
                                    d_qk,
                                    d_v,
                                    n_enc_layers,                                     
                                    n_labels,
                                    dropout, 
                                    attn_dropout,
                                    emb_dropout,
                                    ln_eps, 
                                    embedding_size,
                                    share_layer_params, 
                                    n_share_across_layers,
                                    use_pre_norm,
                                    activation,
                                    scale_embedding,
                                    norm_before_pred,
                                    norm_after_embedding,
                                    pos_need_train)
    
    return transformer


def build_text_cnn_model(config):
    """
    """
    vocab_size = config["src_vocab_size"]
    n_class = config["n_class"]
    emb_size = config["src_emb_size"]
    num_filters = config["n_filters"]
    
    text_cnn = CnnTextClassifier(config["symbol2id"], 
                                 vocab_size, n_class, emb_size, num_filters)
    
    
    return text_cnn


def build_classify_model(config):
    """
    """
    if config["model"] == "transformer":
        model = build_transformer_classify_model(config)
    elif config["model"] == "text_cnn":
        model = build_text_cnn_model(config)
    else:
        raise ValueError("model not correct!")
        
    return model


def build_sequence_labeling_model(config):
    """
    """
    if config["model"] == "transformer_crf":
        model = build_transformer_crf_model(config)
    elif config["model"] == "transformer":
        model = build_transformer_seq_cls_model(config)
    else:
        raise ValueError("model not correct!")
        
    return model    


model_builder_dict = {
            "enc_dec": build_enc_dec_model,
            "lm": build_lm_model,
            "classify": build_classify_model,
            "bi_lm": build_bi_lm_model,
            "sequence_labeling": build_sequence_labeling_model,
            "match": build_match_text_model,
        }


def build_model(config):
    """
    """
    if config["task"] in model_builder_dict:
        return model_builder_dict[config["task"]](config)
    else:
        raise ValueError("model not correct!")
