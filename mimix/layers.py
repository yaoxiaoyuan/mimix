# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:29:30 2019

@author: Xiaoyuan Yao
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
    
class CRF(nn.Module):
    """
    """
    def __init__(self, n_labels):
        """
        """
        super(CRF, self).__init__()
        self.n_labels = n_labels
        self.start_trans = nn.Parameter(torch.Tensor(n_labels))
        self.trans = nn.Parameter(torch.Tensor(n_labels, n_labels))
        self.end_trans = nn.Parameter(torch.Tensor(n_labels))
        self.reset_parameters()
        
    
    def reset_parameters(self):
        """
        """
        for weight in [self.start_trans, self.trans, self.end_trans]:
            weight.data.uniform_(-0.1, 0.1)


    def get_end_mask(self, mask):
        """
        """
        left_shift_mask = torch.cat([mask[:,1:], torch.zeros_like(mask[:,:1])], 1)
        end_mask = (mask > left_shift_mask).to(mask.dtype)
        return end_mask
    
    
    def get_normalizer(self, emission, mask=None, end_mask=None):
        """
        """
        #start -> first tag 
        #BxT
        scores = self.start_trans + emission[:,0]
            
        seq_len = emission.size(1)
        for i in range(1, seq_len):
            #BxT -> BxTx1
            next_scores = scores.unsqueeze(2) 
            
            #Bx1xT + TxT -> BxTxT
            next_scores = next_scores + self.trans
            
            #BxTxT -> BxT
            next_scores = torch.logsumexp(next_scores, 1)

            #BxT + BxT -> BxT
            next_scores = next_scores + emission[:, i]

            #add mask
            if mask is not None:
                _mask = mask[:, i].unsqueeze(1)
                next_scores = _mask * next_scores + (1 - _mask) * scores

                _mask = end_mask[:, i].unsqueeze(1)
                next_scores = next_scores + _mask * self.end_trans

            scores = next_scores
        
        scores = torch.logsumexp(scores, 1)

        return scores


    def get_path_score(self, emission, target, mask=None, end_mask=None):
        """
        """
        batch_size, seq_len = emission.size(0), emission.size(1)
        
        scores = self.start_trans[target[:, 0]] 
        scores += emission[torch.arange(0, batch_size), 0, target[:, 0]]
        
        for i in range(1, seq_len):
            next_scores = scores + self.trans[target[:, i-1], target[:, i]]
            next_scores = next_scores + emission[torch.arange(0, batch_size), i, target[:, i]]

            if mask is not None:
                _mask = mask[:,i]
                next_scores = _mask * next_scores + (1 - _mask) * scores
            if end_mask is not None:
                _mask = end_mask[:,i]
                next_scores = next_scores + _mask * self.end_trans[target[:, i]]

            scores = next_scores
        
        return scores
    
    
    def forward(self, emission, target, mask=None):
        """
        """
        if mask is not None:
            end_mask = self.get_end_mask(mask)
        
        path_scores = self.get_path_score(emission, target, mask, end_mask)
        
        normalizer = self.get_normalizer(emission, mask, end_mask)

        neg_log_likelihood = torch.mean(normalizer - path_scores)
        
        return neg_log_likelihood


class Embedding(nn.Module):
    """
    """
    def __init__(self, vocab_size, embedding_size, factorized_size=None):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.factorized_size = factorized_size
        if self.factorized_size is None:
            self.W = nn.Parameter(torch.Tensor(vocab_size, embedding_size))
        else:
            self.W = nn.Parameter(torch.Tensor(vocab_size, embedding_size))
            self.We = nn.Parameter(torch.Tensor(factorized_size, embedding_size))             
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        """
        #stdv = 1.0 / np.sqrt(self.embedding_size)
        stdv = np.log(self.vocab_size) / self.embedding_size
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        """
        if self.factorized_size is None:
            return self.W[x]
        return F.linear(self.W[x], self.We)


    def get_embedding(self):
        """
        """
        if self.factorized_size is None:
            return self.W
        return F.linear(self.W, self.We)


class PositionEmbedding(nn.Module):
    """
    """
    def __init__(self, max_len, d_model, pos_type="learned"):
        super(PositionEmbedding, self).__init__()
        self.max_len = max_len
        self.embedding_size = d_model
        self.pos_type = pos_type
        if pos_type == "sinusoidal":
            W = torch.zeros(max_len, d_model)
            for i in range(max_len):
                for j in range(0, d_model, 2):
                    W[i, j] = np.sin(i / np.power(10000, 2 * j / d_model))
                    W[i, j + 1] = np.cos(i / np.power(10000, 2 * j / d_model))
            self.register_buffer('W', W)
        elif pos_type == "learned":
            self.W = nn.Parameter(torch.Tensor(max_len, d_model))
            self.reset_parameters()
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
    def forward(self, pos_ids):
        """
        """
        pos_ids[pos_ids >=self.max_len] = -1 
        if self.pos_type == "sinusoidal":
            pe = Variable(self.W[pos_ids], requires_grad=False)
            return pe
        elif self.pos_type == "learned":
            return self.W[pos_ids]


class RelativePositionEmbedding(nn.Module):
    """
    """
    def __init__(self, max_relative_len, d_model, pos_type="learned"):
        """
        """
        super(RelativePositionEmbedding, self).__init__()
        self.embedding_size = d_model
        self.max_relative_len = max_relative_len
        self.pos_type = pos_type
        if pos_type == "sinusoidal":
            W = torch.zeros(2*max_relative_len+1, d_model)
            for i in range(2*max_relative_len+1):
                for j in range(0, d_model, 2):
                    W[i, j] = np.sin(i / np.power(10000, 2 * j / d_model))
                    W[i, j + 1] = np.cos(i / np.power(10000, 2 * j / d_model))
            self.register_buffer('W', W)
        elif pos_type == "learned":
            self.W = nn.Parameter(torch.Tensor(2*max_relative_len+1, d_model))
            self.reset_parameters()
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
    def forward(self, relative_dis):
        """
        """
        relative_dis = torch.clamp(relative_dis, -self.max_relative_len, self.max_relative_len)
        idx = relative_dis + self.max_relative_len
        if self.pos_type == "sinusoidal":
            pe = Variable(self.W[idx], 
                          requires_grad=False)
            return pe
        elif self.pos_type == "learned":
            return self.W[idx]


class Dropout(nn.Module):
    def __init__(self, p=0):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability incorrect!")
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0:
            mask = torch.rand_like(x, device = x.device) < self.p
            x = torch.masked_fill(x, mask, 0)
            scale = (1.0/(1-self.p))
            return x * scale
        return x


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

 
act2fn = {"relu": F.relu, "gelu":F.gelu, "gelu_new":gelu_new}


class FeedForward(nn.Module):
    """
    """
    def __init__(self, d_model, d_ff, activation="relu", dropout=0, use_bias=True, use_glu=False):
        """
        """
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.dropout = None
        if dropout > 0:
            self.dropout = Dropout(dropout)
        self.use_glu = use_glu
        self.W1 = nn.Parameter(torch.Tensor(self.d_ff, self.d_model))
        self.use_bias = use_bias
        if self.use_bias == True:
            self.b1 = nn.Parameter(torch.Tensor(self.d_ff))
        if self.use_glu == False:
            self.W2 = nn.Parameter(torch.Tensor(self.d_model, self.d_ff))
            if self.use_bias == True:
                self.b2 = nn.Parameter(torch.Tensor(self.d_model))
        else:
            self.W2 = nn.Parameter(torch.Tensor(self.d_ff, self.d_model))
            if self.use_bias == True:
                self.b2 = nn.Parameter(torch.Tensor(self.d_ff))            
        if self.use_glu == True:
            self.W3 = nn.Parameter(torch.Tensor(self.d_model, self.d_ff))
            self.b3 = nn.Parameter(torch.Tensor(self.d_model))
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.d_model)
        w_list = [self.W1, self.W2]
        if self.use_glu:
            w_list.append(self.W3)
        for weight in w_list:
            weight.data.uniform_(-stdv, stdv)
        if self.use_bias == True:
            b_list = [self.b1, self.b2]
            if self.use_glu == True:
                b_list = b_list + [self.b3]
            for weight in b_list:
                weight.data.fill_(0)
            

    def forward(self, x):
        """
        """
        act_fn = act2fn[self.activation]
        if self.use_glu == False:
            if self.use_bias == True:
                x = F.linear(act_fn(F.linear(x, self.W1) + self.b1), self.W2) + self.b2
            else:
                x = F.linear(act_fn(F.linear(x, self.W1)), self.W2)
        else:
            if self.use_bias == True:
                x = F.linear(act_fn(F.linear(x, self.W1) + self.b1) * (F.linear(x, self.W2) + self.b2), self.W3) + self.b3
            else:
                x = F.linear(act_fn(F.linear(x, self.W1)) * F.linear(x, self.W2), self.W3)
            
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    """
    """
    def __init__(self, d_model, eps=1e-5, use_rms_norm=False, use_bias=True, use_scale=True):
        """
        """
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.d_model = d_model
        self.use_scale = use_scale
        if use_scale == True:
            self.alpha = nn.Parameter(torch.ones(self.d_model))
        self.use_bias = use_bias
        if use_bias == True:
            self.bias = nn.Parameter(torch.zeros(self.d_model))
        self.use_rms_norm = use_rms_norm
        
    def forward(self, x):
        """
        """
        if self.use_rms_norm == True:
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            norm = x * rms      
        else:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, unbiased=False, keepdim=True)
            norm = (x - mean) / (std + self.eps)
            
        if self.use_scale == True:
            norm = self.alpha * norm
        if self.use_bias == True:
            norm = norm + self.bias  
                
        return norm


def scaled_dot_product_attention(query, 
                                 key, 
                                 value, 
                                 attn_mask=None, 
                                 dropout=None, 
                                 pos_key=None, 
                                 pos_value=None,
                                 alibi_bias=None,
                                 attention_residual=None,
                                 talk_w=None):
    """
    """
    d = query.size(-1)
    #q:B x L_q x n_heads x d_qk 
    #k:B x L_kv x n_heads x d_v 
    #scores:B x n_heads x L_q x L_kv
    scores = torch.einsum("bqnd,bknd->bnqk", query, key)
    
    if pos_key is not None:
        #p_k:L_q x L_k x d_qk
        scores += torch.einsum("bqnd,bqkd->bnqk", query, pos_key)
        
    if alibi_bias is not None:        
        scores += alibi_bias

    if talk_w is not None:
        scores = torch.einsum("bnqk,nm->bmqk", scores, talk_w)

    if attention_residual is not None:
        scores += attention_residual
        
    scores = scores / np.sqrt(d)
    
    if attn_mask is not None:
        attn_mask = attn_mask.bool()
        scores = scores.masked_fill(attn_mask, -1e4)
    
    attn_scores = F.softmax(scores, dim = -1)
    
    if dropout is not None:
        attn_scores = dropout(attn_scores)

    #scores:B x n_heads x L_q x L_kv
    #v:B x L_kv x n_heads x d_v 
    output = torch.einsum("bnqk,bknd->bnqd", attn_scores, value)
    if pos_value is not None:
        #p_v:L_q x L_kv x d_v
        output += torch.einsum("bnqk,bqkd->bnqd", attn_scores, pos_value)
    
    return output, attn_scores, scores
            

class MultiHeadAttention(nn.Module):
    """
    """
    def __init__(self, 
                 n_heads, 
                 d_model, 
                 d_qk, 
                 d_v, 
                 dropout=0, 
                 attn_dropout=0, 
                 use_bias=True, 
                 max_relative_len=-1,
                 use_rel_pos_value=False,
                 rel_pos_type="learned",
                 use_multi_query_attention=False,
                 use_alibi_bias=False,
                 use_talking_attention=False):
        """
        """
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = None
        if dropout > 0:
            self.dropout = Dropout(dropout)
        self.attn_dropout = None
        if attn_dropout > 0:
            self.attn_dropout = Dropout(dropout)
        self.d_qk = d_qk
        self.d_v = d_v
        self.max_relative_len = max_relative_len
        self.W_q = nn.Parameter(torch.Tensor(n_heads*d_qk, d_model))
        self.use_multi_query_attention = use_multi_query_attention
        if use_multi_query_attention == True:
            self.W_k = nn.Parameter(torch.Tensor(d_qk, d_model))
            self.W_v = nn.Parameter(torch.Tensor(d_v, d_model))            
        else:
            self.W_k = nn.Parameter(torch.Tensor(n_heads*d_qk, d_model))
            self.W_v = nn.Parameter(torch.Tensor(n_heads*d_v, d_model))
        self.W_o = nn.Parameter(torch.Tensor(d_model, n_heads*d_v))
        self.use_bias = use_bias
        if self.use_bias == True:
            self.b_q = nn.Parameter(torch.Tensor(n_heads*d_qk))
            if use_multi_query_attention == True:
                self.b_k = nn.Parameter(torch.Tensor(d_qk))
                self.b_v = nn.Parameter(torch.Tensor(d_v))                
            else:
                self.b_k = nn.Parameter(torch.Tensor(n_heads*d_qk))
                self.b_v = nn.Parameter(torch.Tensor(n_heads*d_v))
            self.b_o = nn.Parameter(torch.Tensor(d_model))
        
        self.rel_pos_k_emb = None
        self.rel_pos_v_emb = None
        self.use_rel_pos_value = use_rel_pos_value
        if max_relative_len > 0:
            self.rel_pos_k_emb = RelativePositionEmbedding(max_relative_len, self.d_qk, rel_pos_type)
            if use_rel_pos_value == True:
                self.rel_pos_v_emb = RelativePositionEmbedding(max_relative_len, self.d_v, rel_pos_type)
        
        self.use_alibi_bias = use_alibi_bias
        
        self.talking_w = None
        self.use_talking_attention = use_talking_attention
        if use_talking_attention == True:
            self.talking_w = nn.Parameter(torch.Tensor(n_heads, n_heads))
        
        self.reset_parameters()
        
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.d_model)
        for weight in [self.W_q, self.W_k, self.W_v, self.W_o]:
            weight.data.uniform_(-stdv, stdv)
        if self.use_bias == True:
            for weight in [self.b_q, self.b_k, self.b_v, self.b_o]:
                weight.data.zero_()
        if self.use_talking_attention == True:
            stdv = 1.0 / np.sqrt(self.n_heads)
            self.talking_w.data.uniform_(-stdv, stdv)
    
    def forward(self, 
                query, 
                key, 
                value, 
                attn_mask=None, 
                cached_kv=False, 
                q_pos_ids=None, 
                kv_pos_ids=None,
                attention_residual=None):
        """
        """
        #B x L x d_model -> B x l x (d*n_heads)
        query = F.linear(query, self.W_q)
        if self.use_bias == True:
            query = query + self.b_q
        if cached_kv == False:
            key = F.linear(key, self.W_k) 
            value = F.linear(value, self.W_v) 
            if self.use_bias == True:
                key = key + self.b_k
                value = value + self.b_v

        batch_size = query.size(0)
        #B x l x (d*n_heads) -> B x L x n_heads x d_qk
        query = query.view(batch_size, -1, self.n_heads, self.d_qk)
        if cached_kv == False:
            if self.use_multi_query_attention == True:
                key = key.view(batch_size, -1, 1, self.d_qk)
                value = value.view(batch_size, -1, 1, self.d_v)            
            else:
                key = key.view(batch_size, -1, self.n_heads, self.d_qk)
                value = value.view(batch_size, -1, self.n_heads, self.d_v)
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        
        relative_dis = None
        if self.max_relative_len > 0 or self.use_alibi_bias == True:
            relative_dis = q_pos_ids[:,:,None] - kv_pos_ids[:,None,:]
        
        pos_key = None
        pos_value = None
        if self.max_relative_len > 0:
            pos_key = self.rel_pos_k_emb(relative_dis=relative_dis)
            if self.use_rel_pos_value == True:
                pos_value = self.rel_pos_k_emb(relative_dis=relative_dis)
        
        alibi_bias = None
        if self.use_alibi_bias == True:
            start = (2**(-2**-(math.log2(self.n_heads)-3)))
            ratio = start
            #slopes: n_heads
            #relative_dis : B x L_q x L_k
            #alibi_bias: B x n_heads x L_q x L_k
            slopes = torch.tensor([start*ratio**i for i in range(self.n_heads)]).to(query.device, dtype=query.dtype)
            relative_dis[relative_dis<0] = -relative_dis[relative_dis<0]
            alibi_bias = torch.einsum("bqk,n->bnqk", relative_dis, slopes)
            
        output, attn_scores,scores = scaled_dot_product_attention(query, 
                                                                  key, 
                                                                  value, 
                                                                  attn_mask,
                                                                  self.attn_dropout,
                                                                  pos_key,
                                                                  pos_value,
                                                                  alibi_bias,
                                                                  attention_residual,
                                                                  self.talking_w)
        
        #B x n_heads x L x d -> B x L x n_heads x d -> B x L x d_model
        output = output.transpose(1,2)
        output = output.contiguous().view(batch_size, -1, 
                                  self.n_heads*self.d_v)
        output = F.linear(output, self.W_o)
        if self.use_bias == True:
            output = output + self.b_o
            
        if self.attn_dropout is not None:
            output = self.dropout(output)
            
        return output, attn_scores, scores


    def cache_kv(self, x):
        """
        """
        batch_size = x.size(0)
        
        key = F.linear(x, self.W_k)
        value = F.linear(x, self.W_v) 
        if self.use_bias == True:
            key = key + self.b_k
            value = value + self.b_v
        
        if self.use_multi_query_attention == True:
            key = key.view(batch_size, -1, 1, self.d_qk)
            value = value.view(batch_size, -1, 1, self.d_v)            
        else:   
            key = key.view(batch_size, -1, self.n_heads, self.d_qk)
            value = value.view(batch_size, -1, self.n_heads, self.d_v)
        
        return [key, value]


class TransformerLayer(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        """
        """
        super(TransformerLayer, self).__init__()
        
        self.n_heads = kwargs["n_heads"]
        self.d_model = kwargs["d_model"]
        self.d_ff = kwargs.get("d_ff", 4 * self.d_model) 
        self.d_qk = kwargs.get("d_qk", self.d_model//self.n_heads)
        self.d_v = kwargs.get("d_v", self.d_model//self.n_heads)
        self.dropout = kwargs.get("dropout", 0)
        self.attn_dropout = kwargs.get("attn_dropout", 0)
        self.ln_eps = kwargs.get("ln_eps", 1e-5)
        self.use_pre_norm = kwargs.get("use_pre_norm", False)
        self.activation = kwargs.get("activation", "relu")
        self.use_rms_norm = kwargs.get("use_rms_norm", False)
        self.use_attention_bias = kwargs.get("use_attention_bias", True)
        self.use_ffn_bias = kwargs.get("use_ffn_bias", True)
        self.use_multi_query_attention = kwargs.get("use_multi_query_attention", False)
        self.use_alibi_bias = kwargs.get("use_alibi_bias", False)
        self.max_relative_len = kwargs.get("max_relative_len", -1)
        self.use_rel_pos_value = kwargs.get("use_rel_pos_value", False)
        self.rel_pos_type = kwargs.get("rel_pos_type", "learned")
        self.with_across_attention = kwargs.get("with_across_attention", False)
        self.use_talking_attention = kwargs.get("use_talking_attention", False)
        self.use_glu = kwargs.get("use_glu", False)
        self.use_ln_scale = kwargs.get("use_ln_scale", True)
        self.use_ln_bias = kwargs.get("use_ln_bias", True)
        
        self.self_attention = MultiHeadAttention(self.n_heads,
                                                 self.d_model, 
                                                 self.d_qk, 
                                                 self.d_v, 
                                                 self.dropout,
                                                 self.attn_dropout,
                                                 self.use_attention_bias,
                                                 self.max_relative_len,
                                                 self.use_rel_pos_value,
                                                 self.rel_pos_type,
                                                 self.use_multi_query_attention,
                                                 self.use_alibi_bias,
                                                 self.use_talking_attention)
        
        self.norm_1 = LayerNorm(self.d_model,self.ln_eps,self.use_rms_norm,self.use_ln_scale,self.use_ln_bias)
        
        if self.with_across_attention == True:
            self.enc_attention = MultiHeadAttention(self.n_heads,
                                                    self.d_model, 
                                                    self.d_qk, 
                                                    self.d_v, 
                                                    self.dropout,
                                                    self.attn_dropout,
                                                    self.use_attention_bias,
                                                    self.max_relative_len,
                                                    self.use_rel_pos_value,
                                                    self.rel_pos_type,
                                                    self.use_multi_query_attention,
                                                    False,
                                                    self.use_talking_attention)
        
            self.norm_2 = LayerNorm(self.d_model,self.ln_eps,self.use_rms_norm,self.use_ln_scale,self.use_ln_bias)
            
        self.ffn = FeedForward(self.d_model, 
                               self.d_ff, 
                               self.activation, 
                               self.dropout,
                               self.use_ffn_bias,
                               self.use_glu)
        if self.with_across_attention == True:
            self.norm_3 = LayerNorm(self.d_model,self.ln_eps,self.use_rms_norm,self.use_ln_scale,self.use_ln_bias)
        else:
            self.norm_2 = LayerNorm(self.d_model,self.ln_eps,self.use_rms_norm,self.use_ln_scale,self.use_ln_bias)
        
        
    def forward(self, 
                output,  
                self_attn_mask, 
                cached_kv=False, 
                self_keys=None, 
                self_values=None,
                enc_keys=None, 
                enc_values=None,
                dec_enc_attn_mask=None,
                self_pos_ids=None,
                enc_pos_ids=None,
                past_pos_ids=None,
                attention_residual=None,
                enc_attention_residual=None):
        """
        """
        residual = output
        if self.use_pre_norm == True:
            output = self.norm_1(output)
        
        if cached_kv == True:
            kv = self.cache_dec_kv(output)
            
            if self_keys is None:
                self_keys = kv[0]                
            else:
                self_keys = torch.cat([self_keys, kv[0]], 1)
            if self_values is None:
                self_values = kv[1]
            else:
                self_values = torch.cat([self_values, kv[1]], 1)
                
        else:
            self_keys = output
            self_values = output
        
        output, self_attn_scores, self_scores = self.self_attention(output, 
                                                                    self_keys, 
                                                                    self_values, 
                                                                    self_attn_mask,
                                                                    cached_kv,
                                                                    self_pos_ids,
                                                                    past_pos_ids,
                                                                    attention_residual)
        
        output = residual + output
        if self.use_pre_norm == False:
            output = self.norm_1(output)
        
        residual = output
        
        if self.with_across_attention == True:
            if self.use_pre_norm == True:
                output = self.norm_2(output)
            
            output, enc_attn_scores, enc_scores = self.enc_attention(output, 
                                                                     enc_keys, 
                                                                     enc_values, 
                                                                     dec_enc_attn_mask,
                                                                     cached_kv,
                                                                     self_pos_ids,
                                                                     enc_pos_ids,
                                                                     enc_attention_residual)

            output = residual + output
            if self.use_pre_norm == False:
                output = self.norm_2(output)
            
            residual = output
                
        if self.use_pre_norm == True:
            if self.with_across_attention == True:
                output = self.norm_3(output)
            else:
                output = self.norm_2(output)
        output = self.ffn(output)
        output = residual + output
        if self.use_pre_norm == False:
            if self.with_across_attention == True:
                output = self.norm_3(output)
            else:
                output = self.norm_2(output)
   
        outputs = [output, self_attn_scores, self_scores]
        if self.with_across_attention == True:
            outputs += [enc_attn_scores, enc_scores]

        if cached_kv == True:
            outputs = outputs + [self_keys, self_values]
        
        return outputs

    def cache_enc_kv(self, enc_output):
        """
        """
        return self.enc_attention.cache_kv(enc_output)


    def cache_dec_kv(self, dec_output):
        """
        """
        return self.self_attention.cache_kv(dec_output)


