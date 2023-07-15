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
        end_mask = (mask > left_shift_mask).float() 
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
    def __init__(self, vocab_size, embedding_size, scale_embedding=False):
        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.W = nn.Parameter(torch.Tensor(vocab_size, embedding_size))
        self.scale_embedding = scale_embedding
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, x):
        """
        """
        embeded = self.W[x]
        if self.scale_embedding == True:
            embeded = embeded * np.sqrt(self.embedding_size)
        return embeded


    def get_embedding(self):
        """
        """
        return self.W


class FactorizedEmbedding(nn.Module):
    """
    """
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(FactorizedEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.W = nn.Parameter(torch.Tensor(vocab_size, embedding_size))
        self.We = nn.Parameter(torch.Tensor(hidden_size, embedding_size))
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, x):
        """
        """
        return F.linear(self.W[x], self.We)


    def get_embedding(self):
        """
        """
        return F.linear(self.W, self.We)


class PositionEmbedding(nn.Module):
    """
    """
    def __init__(self, max_len, d_model, need_train=False):
        super(PositionEmbedding, self).__init__()
        self.max_len = max_len
        self.embedding_size = d_model
        self.need_train = need_train
        if need_train == False:
            W = torch.zeros(max_len, d_model)
            for i in range(max_len):
                for j in range(0, d_model, 2):
                    W[i, j] = np.sin(i / np.power(10000, 2 * j / d_model))
                    W[i, j + 1] = np.cos(i / np.power(10000, 2 * j / d_model))
            self.register_buffer('W', W)
        else:
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
        if self.need_train == False:
            pe = Variable(self.W[pos_ids], requires_grad=False)
            return pe
        else:
            return self.W[pos_ids]


class Dropout(nn.Module):
    def __init__(self, p=0):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability incorrect!")
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0:
            mask = torch.rand_like(x, device = x.device) > self.p
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


class FeedForward(nn.Module):
    """
    """
    def __init__(self, d_model, d_ff, activation="relu", dropout=0, use_bias=True):
        """
        """
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.dropout = None
        if dropout > 0:
            self.dropout = Dropout(dropout)
        self.W1 = nn.Parameter(torch.Tensor(self.d_ff, self.d_model))
        self.use_bias = use_bias
        if self.use_bias == True:
            self.b1 = nn.Parameter(torch.Tensor(self.d_ff))
        self.W2 = nn.Parameter(torch.Tensor(self.d_model, self.d_ff))
        if self.use_bias == True:
            self.b2 = nn.Parameter(torch.Tensor(self.d_model))
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.d_model)
        for weight in [self.W1, self.W2]:
            weight.data.uniform_(-stdv, stdv)
        if self.use_bias == True:
            for weight in [self.b1, self.b2]:
                weight.data.fill_(0)
            

    def forward(self, x):
        """
        """
        if self.activation == "relu":
            if self.use_bias == True:
                x = F.linear(F.relu(F.linear(x, self.W1) + self.b1), self.W2) + self.b2
            else:
                x = F.linear(F.relu(F.linear(x, self.W1)), self.W2)
        elif self.activation == "gelu" or self.activation == "gelu_new":
            if self.use_bias == True:
                x = F.linear(gelu_new(F.linear(x, self.W1) + self.b1), self.W2) + self.b2
            else:
                x = F.linear(F.relu(F.linear(x, self.W1)), self.W2)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    """
    """
    def __init__(self, d_model, eps=1e-5, use_rms_norm=False):
        """
        """
        super(LayerNorm, self).__init__()
        
        self.eps = eps
        self.d_model = d_model
        self.alpha = nn.Parameter(torch.ones(self.d_model))
        self.bias = nn.Parameter(torch.zeros(self.d_model))
        self.use_rms_norm = use_rms_norm
        
    def forward(self, x):
        """
        """
        if self.use_rms_norm == True:
            std = x.std(dim=-1, unbiased=False, keepdim=True)
            norm = self.alpha * x / (std + self.eps) + self.bias        
        else:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, unbiased=False, keepdim=True)
            norm = self.alpha * (x - mean) / (std + self.eps) + self.bias
        return norm


def scaled_dot_product_attention(query, 
                                 key, 
                                 value, 
                                 attn_mask=None, 
                                 dropout=None, 
                                 pos_key=None, 
                                 pos_value=None,
                                 alibi_bias=None):
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
        
    scores = scores / np.sqrt(d)

    if alibi_bias is not None:        
        scores += alibi_bias
    
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
    
    return output, attn_scores


class RelativePositionEmbedding(nn.Module):
    """
    """
    def __init__(self, max_relative_len, d_model, need_train=True):
        """
        """
        super(RelativePositionEmbedding, self).__init__()
        self.embedding_size = d_model
        self.max_relative_len = max_relative_len
        self.need_train = need_train
        if need_train == False:
            W = torch.zeros(2*max_relative_len+1, d_model)
            for i in range(2*max_relative_len+1):
                for j in range(0, d_model, 2):
                    W[i, j] = np.sin(i / np.power(10000, 2 * j / d_model))
                    W[i, j + 1] = np.cos(i / np.power(10000, 2 * j / d_model))
            self.register_buffer('W', W)
        else:
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
        if self.need_train == False:
            pe = Variable(self.W[idx], 
                          requires_grad=False)
            return pe
        else:
            return self.W[idx]
            

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
                 rel_pos_need_train=True,
                 use_multi_query_attention=False,
                 use_alibi_bias=False):
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
            self.rel_pos_k_emb = RelativePositionEmbedding(max_relative_len, self.d_qk, rel_pos_need_train)
            if use_rel_pos_value == True:
                self.rel_pos_v_emb = RelativePositionEmbedding(max_relative_len, self.d_v, rel_pos_need_train)
        
        self.use_alibi_bias = use_alibi_bias
        
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
    
    
    def forward(self, query, key, value, attn_mask=None, cached_kv=False, q_pos_ids=None, kv_pos_ids=None):
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
            slopes = torch.tensor([start*ratio**i for i in range(self.n_heads)]).to(query.device)
            relative_dis[relative_dis<0] = -relative_dis[relative_dis<0]
            alibi_bias = torch.einsum("bqk,n->bnqk", relative_dis, slopes)
            
        output, attn_scores = scaled_dot_product_attention(query, 
                                                           key, 
                                                           value, 
                                                           attn_mask,
                                                           self.attn_dropout,
                                                           pos_key,
                                                           pos_value,
                                                           alibi_bias)
        
        #B x n_heads x L x d -> B x L x n_heads x d -> B x L x d_model
        output = output.transpose(1,2)
        output = output.contiguous().view(batch_size, -1, 
                                  self.n_heads*self.d_v)
        output = F.linear(output, self.W_o)
        if self.use_bias == True:
            output = output + self.b_o
            
        if self.attn_dropout is not None:
            output = self.dropout(output)
            
        return output, attn_scores


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
    def __init__(self, 
                 n_heads,
                 d_model, 
                 d_ff, 
                 d_qk, 
                 d_v, 
                 dropout=0,
                 attn_dropout=0,
                 ln_eps=1e-5,
                 use_pre_norm=True,
                 activation="relu",
                 use_rms_norm=False,
                 use_attention_bias=True,
                 use_ffn_bias=True,
                 use_multi_query_attention=False,
                 use_alibi_bias=False,
                 max_relative_len=-1,
                 use_rel_pos_value=False,
                 rel_pos_need_train=True,
                 with_across_attention=True):
        """
        """
        super(TransformerLayer, self).__init__()
        self.self_attention = MultiHeadAttention(n_heads,
                                                 d_model, 
                                                 d_qk, 
                                                 d_v, 
                                                 dropout,
                                                 attn_dropout,
                                                 use_attention_bias,
                                                 max_relative_len,
                                                 use_rel_pos_value,
                                                 rel_pos_need_train,
                                                 use_multi_query_attention,
                                                 use_alibi_bias)
        
        self.norm_1 = LayerNorm(d_model,ln_eps,use_rms_norm)
        
        self.with_across_attention = with_across_attention
        if self.with_across_attention == True:
            self.enc_attention = MultiHeadAttention(n_heads,
                                                    d_model, 
                                                    d_qk, 
                                                    d_v, 
                                                    dropout,
                                                    attn_dropout,
                                                    use_attention_bias,
                                                    max_relative_len,
                                                    use_rel_pos_value,
                                                    rel_pos_need_train,
                                                    use_multi_query_attention,
                                                    False)
        
            self.norm_2 = LayerNorm(d_model,ln_eps,use_rms_norm)
            
        self.ffn = FeedForward(d_model, 
                               d_ff, 
                               activation, 
                               dropout,
                               use_ffn_bias)
        if self.with_across_attention == True:
            self.norm_3 = LayerNorm(d_model,ln_eps,use_rms_norm)
        else:
            self.norm_2 = LayerNorm(d_model,ln_eps,use_rms_norm)
        self.use_pre_norm = use_pre_norm
        
        
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
                past_pos_ids=None):
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

        output, self_attn_scores = self.self_attention(output, 
                                                       self_keys, 
                                                       self_values, 
                                                       self_attn_mask,
                                                       cached_kv,
                                                       self_pos_ids,
                                                       past_pos_ids)
        
        output = residual + output
        if self.use_pre_norm == False:
            output = self.norm_1(output)
            
        residual = output
        
        if self.with_across_attention == True:
            if self.use_pre_norm == True:
                output = self.norm_2(output)
            
            output, enc_attn_scores = self.enc_attention(output, 
                                                         enc_keys, 
                                                         enc_values, 
                                                         dec_enc_attn_mask,
                                                         cached_kv,
                                                         self_pos_ids,
                                                         enc_pos_ids)

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
       
        outputs = [output, self_attn_scores]
        if self.with_across_attention == True:
            outputs += [enc_attn_scores]

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


class Encoder(nn.Module):
    """
    """
    def __init__(self, 
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
                 use_multi_query_attention=False,
                 use_alibi_bias=False,
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
                 use_pos_embeding=True,
                 pos_need_train=False,
                 n_types=None):
        """
        """
        super(Encoder, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers
        self.share_layer_params = share_layer_params
        self.n_share_across_layers = n_share_across_layers
        self.scale_embedding = scale_embedding
        self.norm_before_pred = norm_before_pred
        self.norm_after_embedding = norm_after_embedding
        
        if embedding_size is None:
            self.src_embedding = Embedding(src_vocab_size, 
                                           d_model, 
                                           scale_embedding)
        else:
            self.src_embedding = FactorizedEmbedding(src_vocab_size, 
                                                     embedding_size,
                                                     d_model)

        self.use_pos_embeding = use_pos_embeding
        if use_pos_embeding:
            self.pos_embedding = PositionEmbedding(src_max_len, 
                                                   d_model, 
                                                   pos_need_train)
        
        self.n_types = n_types
        if self.n_types is not None:
            self.type_embedding = Embedding(self.n_types, self.d_model)
        
        if self.norm_after_embedding == True:
            self.norm_emb = LayerNorm(self.d_model, ln_eps, use_rms_norm)
        
        self.emb_dropout = None
        if emb_dropout > 0:
            self.emb_dropout = Dropout(dropout)
            
        self.layers = nn.ModuleList([
                TransformerLayer(n_heads, 
                                 d_model, 
                                 d_ff, 
                                 d_qk, 
                                 d_v,
                                 dropout=dropout,
                                 attn_dropout=attn_dropout,
                                 ln_eps=ln_eps,
                                 use_pre_norm=use_pre_norm,
                                 activation=activation,
                                 use_rms_norm=use_rms_norm,
                                 use_attention_bias=use_attention_bias,
                                 use_ffn_bias=use_ffn_bias,
                                 use_multi_query_attention=use_multi_query_attention,
                                 use_alibi_bias=use_alibi_bias,
                                 max_relative_len=max_relative_len,
                                 use_rel_pos_value=use_rel_pos_value,
                                 rel_pos_need_train=rel_pos_need_train,
                                 with_across_attention=False)
                    for i in range(n_layers//n_share_across_layers)])
    
        if self.norm_before_pred == True:
            self.norm = LayerNorm(self.d_model, ln_eps, use_rms_norm)
            
    
    def forward(self, x, attn_mask, self_pos_ids=None, x_type=None, return_states=False):
        """
        """
        enc_self_attn_list = []
        
        word_embeded = self.src_embedding(x)
        
        embeded = word_embeded
        if self.use_pos_embeding == True:
           embeded += self.pos_embedding(self_pos_ids)
        
        if self.n_types is not None:
            embeded += self.type_embedding(x_type)
        enc_output = embeded
           
        if self.norm_after_embedding == True:
            enc_output = self.norm_emb(enc_output)
        
        if self.emb_dropout is not None:
            enc_output = self.emb_dropout(enc_output)
        
        enc_states = [enc_output]
        
        for i in range(self.n_layers):
            if self.share_layer_params == False:
                enc_layer = self.layers[i]
            else:
                enc_layer = self.layers[i // self.n_share_across_layers]
            
            enc_output, enc_attn_scores = enc_layer(enc_output, 
                                                    attn_mask, 
                                                    self_pos_ids=self_pos_ids, 
                                                    past_pos_ids=self_pos_ids)
            
            enc_self_attn_list.append(enc_attn_scores)
            enc_states.append(enc_output)
        
        if self.norm_before_pred == True:
            enc_output = self.norm(enc_output)

        outputs = [enc_output]
                
        if return_states == True:
            outputs = outputs + [embeded, enc_states, enc_self_attn_list]
        
        return outputs


class Decoder(nn.Module):
    """
    """
    def __init__(self, 
                 trg_vocab_size, 
                 trg_max_len, 
                 n_heads, 
                 d_model, 
                 d_ff, 
                 d_qk,
                 d_v, 
                 n_layers, 
                 with_across_attention=True,
                 dropout=0, 
                 attn_dropout=0,
                 emb_dropout=0,
                 ln_eps=1e-5,
                 use_rms_norm=False,
                 use_attention_bias=True,
                 use_ffn_bias=True,
                 use_multi_query_attention=False,
                 use_alibi_bias=False,
                 max_relative_len=-1,
                 use_rel_pos_value=False,
                 rel_pos_need_train=True,
                 embedding_size=None,
                 share_layer_params=False, 
                 n_share_across_layers=1,
                 share_src_trg_emb=False, 
                 share_emb_out_proj=False,
                 use_pre_norm=True, 
                 activation="relu", 
                 scale_embedding=False,
                 norm_before_pred=False,
                 norm_after_embedding=False,
                 use_pos_embedding=True,
                 pos_need_train=False,
                 use_output_bias=False):
        """
        """
        super(Decoder, self).__init__()
        self.trg_vocab_size = trg_vocab_size
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers
        self.with_across_attention = with_across_attention
        self.share_layer_params = share_layer_params
        self.n_share_across_layers = n_share_across_layers
        self.scale_embedding = scale_embedding
        self.norm_before_pred = norm_before_pred
        self.norm_after_embedding = norm_after_embedding
        
        if share_src_trg_emb == False:
            self.trg_embedding = Embedding(trg_vocab_size, 
                                           d_model, 
                                           scale_embedding)
            
        self.emb_dropout = None
        if emb_dropout > 0:
            self.emb_dropout = Dropout(dropout)
        
        self.use_pos_embedding = use_pos_embedding
        if use_pos_embedding == True:    
            self.pos_embedding = PositionEmbedding(trg_max_len, 
                                                   d_model, 
                                                   pos_need_train)
        
        if self.norm_after_embedding == True:
            self.norm_emb = LayerNorm(self.d_model, ln_eps, use_rms_norm)
        
        self.layers = nn.ModuleList([
                TransformerLayer(n_heads, 
                                 d_model, 
                                 d_ff, 
                                 d_qk, 
                                 d_v,
                                 dropout=dropout,
                                 attn_dropout=attn_dropout,
                                 ln_eps=ln_eps,
                                 use_pre_norm=use_pre_norm,
                                 activation=activation,
                                 use_rms_norm=use_rms_norm,
                                 use_attention_bias=use_attention_bias,
                                 use_ffn_bias=use_ffn_bias,
                                 use_multi_query_attention=use_multi_query_attention,
                                 use_alibi_bias=use_alibi_bias,
                                 max_relative_len=max_relative_len,
                                 use_rel_pos_value=use_rel_pos_value,
                                 rel_pos_need_train=rel_pos_need_train,
                                 with_across_attention=with_across_attention)
                    for i in range(n_layers//n_share_across_layers)])
    
        self.share_emb_out_proj = share_emb_out_proj
        if share_emb_out_proj == False: 
            self.W = nn.Parameter(torch.Tensor(trg_vocab_size, d_model))
        self.use_output_bias = use_output_bias
        if use_output_bias == True:
            self.b = nn.Parameter(torch.Tensor(trg_vocab_size))

        if self.norm_before_pred == True:
            self.norm = LayerNorm(self.d_model, ln_eps, use_rms_norm)
        
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.d_model)
        if self.share_emb_out_proj == False: 
            self.W.data.uniform_(-stdv, stdv)
        if self.use_output_bias == True:
            self.b.data.zero_()
            
            
    def forward(self, 
                y, 
                self_attn_mask=None,
                dec_kv_list=None, 
                dec_enc_attn_mask=None,
                enc_kv_list=None,
                self_pos_ids=None,
                enc_pos_ids=None,
                past_pos_ids=None,
                cached_kv=False, 
                trg_embedding=None,
                return_states=False):
        """
        """
        if trg_embedding is None:
            trg_embedding = self.trg_embedding

        embeded = trg_embedding(y)
        
        if self.use_pos_embedding == True:    
            embeded = embeded + self.pos_embedding(self_pos_ids)

        dec_output = embeded

        if self.norm_after_embedding == True:
            dec_output = self.norm_emb(dec_output)
        if self.emb_dropout is not None:
            dec_output = self.emb_dropout(dec_output)
 
        self_attn_scores_list = []
        enc_attn_scores_list = []
        dec_states = [dec_output]
        for i in range(self.n_layers):
            
            if self.share_layer_params == False:
                layer = self.layers[i]
            else:
                layer = self.layers[i // self.n_share_across_layers]
                
            outputs = layer(dec_output,
                            self_attn_mask,
                            cached_kv,
                            dec_kv_list[i][0] if cached_kv else None,
                            dec_kv_list[i][1] if cached_kv else None,
                            enc_kv_list[i][0] if cached_kv and self.with_across_attention else enc_kv_list, 
                            enc_kv_list[i][1] if cached_kv and self.with_across_attention else enc_kv_list, 
                            dec_enc_attn_mask,
                            self_pos_ids,
                            enc_pos_ids,
                            past_pos_ids)

            dec_output, self_attn_scores = outputs[:2]
            
            if self.with_across_attention == True:
                enc_attn_scores = outputs[2]
                
            self_attn_scores_list.append(self_attn_scores)
            if self.with_across_attention == True:
                enc_attn_scores_list.append(enc_attn_scores)
            dec_states.append(dec_output)

            dec_output = outputs[0]

            if cached_kv == True:
                dec_keys, dec_values = outputs[-2:]
                dec_kv_list[i][0] = dec_keys
                dec_kv_list[i][1] = dec_values

        if self.norm_before_pred == True:
            dec_output = self.norm(dec_output)
            
        if self.share_emb_out_proj == False:
            W = self.W
        else:
            W = trg_embedding.get_embedding()
            
        logits = F.linear(dec_output, W)
        if self.use_output_bias == True:
            logits = logits + self.b
        
        outputs = [logits]
        if cached_kv == True:
            outputs += [dec_kv_list]

        if return_states == True:
            outputs = outputs + \
            [embeded, dec_states, self_attn_scores_list, enc_attn_scores_list]
            
        return outputs


    def cache_enc_kv(self, enc_output):
        """
        """
        kv_list = []
        for i in range(self.n_layers):
            if not self.share_layer_params:
                layer = self.layers[i]
                cached_kv = layer.cache_enc_kv(enc_output)
            elif i % self.n_share_across_layers == 0:
                layer = self.layers[i // self.n_share_across_layers]
                cached_kv = layer.cache_enc_kv(enc_output)
            else:
                cached_kv = kv_list[-1]
            kv_list.append(cached_kv)
            
        return kv_list
    

    def cache_dec_kv(self, 
                     y=None, 
                     self_attn_mask=None, 
                     enc_kv_list=None, 
                     dec_enc_attn_mask=None, 
                     self_pos_ids=None, 
                     enc_pos_ids=None,
                     past_pos_ids=None,
                     trg_embedding=None):
        """
        """
        dec_kv_list = []
        for i in range(self.n_layers):
            dec_kv_list.append([None, None])
        if y is None:
            return dec_kv_list
        
        if trg_embedding is None:
            trg_embedding = self.trg_embedding
        word_embeded = trg_embedding(y)
        if self.use_pos_embedding == True:
            word_embeded = word_embeded + self.pos_embedding(self_pos_ids)
        dec_output = word_embeded
        
        if self.norm_after_embedding == True:
            dec_output = self.norm_emb(dec_output)

        for i in range(self.n_layers):
            if self.share_layer_params == False:
                layer = self.layers[i]
            else:
                layer = self.layers[i // self.n_share_across_layers]

            outputs = layer(dec_output,
                            self_attn_mask,
                            True,
                            dec_kv_list[i][0],
                            dec_kv_list[i][1],
                            enc_kv_list[i][0] if self.with_across_attention else None,
                            enc_kv_list[i][1] if self.with_across_attention else None,
                            dec_enc_attn_mask,
                            self_pos_ids,
                            enc_pos_ids,
                            past_pos_ids
                            )
            dec_output = outputs[0]

            dec_keys, dec_values = outputs[-2:]
            dec_kv_list[i][0] = dec_keys
            dec_kv_list[i][1] = dec_values

        return dec_kv_list


if __name__ == "__main__":
    pass
