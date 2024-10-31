"""
Created on Tue Aug 13 13:29:30 2019

@author: Xiaoyuan Yao
"""
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Linear(nn.Module):
    """
    """
    def __init__(self, d_in, d_out, use_bias=True):
        """
        """
        super(Linear, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.W = nn.Parameter(torch.Tensor(d_out, d_in))
        self.b = nn.Parameter(torch.Tensor(d_out))
        self.reset_parameters()
        
    
    def reset_parameters(self):
        """
        """
        stdv = math.sqrt(6) / np.sqrt(self.d_in + self.d_out)
        self.W.data.uniform_(-stdv, stdv)
        self.b.data.fill_(0)


    def forward(self, x):
        """
        """
        return F.linear(x, self.W) + self.b
    
    
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
        self.reset_parameters()
        
    
    def reset_parameters(self):
        """
        """
        for weight in [self.start_trans, self.trans]:
            weight.data.uniform_(-0.1, 0.1)
    
    
    def get_normalizer(self, emission, mask=None):
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

            scores = next_scores
        
        scores = torch.logsumexp(scores, 1)

        return scores


    def get_path_score(self, emission, target, mask=None):
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

            scores = next_scores
        
        return scores
    
    
    def forward(self, emission, target, mask=None):
        """
        """
        path_scores = self.get_path_score(emission, target, mask)
        
        normalizer = self.get_normalizer(emission, mask)

        neg_log_likelihood = torch.mean(normalizer - path_scores)
        
        return neg_log_likelihood


class Embedding(nn.Module):
    """
    """
    def __init__(self, vocab_size, embedding_size):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.W = nn.Parameter(torch.Tensor(vocab_size, embedding_size))
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.vocab_size)
        for weight in self.parameters():
            weight.data.uniform_(-math.sqrt(3)*stdv, math.sqrt(3)*stdv)


    def forward(self, x):
        """
        """
        return self.W[x]


    def get_embedding(self):
        """
        """
        return self.W


class FactorizedEmbedding(nn.Module):
    """
    """
    def __init__(self, vocab_size, embedding_size, factorized_size):
        super(FactorizedEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.factorized_size = factorized_size
        self.W = nn.Parameter(torch.Tensor(vocab_size, factorized_size))
        self.We = nn.Parameter(torch.Tensor(embedding_size, factorized_size))             
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.vocab_size)
        for weight in self.parameters():
            weight.data.uniform_(-math.sqrt(3)*stdv, math.sqrt(3)*stdv)


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
    def __init__(self, max_len, d_model, pos_type="learned", base=10000):
        super(PositionEmbedding, self).__init__()
        self.max_len = max_len
        self.embedding_size = d_model
        self.pos_type = pos_type
        if pos_type == "sinusoidal":  
                  
            W = np.zeros([max_len, d_model])
            angle = np.arange(max_len).reshape([-1,1])/np.power(base, np.arange(0, d_model, 2).reshape(1,-1)/d_model)
            W[:,0::2] = np.sin(angle)
            W[:,1::2] = np.cos(angle)
            W = torch.from_numpy(W).float()
            
            self.register_buffer('W', W)
        elif pos_type == "learned":
            self.W = nn.Parameter(torch.Tensor(max_len, d_model))
            self.reset_parameters()
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-math.sqrt(3)*stdv, math.sqrt(3)*stdv)
        
    def forward(self, pos_ids):
        """
        """
        pos_ids[pos_ids >=self.max_len] = -1 
        if self.pos_type == "sinusoidal":
            pe = Variable(self.W[pos_ids], requires_grad=False)
            return pe
        elif self.pos_type == "learned":
            return self.W[pos_ids]


class RoPE(nn.Module):
    """
    """
    def __init__(self, max_len, d_qk, pos_type="sinusoidal", base=10000):
        super(RoPE, self).__init__()
        self.max_len = max_len
        self.embedding_size = d_qk
        self.pos_type = pos_type
        if pos_type == "sinusoidal":

            W = np.zeros([max_len, d_qk])
            angle = np.arange(max_len).reshape([-1,1])/np.power(base, np.arange(0, d_qk, 2).reshape(1,-1)/d_qk)
            W[:,0::2] = np.sin(angle)
            W[:,1::2] = np.cos(angle)
            W = torch.from_numpy(W).float()            

            self.register_buffer('W', W)
        elif pos_type == "learned":
            self.W = nn.Parameter(torch.Tensor(max_len, d_qk))
            self.reset_parameters()
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.max_len + self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-math.sqrt(6)*stdv, math.sqrt(6)*stdv)
       

    def apply_rope(self, x, pos_ids):
        """
        """
        if self.pos_type == "sinusoidal":
            pe = Variable(self.W[pos_ids], requires_grad=False)
        elif self.pos_type == "learned":
            pe = self.W[pos_ids]

        #B x L x d_qk -> B x 1 x L x d_qk
        cos_pos = pe[:, :, 1::2].repeat([1, 1, 2]).unsqueeze(2).transpose(1,2)
        sin_pos = pe[:, :, 0::2].repeat([1, 1, 2]).unsqueeze(2).transpose(1,2)
         
        #B x n x L x d_qk -> B x n x L x d_qk
        x2 = torch.cat([-x[..., self.embedding_size//2:], x[..., :self.embedding_size//2]], -1)
        
        x = x * cos_pos + x2 * sin_pos
        
        return x
        

    def forward(self, q, q_pos_ids, k=None, k_pos_ids=None):
        """
        """
        assert (q_pos_ids < self.max_len).all()
        qw = self.apply_rope(q, q_pos_ids)
        if k is not None:
            kw = self.apply_rope(k, k_pos_ids)
            return qw, kw
        return qw


class RelativePositionEmbedding(nn.Module):
    """
    """
    def __init__(self, max_relative_len, d_model, pos_type="learned", base=10000):
        """
        """
        super(RelativePositionEmbedding, self).__init__()
        self.embedding_size = d_model
        self.max_relative_len = max_relative_len
        self.pos_type = pos_type
        if pos_type == "sinusoidal":
            max_len = 2*max_relative_len + 1
            W = np.zeros([max_len, d_model])
            angle = np.arange(max_len).reshape([-1,1])/np.power(base, np.arange(0, d_model, 2).reshape(1,-1)/d_model)
            W[:,0::2] = np.sin(angle)
            W[:,1::2] = np.cos(angle)
            W = torch.from_numpy(W).float()
            self.register_buffer('W', W)
        elif pos_type == "learned":
            self.W = nn.Parameter(torch.Tensor(2*max_relative_len+1, d_model))
            self.reset_parameters()
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-math.sqrt(3)*stdv, math.sqrt(3)*stdv)
        
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

 
act2fn = {"relu": F.relu, "gelu":F.gelu, "gelu_new":gelu_new, "swish":F.silu}


class GatedFeedForward(nn.Module):
    """
    """
    def __init__(self, d_model, d_ff, activation="relu", dropout=0, use_bias=True):
        """
        """
        super(GatedFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.dropout = None
        if dropout > 0:
            self.dropout = Dropout(dropout)

        self.use_bias = use_bias
        self.W1 = nn.Parameter(torch.Tensor(self.d_ff, self.d_model))
        if self.use_bias == True:
            self.b1 = nn.Parameter(torch.Tensor(self.d_ff))
        self.W2 = nn.Parameter(torch.Tensor(self.d_ff, self.d_model))
        if self.use_bias == True:
            self.b2 = nn.Parameter(torch.Tensor(self.d_ff))            
        self.W3 = nn.Parameter(torch.Tensor(self.d_model, self.d_ff))
        if self.use_bias == True:
            self.b3 = nn.Parameter(torch.Tensor(self.d_model))
            
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """
        """
        stdv = math.sqrt(6) / np.sqrt(self.d_model + self.d_ff)
        for weight in [self.W1, self.W2, self.W3]:
            weight.data.uniform_(-stdv, stdv)
        if self.use_bias == True:
            for weight in [self.b1, self.b2, self.b3]:
                weight.data.fill_(0)
            

    def forward(self, x):
        """
        """
        act_fn = act2fn[self.activation]
        if self.use_bias == True:
            x = F.linear(act_fn(F.linear(x, self.W1) + self.b1) * (F.linear(x, self.W2) + self.b2), self.W3) + self.b3
        else:
            x = F.linear(act_fn(F.linear(x, self.W1)) * F.linear(x, self.W2), self.W3)
            
        if self.dropout is not None:
            x = self.dropout(x)
        return x


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
        self.use_bias = use_bias
        
        self.W1 = nn.Parameter(torch.Tensor(self.d_ff, self.d_model))
        if self.use_bias == True:
            self.b1 = nn.Parameter(torch.Tensor(self.d_ff))
        self.W2 = nn.Parameter(torch.Tensor(self.d_model, self.d_ff))
        if self.use_bias == True:
            self.b2 = nn.Parameter(torch.Tensor(self.d_model))

        self.reset_parameters()
    
    
    def reset_parameters(self):
        """
        """
        stdv = math.sqrt(6) / np.sqrt(self.d_model + self.d_ff)
        for weight in [self.W1, self.W2]:
            weight.data.uniform_(-stdv, stdv)
        if self.use_bias == True:
            for weight in [self.b1, self.b2]:
                weight.data.fill_(0)
            

    def forward(self, x):
        """
        """
        act_fn = act2fn[self.activation]
        if self.use_bias == True:
            x = F.linear(act_fn(F.linear(x, self.W1) + self.b1), self.W2) + self.b2
        else:
            x = F.linear(act_fn(F.linear(x, self.W1)), self.W2)
            
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class TopkRouter(nn.Module):
    """
    """
    def __init__(self, n_experts, topk, d_model, randk=0, use_noise=False):
        """
        """
        super(TopkRouter, self).__init__()
        self.n_experts = n_experts
        self.topk = topk
        self.linear = Linear(d_model, self.n_experts)
        self.randk = randk
        self.use_noise = use_noise
        if self.use_noise == True:
            self.noise_linear = Linear(d_model, self.n_experts)

    
    def forward(self, x):
        """
        """
        logits = self.linear(x)

        if self.use_noise == True:
            noise_logits = self.noise_linear(x)
            noise = torch.randn_like(logits)*F.softplus(noise_logits)
            logits = logits + noise

        #choose real topk = topk - randk
        top_k_logits, indices = logits.topk(self.topk-self.randk, dim=-1)
        
        #choose randk for balance
        if self.randk > 0:
            probs = torch.full_like(logits, 1/self.randk)
            probs = probs.scatter(-1, indices, 0).view(-1, self.n_experts)
            rand_indices = torch.multinomial(probs, 1)
            rand_logits = torch.gather(logits.view(-1, self.n_experts), 1, rand_indices)
            
            indices = torch.cat([indices, rand_indices.view(x.shape[0], x.shape[1], self.randk)], -1)
            top_k_logits = torch.cat([top_k_logits, rand_logits.view(x.shape[0], x.shape[1], self.randk)], -1)

        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return logits, router_output, indices


class SparseMoE(nn.Module):
    """
    """
    def __init__(self, n_experts, topk, d_model, expert, randk=0, use_moe_noise=False):
        """
        """
        super(SparseMoE, self).__init__()
        self.n_experts = n_experts
        self.topk = topk
        self.d_model = d_model
        self.router = TopkRouter(n_experts, topk, d_model, randk, use_moe_noise) 
        self.experts = nn.ModuleList([expert() for _ in range(self.n_experts)])


    def forward(self, x):
        """
        """
        logits, gating_output, indices = self.router(x)
        #B x L x d
        output = torch.zeros_like(x)

        #(BxL) x d
        flat_x = x.view(-1, x.size(-1))
        #(BxL) x n
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i,expert in enumerate(self.experts):
            #B x L
            expert_mask = (indices == i).any(dim=-1)
            #(BL)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                #ne x d
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)
                #ne x 1
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                output[expert_mask] += weighted_output
        
        return output, logits, gating_output, indices


class LayerNorm(nn.Module):
    """
    """
    def __init__(self, d_model, eps=1e-5, use_scale=True, use_bias=True):
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

        
    def forward(self, x):
        """
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, unbiased=False, keepdim=True)
        norm = (x - mean) / (std + self.eps)
            
        if self.use_scale == True:
            norm = self.alpha * norm
        if self.use_bias == True:
            norm = norm + self.bias  
                
        return norm


class RMSNorm(nn.Module):
    """
    """
    def __init__(self, d_model, eps=1e-5, use_scale=True, use_bias=True):
        """
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d_model = d_model
        self.use_scale = use_scale
        if use_scale == True:
            self.alpha = nn.Parameter(torch.ones(self.d_model))
        self.use_bias = use_bias
        if use_bias == True:
            self.bias = nn.Parameter(torch.zeros(self.d_model))

        
    def forward(self, x):
        """
        """
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        norm = x * rms      
            
        if self.use_scale == True:
            norm = self.alpha * norm
        if self.use_bias == True:
            norm = norm + self.bias  
                
        return norm


def scaled_dot_product_attention(query, 
                                 key, 
                                 value, 
                                 attn_mask=None,
                                 attn_scale=None,
                                 dropout=None, 
                                 pos_key=None, 
                                 pos_value=None,
                                 pos_bias=None,
                                 attention_residual=None,
                                 talking_w_pre_softmax=None, 
                                 talking_w_post_softmax=None):
    """
    """
    d = query.size(-1)
    
    #q:B x n_heads x L_q x d_qk 
    #k:B x n_heads x L_kv x d_v 
    #scores:B x n_heads x L_q x L_kv
    scores = torch.einsum("bnqd,bnkd->bnqk", query, key)
    
    if pos_key is not None:
        #p_k:L_q x L_k x d_qk
        scores += torch.einsum("bnqd,bqkd->bnqk", query, pos_key)
    
    if attn_scale is not None:
        scores = scores / attn_scale
 
    if pos_bias is not None:        
        scores += pos_bias

    if talking_w_pre_softmax is not None:
        scores = torch.einsum("bnqk,nm->bmqk", scores, talking_w_pre_softmax)

    if attention_residual is not None:
        scores += attention_residual
    
    if attn_mask is not None:
        attn_mask = attn_mask.bool()
        scores = scores.masked_fill(attn_mask, -1e4)
    
    attn_weight = F.softmax(scores, dim = -1)

    if talking_w_post_softmax is not None:
        attn_weight = torch.einsum("bmqk,mn->bnqk", attn_weight, talking_w_post_softmax)
    
    if dropout is not None:
        attn_weight = dropout(attn_weight)

    #scores:B x n_heads x L_q x L_kv
    #v:B x n_heads x L_kv x d_v 
    output = torch.einsum("bnqk,bnkd->bnqd", attn_weight, value)
    if pos_value is not None:
        #p_v:L_q x L_kv x d_v
        output += torch.einsum("bnqk,bqkd->bnqd", attn_weight, pos_value)
    
    return output, attn_weight, scores
            

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
                 attn_scale=None,
                 use_rope_embedding=False,
                 max_rope_len=-1,
                 max_relative_len=-1,
                 use_rel_pos_value=False,
                 rel_pos_type="learned",
                 n_kv_heads=None,
                 use_pos_bias=False,
                 pos_bias_type=None,
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
        self.n_kv_heads = n_kv_heads
        if self.n_kv_heads is None:
            self.n_kv_heads = n_heads
        self.W_k = nn.Parameter(torch.Tensor(self.n_kv_heads*d_qk, d_model))
        self.W_v = nn.Parameter(torch.Tensor(self.n_kv_heads*d_v, d_model))
        
        self.W_o = nn.Parameter(torch.Tensor(d_model, n_heads*d_v))
        self.use_bias = use_bias
        if self.use_bias == True:
            self.b_q = nn.Parameter(torch.Tensor(n_heads*d_qk))
            self.b_k = nn.Parameter(torch.Tensor(self.n_kv_heads*d_qk))
            self.b_v = nn.Parameter(torch.Tensor(self.n_kv_heads*d_v))
            self.b_o = nn.Parameter(torch.Tensor(d_model))
        
        self.attn_scale = attn_scale
        
        self.use_rope_embedding = use_rope_embedding
        if self.use_rope_embedding == True:
            self.max_rope_len = max_rope_len
            self.rope_emb = RoPE(self.max_rope_len, self.d_qk)
        
        self.rel_pos_k_emb = None
        self.rel_pos_v_emb = None
        self.use_rel_pos_value = use_rel_pos_value
        if max_relative_len > 0:
            self.rel_pos_k_emb = RelativePositionEmbedding(max_relative_len, self.d_qk, rel_pos_type)
            if use_rel_pos_value == True:
                self.rel_pos_v_emb = RelativePositionEmbedding(max_relative_len, self.d_v, rel_pos_type)
        
        self.use_pos_bias = use_pos_bias
        self.pos_bias_type = pos_bias_type
        
        self.use_talking_attention = use_talking_attention
        self.talking_w_pre_softmax = None
        self.talking_w_post_softmax = None
        if use_talking_attention == True:
            self.talking_w_pre_softmax = nn.Parameter(torch.Tensor(n_heads, n_heads))
            self.talking_w_post_softmax = nn.Parameter(torch.Tensor(n_heads, n_heads))
            
        self.reset_parameters()
        
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.d_model)
        for weight in [self.W_q, self.W_k, self.W_v, self.W_o]:
            weight.data.uniform_(-math.sqrt(3)*stdv, math.sqrt(3)*stdv)
        if self.use_bias == True:
            for weight in [self.b_q, self.b_k, self.b_v, self.b_o]:
                weight.data.zero_()
        if self.use_talking_attention == True:
            stdv = 1.0 / np.sqrt(self.n_heads)
            self.talking_w_pre_softmax.data.uniform_(-math.sqrt(3)*stdv, math.sqrt(3)*stdv)
            self.talking_w_after_softmax.data.uniform_(-math.sqrt(3)*stdv, math.sqrt(3)*stdv)
   

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
        #B x l x (d*n_heads) -> B x n_heads x L x d_qk
        query = query.view(batch_size, -1, self.n_heads, self.d_qk).transpose(1, 2)
        if cached_kv == False:
            key = key.view(batch_size, -1, self.n_kv_heads, self.d_qk).transpose(1, 2)
            value = value.view(batch_size, -1, self.n_kv_heads, self.d_v).transpose(1, 2)

        if self.use_rope_embedding == True:
            if cached_kv == False:
                query, key = self.rope_emb(query, q_pos_ids, key, kv_pos_ids)
            else:
                query = self.rope_emb(query, q_pos_ids)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        
        relative_dis = None
        if self.max_relative_len > 0 or self.use_pos_bias == True:
            relative_dis = q_pos_ids[:,:,None] - kv_pos_ids[:,None,:]
        
        pos_key = None
        pos_value = None
        if self.max_relative_len > 0:
            pos_key = self.rel_pos_k_emb(relative_dis=relative_dis)
            if self.use_rel_pos_value == True:
                pos_value = self.rel_pos_k_emb(relative_dis=relative_dis)
        
        pos_bias = None
        if self.use_pos_bias == True:
            if self.pos_bias_type == "alibi":
                start = (2**(-2**-(math.log2(self.n_heads)-3)))
                ratio = start
                #slopes: n_heads
                #relative_dis : B x L_q x L_k
                #alibi_bias: B x n_heads x L_q x L_k
                slopes = torch.tensor([start*ratio**i for i in range(self.n_heads)]).to(query.device, dtype=query.dtype)
                relative_dis[relative_dis<0] = -relative_dis[relative_dis<0]
                pos_bias = torch.einsum("bqk,n->bnqk", relative_dis, slopes)
        
        if self.n_heads != self.n_kv_heads and self.n_kv_heads != 1:
            n_rep = self.n_heads // self.n_kv_heads
            key = key[:, :, None, :, :].repeat(1, 1, n_rep, 1, 1).reshape(batch_size, self.n_heads, -1,  self.d_qk)
            value = value[:, :, None, :, :].repeat(1, 1, n_rep, 1, 1).reshape(batch_size, self.n_heads, -1,  self.d_qk)
        
        output, attn_weight, attn_scores = scaled_dot_product_attention(query, 
                                                                        key, 
                                                                        value, 
                                                                        attn_mask,
                                                                        self.attn_scale,
                                                                        self.attn_dropout,
                                                                        pos_key,
                                                                        pos_value,
                                                                        pos_bias,
                                                                        attention_residual,
                                                                        self.talking_w_pre_softmax,
                                                                        self.talking_w_post_softmax)
        
        #B x n_heads x L x d -> B x L x n_heads x d -> B x L x d_model
        output = output.transpose(1,2)
        output = output.contiguous().view(batch_size, -1, self.n_heads*self.d_v)

        
        output = F.linear(output, self.W_o)
        if self.use_bias == True:
            output = output + self.b_o
            
        if self.attn_dropout is not None:
            output = self.dropout(output)

        return output, attn_weight, attn_scores


    def cache_kv(self, x, pos_ids=None):
        """
        """
        batch_size = x.size(0)

        key = F.linear(x, self.W_k)
        value = F.linear(x, self.W_v) 
        if self.use_bias == True:
            key = key + self.b_k
            value = value + self.b_v

        key = key.view(batch_size, -1, self.n_kv_heads, self.d_qk).transpose(1, 2)
        value = value.view(batch_size, -1, self.n_kv_heads, self.d_v).transpose(1, 2)

        if self.use_rope_embedding == True:
            key = self.rope_emb(key, pos_ids)

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
        self.norm_type = kwargs.get("layer_norm_type", "layer_norm")
        self.use_attention_bias = kwargs.get("use_attention_bias", True)
        self.use_ffn_bias = kwargs.get("use_ffn_bias", True)
        self.n_kv_heads = kwargs.get("n_kv_heads", None)
        self.use_rope_embedding = kwargs.get("use_rope_embedding", False)
        self.max_rope_len = kwargs.get("max_rope_len", -1)
        self.use_pos_bias = kwargs.get("use_pos_bias", False)
        self.pos_bias_type = kwargs.get("pos_bias_type", None)
        self.max_relative_len = kwargs.get("max_relative_len", -1)
        self.use_rel_pos_value = kwargs.get("use_rel_pos_value", False)
        self.rel_pos_type = kwargs.get("rel_pos_type", "learned")
        self.with_across_attention = kwargs.get("with_across_attention", False)
        self.use_talking_attention = kwargs.get("use_talking_attention", False)
        self.use_glu = kwargs.get("use_glu", False)
        self.use_moe = kwargs.get("use_moe", False)
        self.n_experts = kwargs.get("n_experts", 1)
        self.moe_topk = kwargs.get("moe_topk", 1)
        self.moe_randk = kwargs.get("moe_randk", False) 
        self.use_moe_noise = kwargs.get("use_moe_noise", True) 
        self.use_ln_scale = kwargs.get("use_ln_scale", True)
        self.use_ln_bias = kwargs.get("use_ln_bias", True)
        self.layer_scale = kwargs.get("layer_scale", None)
        self.attn_scale = kwargs.get("attn_scale", np.sqrt(self.d_qk))
        
        self.self_attention = MultiHeadAttention(self.n_heads,
                                                 self.d_model, 
                                                 self.d_qk, 
                                                 self.d_v, 
                                                 self.dropout,
                                                 self.attn_dropout,
                                                 self.use_attention_bias,
                                                 self.attn_scale,
                                                 self.use_rope_embedding,
                                                 self.max_rope_len,
                                                 self.max_relative_len,
                                                 self.use_rel_pos_value,
                                                 self.rel_pos_type,
                                                 self.n_kv_heads,
                                                 self.use_pos_bias,
                                                 self.pos_bias_type,
                                                 self.use_talking_attention)
        
        if self.norm_type == "layer_norm":
            self.norm_1 = LayerNorm(self.d_model,self.ln_eps,self.use_ln_scale,self.use_ln_bias)
        elif self.norm_type == "rms_norm":
            self.norm_1 = RMSNorm(self.d_model,self.ln_eps,self.use_ln_scale,self.use_ln_bias)
            
        if self.with_across_attention == True:
            self.enc_attention = MultiHeadAttention(self.n_heads,
                                                    self.d_model, 
                                                    self.d_qk, 
                                                    self.d_v, 
                                                    self.dropout,
                                                    self.attn_dropout,
                                                    self.use_attention_bias,
                                                    self.attn_scale,
                                                    self.use_rope_embedding,
                                                    self.max_rope_len,
                                                    self.max_relative_len,
                                                    self.use_rel_pos_value,
                                                    self.rel_pos_type,
                                                    self.n_kv_heads,
                                                    False,
                                                    None,
                                                    self.use_talking_attention)
        
            if self.norm_type == "layer_norm":
                self.norm_2 = LayerNorm(self.d_model,self.ln_eps,self.use_ln_scale,self.use_ln_bias)
            elif self.norm_type == "rms_norm":
                self.norm_2 = RMSNorm(self.d_model,self.ln_eps,self.use_ln_scale,self.use_ln_bias)
        
        ffn_cls = FeedForward
        if self.use_glu == True:
            ffn_cls = GatedFeedForward
        
        ffn = partial(ffn_cls,
                      d_model=self.d_model,
                      d_ff=self.d_ff,
                      activation=self.activation,
                      dropout=self.dropout,
                      use_bias=self.use_ffn_bias,
                     )

        if self.use_moe == True:
            self.ffn = SparseMoE(self.n_experts, self.moe_topk, self.d_model, ffn, self.moe_randk, self.use_moe_noise)
        else:
            self.ffn = ffn()
        
        if self.with_across_attention == True:
            if self.norm_type == "layer_norm":
                self.norm_3 = LayerNorm(self.d_model,self.ln_eps,self.use_ln_scale,self.use_ln_bias)
            elif self.norm_type == "rms_norm":
                self.norm_3 = RMSNorm(self.d_model,self.ln_eps,self.use_ln_scale,self.use_ln_bias)
        else:
            if self.norm_type == "layer_norm":
                self.norm_2 = LayerNorm(self.d_model,self.ln_eps,self.use_ln_scale,self.use_ln_bias)
            elif self.norm_type == "rms_norm":
                self.norm_2 = RMSNorm(self.d_model,self.ln_eps,self.use_ln_scale,self.use_ln_bias)


        if self.layer_scale == "learned":
            init_layer_scale = kwargs.get("init_layer_scale", 1)
            self.gamma_1 = nn.Parameter(init_layer_scale * torch.ones((self.d_model)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_layer_scale * torch.ones((self.d_model)), requires_grad=True)
            if self.with_across_attention == True:
                self.gamma_3 = nn.Parameter(init_layer_scale * torch.ones((self.d_model)), requires_grad=True)
        
        
    def forward(self, 
                x,  
                self_attn_mask=None, 
                cached_kv=False, 
                self_keys=None, 
                self_values=None,
                enc_keys=None, 
                enc_values=None,
                dec_enc_attn_mask=None,
                self_pos_ids=None,
                enc_pos_ids=None,
                past_pos_ids=None,
                self_attention_residual=None,
                enc_attention_residual=None):
        """
        """
        output = x

        residual = output
        if self.use_pre_norm == True:
            output = self.norm_1(output)
        
        if cached_kv == True:
            kv = self.cache_dec_kv(output, self_pos_ids)
            
            if self_keys is None:
                self_keys = kv[0]                
            else:
                self_keys = torch.cat([self_keys, kv[0]], 2)
            if self_values is None:
                self_values = kv[1]
            else:
                self_values = torch.cat([self_values, kv[1]], 2)
                
        else:
            self_keys = output
            self_values = output
        
        output, self_attn_weight, self_attn_scores = self.self_attention(output, 
                                                                         self_keys, 
                                                                         self_values, 
                                                                         self_attn_mask,
                                                                         cached_kv,
                                                                         self_pos_ids,
                                                                         past_pos_ids,
                                                                         self_attention_residual)
        if self.layer_scale is None:
            output = residual + output
        else:
            output = residual + self.gamma_1 * output
            
        if self.use_pre_norm == False:
            output = self.norm_1(output)
        
        residual = output
        
        if self.with_across_attention == True:
            if self.use_pre_norm == True:
                output = self.norm_2(output)
            
            output, enc_attn_weight, enc_attn_scores = self.enc_attention(output, 
                                                                          enc_keys, 
                                                                          enc_values, 
                                                                          dec_enc_attn_mask,
                                                                          cached_kv,
                                                                          self_pos_ids,
                                                                          enc_pos_ids,
                                                                          enc_attention_residual)

            if self.layer_scale is None:
                output = residual + output
            else:
                output = residual + self.gamma_2 * output
            
            if self.use_pre_norm == False:
                output = self.norm_2(output)
            
            residual = output
                
        if self.use_pre_norm == True:
            if self.with_across_attention == True:
                output = self.norm_3(output)
            else:
                output = self.norm_2(output)

        if self.use_moe == False:
            output = self.ffn(output)
        else:
            output, moe_logits, moe_weights, moe_indices = self.ffn(output)

        if self.layer_scale is None:
            output = residual + output
        else:
            if self.with_across_attention == True:
                output = residual + self.gamma_3 * output
            else:
                output = residual + self.gamma_2 * output

        if self.use_pre_norm == False:
            if self.with_across_attention == True:
                output = self.norm_3(output)
            else:
                output = self.norm_2(output)

        outputs = {}
        outputs["output"] = output
        outputs["self_attn_weight"] = self_attn_weight
        outputs["self_attn_scores"] = self_attn_scores
        if self.with_across_attention == True:
            outputs["enc_attn_weight"] = enc_attn_weight
            outputs["enc_attn_scores"] = enc_attn_scores

        if self.use_moe == True:
            outputs["moe_logits"] = moe_logits
            outputs["moe_weights"] = moe_weights
            outputs["moe_indices"] = moe_indices

        if cached_kv == True:
            outputs["cache_sa_keys"] = self_keys
            outputs["cache_sa_values"] = self_values
        
        return outputs

    def cache_enc_kv(self, enc_output, enc_pos_ids=None):
        """
        """
        return self.enc_attention.cache_kv(enc_output, enc_pos_ids)


    def cache_dec_kv(self, dec_output, self_pos_ids=None):
        """
        """
        return self.self_attention.cache_kv(dec_output, self_pos_ids)


