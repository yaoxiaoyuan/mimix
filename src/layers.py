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
    
class GRUCell(nn.Module):
    """
    """
    def __init__(self, input_size, hidden_size):
        """
        """
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Ws = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))
        self.Wr = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))
        self.Wz = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))

        self.Us = nn.Parameter(torch.Tensor(self.hidden_size, 
                                            self.hidden_size))
        self.Ur = nn.Parameter(torch.Tensor(self.hidden_size, 
                                            self.hidden_size))
        self.Uz = nn.Parameter(torch.Tensor(self.hidden_size, 
                                            self.hidden_size))
        
        self.bs = nn.Parameter(torch.Tensor(self.hidden_size))
        self.br = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bz = nn.Parameter(torch.Tensor(self.hidden_size))

        self.reset_parameters()


    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, x, mask, last_s):
        """
        """
        if last_s is None:
            batch_size = x.size(0)
            last_s = torch.zeros((batch_size, self.hidden_size), 
                                 dtype=torch.float32)            
            if x.is_cuda == True:
                last_s = last_s.to(x.device)
        
        z = torch.sigmoid(F.linear(x, self.Wz) + 
                          F.linear(last_s, self.Uz) + 
                          self.bz)
        r = torch.sigmoid(F.linear(x, self.Wr) + 
                          F.linear(last_s, self.Ur) + 
                          self.br)
        s_hat = F.relu(F.linear(x, self.Ws) + 
                       F.linear(last_s * r, self.Us) + 
                       self.bs)
        s = (1 - z) * last_s + z * s_hat
        new_s = last_s * (1 - mask) + s * mask
        return new_s


class GRU(nn.Module):
    """
    """
    def __init__(self, input_size, hidden_size, backward=False,
                 dropout=0):
        """
        """
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.backward = backward
        self.cell = GRUCell(input_size, hidden_size)
        self.dropout = Dropout(dropout)
    
    def forward(self, x, mask, s):
        """
        """
        enc_time_steps = x.size(1)
        
        outputs = []
        for t in range(enc_time_steps):
            if self.backward == True:
                t = enc_time_steps - 1 - t
            s = self.cell.forward(x[:,t,:], mask[:,t,:], s)
            outputs.append(s)
        
        if self.backward == True:
            outputs = outputs[::-1]
        
        y = torch.stack(outputs).transpose(0, 1)
        return self.dropout(y)


class BiGRU(nn.Module):
    """
    """
    def __init__(self, input_size, hidden_size, dropout=0):
        """
        """
        super(BiGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = Dropout(dropout)
        self.gru = GRU(input_size, hidden_size, backward=False)
        self.r_gru = GRU(input_size, hidden_size, backward=True)


    def forward(self, x, mask, s):
        """
        """
        s_forward, s_backward = s
        new_s_forward = self.gru(x, mask, s_forward)
        new_s_backward = self.r_gru(x, mask, s_backward)
        new_s =  torch.cat([new_s_forward, new_s_backward],dim=-1)
        return self.dropout(new_s)


class LSTMCell(nn.Module):
    """
    """
    def __init__(self, input_size, hidden_size):
        """
        """
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wi = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))
        self.Wf = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))
        self.Wo = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))
        self.Wc = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))
        
        self.Ui = nn.Parameter(torch.Tensor(self.hidden_size, 
                                            self.hidden_size))
        self.Uf = nn.Parameter(torch.Tensor(self.hidden_size, 
                                            self.hidden_size))
        self.Uo = nn.Parameter(torch.Tensor(self.hidden_size, 
                                            self.hidden_size))
        self.Uc = nn.Parameter(torch.Tensor(self.hidden_size, 
                                            self.hidden_size))
        
        self.bi = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bf = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bo = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bc = nn.Parameter(torch.Tensor(self.hidden_size))

        self.reset_parameters()


    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, x, mask, last_s):
        """
        """
        last_c, last_h = last_s
        if last_c is None:
            batch_size = x.size(0)
            last_c = torch.zeros((batch_size, self.hidden_size), 
                                 dtype=torch.float32)
            if x.is_cuda == True:
                last_c = last_c.to(x.device)

        if last_h is None:
            batch_size = x.size(0)
            last_h = torch.zeros((batch_size, self.hidden_size),
                                 dtype=torch.float32)
            if x.is_cuda == True:
                last_h = last_h.to(x.device)
        
        i = torch.sigmoid(F.linear(x, self.Wi) + 
                          F.linear(last_h, self.Ui) + 
                          self.bi)
        f = torch.sigmoid(F.linear(x, self.Wf) + 
                          F.linear(last_h, self.Uf) + 
                          self.bf)
        o = torch.sigmoid(F.linear(x, self.Wo) + 
                          F.linear(last_h, self.Uo) + 
                          self.bo)
        c_hat = torch.tanh(F.linear(x, self.Wc) + 
                           F.linear(last_h, self.Uc) + 
                           self.bc)
        
        c = i * c_hat + f * last_c 
        h = o * torch.tanh(c)
        
        c = c * mask + last_c * (1 - mask)
        h = h * mask + last_h * (1 - mask)
        
        return c,h


class LSTM(nn.Module):
    """
    """
    def __init__(self, input_size, hidden_size, backward=False, dropout=0):
        """
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.backward = backward
        self.cell = LSTMCell(input_size, hidden_size)
        self.dropout = Dropout(dropout)
    
    def forward(self, x, mask, s):
        """
        """
        enc_time_steps = x.size(1)
        
        c_outputs = []
        s_outputs = []
        for t in range(enc_time_steps):
            if self.backward == True:
                t = enc_time_steps - 1 - t
            s = self.cell.forward(x[:,t,:], mask[:,t,:], s)
            c_outputs.append(s[0])
            s_outputs.append(s[1])
        
        if self.backward == True:
            c_outputs = c_outputs[::-1]
            s_outputs = s_outputs[::-1]
        
        s_output = torch.stack(s_outputs).transpose(0, 1)
        c_output = torch.stack(c_outputs).transpose(0, 1)
        return self.dropout(s_output), c_output


class BiLSTM(nn.Module):
    """
    """
    def __init__(self, input_size, hidden_size, dropout=0):
        """
        """
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = Dropout(dropout)
        self.lstm = LSTM(input_size, hidden_size, backward=False)
        self.r_lstm = LSTM(input_size, hidden_size, backward=True)


    def forward(self, x, mask, s):
        """
        """
        s_forward, s_backward = s
        new_s_forward = self.lstm(x, mask, s_forward)[0]
        new_s_backward = self.r_lstm(x, mask, s_backward)[0]
        new_s =  torch.cat([new_s_forward, new_s_backward],dim=-1)
        return self.dropout(new_s)


class RNNEncoder(nn.Module):
    """
    """
    def __init__(self, src_vocab_size, src_emb_size,
                 hidden_size, key_size, n_layers, 
                 cell_type, dropout=0):
        """
        """
        super(RNNEncoder, self).__init__()
        self.src_embedding = Embedding(src_vocab_size, src_emb_size)
        self.src_emb_size = src_emb_size
        self.hidden_size = hidden_size
        self.key_size = key_size
        self.U_attr = nn.Parameter(torch.Tensor(self.key_size, 
                                                2 * self.hidden_size))
        
        self.cell_type = cell_type
        self.n_layers = n_layers
        layers = []
        for i in range(self.n_layers):
            if self.cell_type == "gru":
                if i == 0:
                    layers.append(BiGRU(src_emb_size, 
                                        hidden_size, 
                                        dropout=dropout))
                else:
                    layers.append(BiGRU(2 * self.hidden_size, 
                                        hidden_size, 
                                        dropout=dropout))
            elif self.cell_type == "lstm":
                if i == 0:
                    layers.append(BiLSTM(src_emb_size, 
                                         hidden_size, 
                                         dropout=dropout))
                else:
                    layers.append(BiLSTM(2 * self.hidden_size, 
                                         hidden_size, 
                                         dropout=dropout))
            else:
                raise ValueError("rnn cell type not correct!")
                
        self.layers = nn.ModuleList(layers)
        self.reset_parameters()


    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in [self.U_attr]:
            weight.data.uniform_(-stdv, stdv)


    def forward(self, x, x_mask):
        """
        """
        x_embedding = self.src_embedding(x)
        
        for i in range(self.n_layers):
            if self.cell_type == "gru":
                if i == 0:
                    enc_states = self.layers[i](x_embedding, 
                                                x_mask, 
                                                [None, None])
                else:
                    enc_states = self.layers[i](enc_states, 
                                                x_mask, 
                                                [None, None])
            elif self.cell_type == "lstm":
                if i == 0:
                    enc_states = self.layers[i](x_embedding, 
                                                x_mask, 
                                                [[None, None], [None, None]])
                else:
                    enc_states = self.layers[i](enc_states, 
                                                x_mask, 
                                                [[None, None], [None, None]])
        enc_keys = F.linear(enc_states, self.U_attr)
        enc_values = enc_states
        return enc_states, enc_keys, enc_values 


class BahdanauAttention(nn.Module):
    """
    """
    def __init__(self, query_size, attr_key_size):
        """
        """
        super(BahdanauAttention, self).__init__()
        self.FLOAT_MIN = -100000.0
        self.query_size = query_size
        self.attr_key_size = attr_key_size
        self.V_attr = nn.Parameter(torch.Tensor(self.attr_key_size))
        self.b_attr = nn.Parameter(torch.Tensor(self.attr_key_size))     
        self.W_attr = nn.Parameter(torch.Tensor(self.attr_key_size, 
                                                self.query_size))
        
        self.reset_parameters()


    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.attr_key_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
            
    def forward(self, query, keys, values, kv_mask):
        """
        """
        query = torch.unsqueeze(F.linear(query, self.W_attr), 1)
        energies = torch.sum(
                torch.tanh(keys + query + self.b_attr)*self.V_attr, 
                2, 
                keepdim=True)
        energies = kv_mask * energies + (1 - kv_mask) * self.FLOAT_MIN
        scores = F.softmax(energies, 1)
        ctx = torch.sum(values * scores, 1)
        return scores,ctx


class LuongAttention(nn.Module):
    """
    """
    def __init__(self, query_size, attr_key_size):
        """
        """
        super(LuongAttention, self).__init__()
        self.FLOAT_MIN = -100000.0
        self.query_size = query_size
        self.attr_key_size = attr_key_size
        assert query_size == attr_key_size

    def forward(self, query, keys, values, kv_mask):
        """
        """
        energies = torch.bmm(keys, query.unsqueeze(2))
        energies = kv_mask * energies + (1 - kv_mask) * self.FLOAT_MIN
        scores = F.softmax(energies, 1)
        ctx = torch.sum(values * scores, 1)
        return scores,ctx


class DecoderGRUCell(nn.Module):
    """
    """
    def __init__(self, input_size, hidden_size, ctx_size):
        """
        """
        super(DecoderGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        
        self.Ws = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))
        self.Wr = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))
        self.Wz = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))

        self.Us = nn.Parameter(torch.Tensor(self.hidden_size, 
                                            self.hidden_size))
        self.Ur = nn.Parameter(torch.Tensor(self.hidden_size, 
                                            self.hidden_size))
        self.Uz = nn.Parameter(torch.Tensor(self.hidden_size, 
                                            self.hidden_size))
                
        self.Cs = nn.Parameter(torch.Tensor(self.hidden_size, self.ctx_size))
        self.Cr = nn.Parameter(torch.Tensor(self.hidden_size, self.ctx_size))
        self.Cz = nn.Parameter(torch.Tensor(self.hidden_size, self.ctx_size))

        self.bs = nn.Parameter(torch.Tensor(self.hidden_size))
        self.br = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bz = nn.Parameter(torch.Tensor(self.hidden_size))        

        self.reset_parameters()


    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, y, ctx, last_s):
        """
        """
        z = torch.sigmoid(F.linear(y, self.Wz) + 
                      F.linear(last_s, self.Uz) + 
                      F.linear(ctx, self.Cz) + 
                      self.bz)
        r = torch.sigmoid(F.linear(y, self.Wr) + 
                      F.linear(last_s, self.Ur) + 
                      F.linear(ctx, self.Cr) + 
                      self.br)
        s_hat = F.relu(F.linear(y, self.Ws) + 
                       F.linear(last_s * r, self.Us) + 
                       F.linear(ctx, self.Cs) + 
                       self.bs)

        new_s = (1 - z) * last_s + z * s_hat
        
        return new_s


class DecoderGRU(nn.Module):
    """
    """
    def __init__(self, input_size, hidden_size, ctx_size, 
                 query_size, key_size, attention, dropout):
        """
        """
        super(DecoderGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        
        if attention == "Bahdanau":
            self._attention = BahdanauAttention(query_size, key_size)
        elif attention == "Luong":
            self._attention = LuongAttention(query_size, key_size)
        else:
            raise ValueError("attention not correct!")
        
        self.dropout = Dropout(dropout)
        self._decoder_cell = DecoderGRUCell(self.input_size, 
                                            self.hidden_size, 
                                            self.ctx_size)
        
    
    def step(self, enc_keys, enc_values, enc_mask, last_s, y):
        """
        """
        if last_s is None:
            batch_size = y.size(0)
            last_s = torch.zeros((batch_size, self.hidden_size), 
                                 dtype=torch.float32)            
            if y.is_cuda == True:
                last_s = last_s.to(y.device)
                
        scores,ctx = self._attention(last_s, enc_keys, enc_values, enc_mask)
        s = self._decoder_cell(y, ctx, last_s)
        
        return s,ctx
    
    
    def forward(self, enc_keys, enc_values, enc_mask, last_s, y):
        """
        """
        if last_s is None:
            batch_size = y.size(0)
            last_s = torch.zeros((batch_size, self.hidden_size), 
                                 dtype=torch.float32)            
            if y.is_cuda == True:
                last_s = last_s.to(y.device)
                
        dec_time_steps = y.size(1)
        s = last_s
        dec_states = []
        dec_ctx = []
        for i in range(dec_time_steps):
            scores,ctx = self._attention(s, enc_keys, enc_values, enc_mask)
            s = self._decoder_cell(y[:,i,:], ctx, s)
            dec_states.append(s)
            dec_ctx.append(ctx)
            
        dec_states = self.dropout(torch.stack(dec_states).transpose(0, 1))
        dec_ctx = self.dropout(torch.stack(dec_ctx).transpose(0, 1))
        
        return dec_states,dec_ctx
        

class DecoderLSTMCell(nn.Module):
    """
    """
    def __init__(self, input_size, hidden_size, ctx_size):
        """
        """
        super(DecoderLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        
        self.Wi = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))
        self.Wf = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))
        self.Wo = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))
        self.Wc = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))
        
        self.Ui = nn.Parameter(torch.Tensor(self.hidden_size, 
                                            self.hidden_size))
        self.Uf = nn.Parameter(torch.Tensor(self.hidden_size, 
                                            self.hidden_size))
        self.Uo = nn.Parameter(torch.Tensor(self.hidden_size, 
                                            self.hidden_size))
        self.Uc = nn.Parameter(torch.Tensor(self.hidden_size, 
                                            self.hidden_size))
    
        self.Ci = nn.Parameter(torch.Tensor(self.hidden_size, self.ctx_size))
        self.Cf = nn.Parameter(torch.Tensor(self.hidden_size, self.ctx_size))
        self.Co = nn.Parameter(torch.Tensor(self.hidden_size, self.ctx_size))
        self.Cc = nn.Parameter(torch.Tensor(self.hidden_size, self.ctx_size))

        self.bi = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bf = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bo = nn.Parameter(torch.Tensor(self.hidden_size))        
        self.bc = nn.Parameter(torch.Tensor(self.hidden_size))        

        self.reset_parameters()


    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, y, ctx, last_s):
        """
        """
        c, h = last_s
        i = torch.sigmoid(F.linear(y, self.Wi) + 
                          F.linear(h, self.Ui) + 
                          F.linear(ctx, self.Ci) + 
                          self.bi)
        f = torch.sigmoid(F.linear(y, self.Wf) + 
                          F.linear(h, self.Uf) + 
                          F.linear(ctx, self.Cf) + 
                          self.bf)
        o = torch.sigmoid(F.linear(y, self.Wo) + 
                          F.linear(h, self.Uo) + 
                          F.linear(ctx, self.Co) + 
                          self.bo)
        
        c_hat = torch.tanh(F.linear(y, self.Wc) + 
                           F.linear(h, self.Uc) + 
                           F.linear(ctx, self.Cc) + 
                           self.bc)
        
        c = i * c_hat + f * c
        h = o * torch.tanh(c)
        
        return c,h


class DecoderLSTM(nn.Module):
    """
    """
    def __init__(self, input_size, hidden_size, ctx_size, 
                 query_size, key_size, attention, dropout):
        """
        """
        super(DecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        
        if attention == "Bahdanau":
            self._attention = BahdanauAttention(query_size, key_size)
        elif attention == "Luong":
            self._attention = LuongAttention(query_size, key_size)
        else:
            raise ValueError("attention not correct!")
        
        self.dropout = Dropout(dropout)
        self._decoder_cell = DecoderLSTMCell(self.input_size, 
                                             self.hidden_size, 
                                             self.ctx_size)
        
    
    def step(self, enc_keys, enc_values, enc_mask, last_s, y):
        """
        """
        c,h = last_s
        if c is None:
            batch_size = y.size(0)
            c = torch.zeros((batch_size, self.hidden_size), 
                            dtype=torch.float32)
            if y.is_cuda == True:
                c = c.to(y.device)

        if h is None:
            batch_size = y.size(0)
            h = torch.zeros((batch_size, self.hidden_size), 
                            dtype=torch.float32)
            if y.is_cuda == True:
                h = h.to(y.device)
                
        scores,ctx = self._attention(h, enc_keys, enc_values, enc_mask)
        c,h = self._decoder_cell(y, ctx, [c, h])
        
        return [c,h],ctx
    
    
    def forward(self, enc_keys, enc_values, enc_mask, last_s, y):
        """
        """
        c,h = last_s
        if c is None:
            batch_size = y.size(0)
            c = torch.zeros((batch_size, self.hidden_size), 
                            dtype=torch.float32)
            if y.is_cuda == True:
                c = c.to(y.device)

        if h is None:
            batch_size = y.size(0)
            h = torch.zeros((batch_size, self.hidden_size), 
                            dtype=torch.float32)
            if y.is_cuda == True:
                h = h.to(y.device)
                
        dec_time_steps = y.size(1)
        dec_states = []
        dec_ctx = []
        for i in range(dec_time_steps):
            scores,ctx = self._attention(h, enc_keys, enc_values, enc_mask)
            c,h = self._decoder_cell(y[:,i,:], ctx, [c, h])
            dec_states.append(h)
            dec_ctx.append(ctx)
            
        dec_states = self.dropout(torch.stack(dec_states).transpose(0, 1))
        dec_ctx = self.dropout(torch.stack(dec_ctx).transpose(0, 1))
        
        return dec_states,dec_ctx


class AttentionDecoder(nn.Module):
    """
    """
    def __init__(self, trg_vocab_size, trg_emb_size, hidden_size, 
                 query_size, key_size, value_size, attention, 
                 n_layers, cell_type, dropout=0, share_src_trg_emb=False):
        """
        """
        super(AttentionDecoder, self).__init__()
        if share_src_trg_emb == False:
            self.trg_embedding = Embedding(trg_vocab_size, trg_emb_size)

        self.trg_emb_size = trg_emb_size
        self.hidden_size = hidden_size
        self.ctx_size = value_size
        self.vocab_size = trg_vocab_size
        
        self.cell_type = cell_type
        self.n_layers = n_layers
        layers = []
        for i in range(n_layers):
            if self.cell_type == "gru":
                if i == 0:
                    layers.append(DecoderGRU(trg_emb_size, hidden_size, 
                                             value_size, query_size, key_size, 
                                             attention, dropout))
                else:
                    layers.append(DecoderGRU(hidden_size, hidden_size, 
                                             value_size, query_size, key_size, 
                                             attention, dropout))
            elif self.cell_type == "lstm":
                if i == 0:
                    layers.append(DecoderLSTM(trg_emb_size, hidden_size, 
                                              value_size, query_size, key_size,
                                              attention, dropout))
                else:
                    layers.append(DecoderLSTM(hidden_size, hidden_size, 
                                              value_size, query_size, key_size,
                                              attention, dropout))
            else:
                raise ValueError("rnn cell type not correct!")
                
        self.layers = nn.ModuleList(layers)
        
        self.Uo = nn.Parameter(torch.Tensor(trg_vocab_size, hidden_size))
        self.Co = nn.Parameter(torch.Tensor(trg_vocab_size, self.ctx_size))
        self.Vo = nn.Parameter(torch.Tensor(trg_vocab_size, trg_emb_size))
    
        self.reset_parameters()


    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in [self.Uo, self.Co, self.Vo]:
            weight.data.uniform_(-stdv, stdv)
        
    
    def _step(self, steps, enc_states, dec_enc_attn_mask, last_dec_states, y, 
              trg_embedding=None):
        """
        """
        y = y.view(-1)
        enc_keys, enc_values = enc_states
        
        if trg_embedding is None:
            trg_embedding = self.trg_embedding
        y_emb = trg_embedding(y)
        
        new_s = []
        for i in range(self.n_layers):
            if self.cell_type == "gru":
                if i == 0:
                    dec_states,ctx = self.layers[i].step(enc_keys, enc_values, 
                                                         dec_enc_attn_mask, 
                                                         last_dec_states[i], 
                                                         y_emb)
                else:
                    dec_states,ctx = self.layers[i].step(enc_keys, enc_values,
                                                         dec_enc_attn_mask, 
                                                         last_dec_states[i], 
                                                         dec_states)
                new_s.append(dec_states)
            else:
                if i == 0:
                    [cell_states, dec_states], ctx = self.layers[i].step(
                            enc_keys, enc_values, dec_enc_attn_mask, 
                            last_dec_states[i], y_emb)
                else:
                    [cell_states, dec_states], ctx = self.layers[i].step(
                            enc_keys, enc_values, dec_enc_attn_mask, 
                            last_dec_states[i], dec_states)
                new_s.append([cell_states, dec_states])
        
        logits = F.linear(dec_states, self.Uo) + \
                 F.linear(y_emb, self.Vo) + \
                 F.linear(ctx, self.Co)
        
        return new_s,logits
    
    
    def forward(self, enc_keys, enc_values, enc_mask, y, trg_embedding=None):
        """
        """
        if trg_embedding is None:
            trg_embedding = self.trg_embedding
        y_emb = trg_embedding(y)
        
        for i in range(self.n_layers):
            if self.cell_type == "gru":
                if i == 0:
                    dec_states,ctx = self.layers[i](enc_keys, enc_values, 
                                                    enc_mask, None, y_emb)
                else:
                    dec_states,ctx = self.layers[i](enc_keys, enc_values, 
                                                    enc_mask, None, dec_states)
            elif self.cell_type == "lstm":
                if i == 0:
                    dec_states,ctx = self.layers[i](enc_keys, enc_values, 
                                                    enc_mask, [None,None], 
                                                    y_emb)
                else:
                    dec_states,ctx = self.layers[i](enc_keys, enc_values, 
                                                    enc_mask, [None,None], 
                                                    dec_states)
            
        logits = F.linear(dec_states, self.Uo) + \
                 F.linear(y_emb, self.Vo) + \
                 F.linear(ctx, self.Co)
        
        return [logits]


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
        
    def forward(self, x, start_pos=0):
        """
        """
        seq_len = x.size(1)
        
        if self.need_train == False:
            pe = Variable(self.W[start_pos:start_pos + seq_len, :], 
                          requires_grad=False)
            return pe
        else:
            return self.W[start_pos:start_pos + seq_len, :]


class Dropout(nn.Module):
    def __init__(self, p=0):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability incorrect!")
        self.p = p

    def forward(self, x):
        if self.training:
            rand = (torch.rand_like(x, device = x.device) > self.p).float() 
            scale = (1.0/(1-self.p))
            return x * rand * scale
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
    def __init__(self, d_model, d_ff, activation="relu", dropout=0):
        """
        """
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.dropout = Dropout(dropout)
        self.W1 = nn.Parameter(torch.Tensor(self.d_ff, self.d_model))
        self.b1 = nn.Parameter(torch.Tensor(self.d_ff))
        self.W2 = nn.Parameter(torch.Tensor(self.d_model, self.d_ff))
        self.b2 = nn.Parameter(torch.Tensor(self.d_model))
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.d_model)
        for weight in [self.W1, self.W2]:
            weight.data.uniform_(-stdv, stdv)
        for weight in [self.b1, self.b2]:
            weight.data.fill_(0)
            

    def forward(self, x):
        """
        """
        if self.activation == "relu":
            x = F.linear(F.relu(F.linear(x, self.W1) + self.b1), 
                         self.W2) + self.b2
        elif self.activation == "gelu":
            x = F.linear(F.gelu(F.linear(x, self.W1) + self.b1), 
                         self.W2) + self.b2
        elif self.activation == "gelu_new":
            x = F.linear(gelu_new(F.linear(x, self.W1) + self.b1), 
                         self.W2) + self.b2
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    """
    """
    def __init__(self, d_model, eps=1e-5):
        """
        """
        super(LayerNorm, self).__init__()
        
        self.eps = eps
        self.d_model = d_model
        self.alpha = nn.Parameter(torch.ones(self.d_model))
        self.bias = nn.Parameter(torch.zeros(self.d_model))
        
        
    def forward(self, x):
        """
        """
        
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, unbiased=False, keepdim=True)
        norm = self.alpha * (x - mean) / (std + self.eps) + self.bias
        return norm


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout=None):
    """
    """
    d = query.size(-1)
    #B x n_heads x L_q x L_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d)
    
    if attn_mask is not None:
        attn_mask = attn_mask.bool()
        scores = scores.masked_fill(attn_mask, -1e9)
    
    attn_scores = F.softmax(scores, dim = -1)
    
    if dropout is not None:
        attn_scores = dropout(attn_scores)
    
    return torch.matmul(attn_scores, value), attn_scores


class MultiHeadAttention(nn.Module):
    """
    """
    def __init__(self, n_heads, d_model, d_qk, d_v, dropout=0, attn_dropout=0):
        """
        """
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = Dropout(dropout)
        self.attn_dropout = Dropout(attn_dropout)
        self.d_qk = d_qk
        self.d_v = d_v
        self.W_q = nn.Parameter(torch.Tensor(n_heads*d_qk, d_model))
        self.W_k = nn.Parameter(torch.Tensor(n_heads*d_qk, d_model))
        self.W_v = nn.Parameter(torch.Tensor(n_heads*d_v, d_model))
        self.W_o = nn.Parameter(torch.Tensor(d_model, n_heads*d_v))
        self.b_q = nn.Parameter(torch.Tensor(n_heads*d_qk))
        self.b_k = nn.Parameter(torch.Tensor(n_heads*d_qk))
        self.b_v = nn.Parameter(torch.Tensor(n_heads*d_v))
        self.b_o = nn.Parameter(torch.Tensor(d_model))
        self.reset_parameters()
        
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.d_model)
        for weight in [self.W_q, self.W_k, self.W_v, self.W_o]:
            weight.data.uniform_(-stdv, stdv)
        for weight in [self.b_q, self.b_k, self.b_v, self.b_o]:
            weight.data.zero_()
    
    
    def forward(self, query, key, value, attn_mask=None, cached_kv=False):
        """
        """
        #B x L x d_model -> B x l x (d*n_heads)
        query = F.linear(query, self.W_q) + self.b_q
        if cached_kv == False:
            key = F.linear(key, self.W_k) + self.b_k
            value = F.linear(value, self.W_v) + self.b_v

        batch_size = query.size(0)
        #B x l x (d*n_heads) -> B x L x n_heads x d_qk
        query = query.view(batch_size, -1, self.n_heads, self.d_qk)
        if cached_kv == False:
            key = key.view(batch_size, -1, self.n_heads, self.d_qk)
            value = value.view(batch_size, -1, self.n_heads, self.d_v)

        #B x L x n_heads x d -> B x n_heads x L x d
        query = query.transpose(1,2)
        if cached_kv == False:
            key = key.transpose(1,2)
            value = value.transpose(1,2)
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        output, attn_scores = scaled_dot_product_attention(query, 
                                                           key, 
                                                           value, 
                                                           attn_mask,
                                                           self.attn_dropout)
        
        #B x n_heads x L x d -> B x L x n_heads x d -> B x L x d_model
        output = output.transpose(1,2)
        output = output.contiguous().view(batch_size, -1, 
                                  self.n_heads*self.d_v)
        output = self.dropout(F.linear(output, self.W_o) + self.b_o)
        return output, attn_scores


    def cache_kv(self, x):
        """
        """
        batch_size = x.size(0)
        
        key = F.linear(x, self.W_k) + self.b_k
        value = F.linear(x, self.W_v) + self.b_v
        
        key = key.view(batch_size, -1, self.n_heads, self.d_qk)
        value = value.view(batch_size, -1, self.n_heads, self.d_v)
        
        key = key.transpose(1,2)
        value = value.transpose(1,2)
        
        return [key, value]


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
    

class EncoderLayer(nn.Module):
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
                 activation="relu"):
        """
        """
        super(EncoderLayer, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = Dropout(dropout)
        self.d = d_model // n_heads

        self.attention = MultiHeadAttention(n_heads,
                                            d_model, 
                                            d_qk, 
                                            d_v, 
                                            dropout,
                                            attn_dropout)
        self.norm_1 = LayerNorm(d_model, ln_eps)
        self.ffn = FeedForward(d_model, d_ff, activation, dropout)
        self.norm_2 = LayerNorm(d_model, ln_eps)
        self.use_pre_norm = use_pre_norm
        
        
    def forward(self, enc_output, attn_mask):
        """
        """
        residual = enc_output
        
        if self.use_pre_norm == True:
            enc_output = self.norm_1(enc_output)
        enc_output, attn_scores = self.attention(enc_output, 
                                                 enc_output, 
                                                 enc_output, 
                                                 attn_mask)
        enc_output = residual + enc_output
        if self.use_pre_norm == False:
            enc_output = self.norm_1(enc_output)
        
        residual = enc_output
        if self.use_pre_norm == True:
            enc_output = self.norm_2(enc_output)
        enc_output = self.ffn(enc_output)
        enc_output = residual + enc_output
        if self.use_pre_norm == False:
            enc_output = self.norm_2(enc_output)
            
        return enc_output, attn_scores


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
                 embedding_size=None,
                 share_layer_params=False, 
                 n_share_across_layers=1,
                 use_pre_norm=True, 
                 activation="relu", 
                 scale_embedding=False,
                 norm_before_pred=False,
                 norm_after_embedding=False,
                 pos_need_train=False,
                 add_segment_embedding=False,
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

        self.pos_embedding = PositionEmbedding(src_max_len, 
                                               d_model, 
                                               pos_need_train)
        
        self.add_segment_embedding = add_segment_embedding
        self.n_types = n_types
        if add_segment_embedding == True:
            self.type_embedding = Embedding(self.n_types, self.d_model)
        
        if self.norm_after_embedding == True:
            self.norm_emb = LayerNorm(self.d_model)
        
        self.emb_dropout = Dropout(emb_dropout)
        
        if share_layer_params == False:
            self.layers = nn.ModuleList([
                    EncoderLayer(n_heads, 
                                 d_model, 
                                 d_ff, 
                                 d_qk, 
                                 d_v,
                                 dropout=dropout,
                                 attn_dropout=attn_dropout,
                                 ln_eps=ln_eps,
                                 use_pre_norm=use_pre_norm,                                 
                                 activation=activation)
                    for _ in range(n_layers)])
        else:
            layers = []
            for i in range(n_layers):
                if i % n_share_across_layers == 0:
                    layer = EncoderLayer(n_heads,
                                         d_model, 
                                         d_ff, 
                                         d_qk, 
                                         d_v,
                                         dropout=dropout, 
                                         attn_dropout=attn_dropout,
                                         ln_eps=ln_eps,
                                         use_pre_norm=use_pre_norm,
                                         activation=activation)
                    layers.append(layer)
            self.layers = nn.ModuleList(layers)
    
        if self.norm_before_pred == True:
            self.norm = LayerNorm(self.d_model)
    
    
    def forward(self, x, attn_mask, x_type=None, return_states=False):
        """
        """
        enc_self_attn_list = []
        
        word_embeded = self.src_embedding(x)

        embeded = word_embeded + self.pos_embedding(x)
        enc_output = embeded
        
        if self.add_segment_embedding == True:
            embeded = embeded + self.type_embedding(x_type)
            enc_output = embeded
           
        if self.norm_after_embedding == True:
            embeded = self.norm_emb(embeded)
            enc_output = embeded
        
        enc_output = self.emb_dropout(enc_output)
        
        enc_states = [enc_output]
        
        for i in range(self.n_layers):
            if self.share_layer_params == False:
                enc_layer = self.layers[i]
            else:
                enc_layer = self.layers[i // self.n_share_across_layers]
                
            enc_output, enc_attn_scores = enc_layer(enc_output,attn_mask)
            
            enc_self_attn_list.append(enc_attn_scores)
            enc_states.append(enc_output)
        
        if self.norm_before_pred == True:
            enc_output = self.norm(enc_output)
        
        outputs = [enc_output]
        if return_states == True:
            outputs = outputs + [embeded, enc_states, enc_self_attn_list]
        
        return outputs


class DecoderLayer(nn.Module):
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
                 activation="relu"):
        """
        """
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(n_heads,
                                                 d_model, 
                                                 d_qk, 
                                                 d_v, 
                                                 dropout,
                                                 attn_dropout)
        self.norm_1 = LayerNorm(d_model,ln_eps)
        self.enc_attention = MultiHeadAttention(n_heads,
                                                d_model, 
                                                d_qk, 
                                                d_v, 
                                                dropout,
                                                attn_dropout)
        self.norm_2 = LayerNorm(d_model,ln_eps)
        self.ffn = FeedForward(d_model, d_ff, activation, dropout)
        self.norm_3 = LayerNorm(d_model,ln_eps)
        self.use_pre_norm = use_pre_norm
        
        
    def forward(self, dec_output, 
                enc_keys, enc_values, self_attn_mask, dec_enc_attn_mask,
                cached_kv=False, dec_keys=None, dec_values=None):
        """
        """
        residual = dec_output
        if self.use_pre_norm == True:
            dec_output = self.norm_1(dec_output)
        
        if cached_kv == True:
            kv = self.cache_dec_kv(dec_output)
            
            if dec_keys is None:
                dec_keys = kv[0]                
            else:
                dec_keys = torch.cat([dec_keys, kv[0]], 2)
            if dec_values is None:
                dec_values = kv[1]
            else:
                dec_values = torch.cat([dec_values, kv[1]], 2)
                
        else:
            dec_keys = dec_output
            dec_values = dec_output
        
        dec_output, self_attn_scores = self.self_attention(dec_output, 
                                                           dec_keys, 
                                                           dec_values, 
                                                           self_attn_mask,
                                                           cached_kv)
        
        dec_output = residual + dec_output
        if self.use_pre_norm == False:
            dec_output = self.norm_1(dec_output)
            
        residual = dec_output
        if self.use_pre_norm == True:
            dec_output = self.norm_2(dec_output)
              
        dec_output, enc_attn_scores = self.enc_attention(dec_output, 
                                                         enc_keys, 
                                                         enc_values, 
                                                         dec_enc_attn_mask,
                                                         cached_kv)
        enc_context = dec_output
        dec_output = residual + dec_output
        if self.use_pre_norm == False:
            dec_output = self.norm_2(dec_output)
            
        residual = dec_output
                
        if self.use_pre_norm == True:
            dec_output = self.norm_3(dec_output)
        dec_output = self.ffn(dec_output)
        dec_output = residual + dec_output
        if self.use_pre_norm == False:
            dec_output = self.norm_3(dec_output)
       
        outputs = [dec_output, self_attn_scores, enc_attn_scores]

        if cached_kv == True:
            outputs = outputs + [dec_keys, dec_values]
        
        outputs = outputs + [enc_context]
        
        return outputs


    def cache_enc_kv(self, enc_output):
        """
        """
        return self.enc_attention.cache_kv(enc_output)


    def cache_dec_kv(self, dec_output):
        """
        """
        return self.self_attention.cache_kv(dec_output)


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
                 dropout=0, 
                 attn_dropout=0,
                 emb_dropout=0,
                 ln_eps=1e-5,
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
                 pos_need_train=False,
                 use_proj_bias=False):
        """
        """
        super(Decoder, self).__init__()
        self.trg_vocab_size = trg_vocab_size
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers
        self.share_layer_params = share_layer_params
        self.n_share_across_layers = n_share_across_layers
        self.scale_embedding = scale_embedding
        self.norm_before_pred = norm_before_pred
        self.norm_after_embedding = norm_after_embedding
        
        if share_src_trg_emb == False:
            self.trg_embedding = Embedding(trg_vocab_size, 
                                           d_model, 
                                           scale_embedding)
            
        self.emb_dropout = Dropout(emb_dropout)
        self.pos_embedding = PositionEmbedding(trg_max_len, 
                                               d_model, 
                                               pos_need_train)
        
        if self.norm_after_embedding == True:
            self.norm_emb = LayerNorm(self.d_model)
        
        if share_layer_params == False:
            self.layers = nn.ModuleList([
                    DecoderLayer(n_heads, 
                                 d_model, 
                                 d_ff, 
                                 d_qk, 
                                 d_v,
                                 dropout=dropout,
                                 attn_dropout=attn_dropout,
                                 ln_eps=ln_eps,
                                 use_pre_norm=use_pre_norm,
                                 activation=activation)
                    for _ in range(n_layers)])
        else:
            layers = []
            for i in range(n_layers):
                if i % n_share_across_layers == 0:
                    layer = DecoderLayer(n_heads,
                                         d_model, 
                                         d_ff, 
                                         d_qk, 
                                         d_v,
                                         dropout=dropout,
                                         attn_dropout=attn_dropout,
                                         ln_eps=ln_eps,
                                         use_pre_norm=use_pre_norm,
                                         activation=activation)
                    layers.append(layer)
            self.layers = nn.ModuleList(layers)
    
        self.share_emb_out_proj = share_emb_out_proj
        if share_emb_out_proj == False: 
            self.W = nn.Parameter(torch.Tensor(trg_vocab_size, d_model))
        self.use_proj_bias = use_proj_bias
        if use_proj_bias == True:
            self.b = nn.Parameter(torch.Tensor(trg_vocab_size))

        if self.norm_before_pred == True:
            self.norm = LayerNorm(d_model)
        
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.d_model)
        if self.share_emb_out_proj == False: 
            self.W.data.uniform_(-stdv, stdv)
        if self.use_proj_bias == True:
            self.b.data.zero_()
            
            
    def forward(self, y, enc_output, self_attn_mask, dec_enc_attn_mask, 
                return_states=False, trg_embedding=None):
        """
        """
        if trg_embedding is None:
            trg_embedding = self.trg_embedding

        word_embeded = trg_embedding(y)
            
        dec_output = word_embeded + self.pos_embedding(y)
        embeded = dec_output
        if self.norm_after_embedding == True:
            dec_output = self.norm_emb(dec_output)
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
                            enc_output, 
                            enc_output, 
                            self_attn_mask, 
                            dec_enc_attn_mask)
            dec_output, self_attn_scores, enc_attn_scores = outputs[:3]

            self_attn_scores_list.append(self_attn_scores)
            enc_attn_scores_list.append(enc_attn_scores)
            dec_states.append(dec_output)

        if self.norm_before_pred == True:
            dec_output = self.norm(dec_output)
            
        if self.share_emb_out_proj == False:
            W = self.W
        else:
            W = trg_embedding.get_embedding()
            
        logits = F.linear(dec_output, W) + self.b
        
        outputs = [logits]
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
    
    
    def _step(self, steps, enc_kv_list, dec_enc_attn_mask, 
              dec_kv_list, y, trg_embedding=None):
        """
        """
        if trg_embedding is None:
            trg_embedding = self.trg_embedding

        word_embeded = trg_embedding(y)
            
        dec_output = word_embeded + self.pos_embedding(y, steps)
        if self.norm_after_embedding == True:
            dec_output = self.norm_emb(dec_output)
        
        dec_enc_attn_list = []
        dec_self_attn_list = []

        for i in range(self.n_layers):
            if self.share_layer_params == False:
                layer = self.layers[i]
            else:
                layer = self.layers[i // self.n_share_across_layers]

            outputs = layer(dec_output,
                            enc_kv_list[i][0], 
                            enc_kv_list[i][1],
                            None,
                            dec_enc_attn_mask,
                            True,
                            dec_kv_list[i][0], 
                            dec_kv_list[i][1]
                            )
            dec_output, self_attn_scores, enc_attn_scores = outputs[:3]

            dec_keys, dec_values = outputs[3:5]
            dec_enc_attn_list.append(enc_attn_scores)
            dec_self_attn_list.append(self_attn_scores)
            dec_kv_list[i][0] = dec_keys
            dec_kv_list[i][1] = dec_values

        if self.norm_before_pred == True:
            dec_output = self.norm(dec_output)
            
        if self.share_emb_out_proj == False:
            W = self.W
        else:
            W = trg_embedding.get_embedding()
            
        logits = F.linear(dec_output, W) + self.b
        
        logits = logits.view(-1, self.trg_vocab_size)
        
        outputs = [dec_kv_list, logits, dec_enc_attn_list, dec_self_attn_list]
        
        return outputs


class LMDecoderLayer(nn.Module):
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
                 activation="relu"):
        """
        """
        super(LMDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(n_heads,
                                                 d_model, 
                                                 d_qk, 
                                                 d_v, 
                                                 dropout,
                                                 attn_dropout)
        self.norm_1 = LayerNorm(d_model, ln_eps)
        self.ffn = FeedForward(d_model, d_ff, activation, dropout)
        self.norm_2 = LayerNorm(d_model, ln_eps)
        self.use_pre_norm = use_pre_norm
        
        
    def forward(self, dec_output, self_attn_mask, 
                cached_kv=False, dec_keys=None, dec_values=None):
        """
        """
        residual = dec_output
        if self.use_pre_norm == True:
            dec_output = self.norm_1(dec_output)
        
        if cached_kv == True:
            kv = self.cache_dec_kv(dec_output)
            if dec_keys is None:
                dec_keys = kv[0]
            else:
                dec_keys = torch.cat([dec_keys, kv[0]], 2)
            if dec_values is None:
                dec_values = kv[1]
            else:
                dec_values = torch.cat([dec_values, kv[1]], 2)
        else:
            dec_keys = dec_output
            dec_values = dec_output
            
        dec_output, self_attn_scores = self.self_attention(dec_output, 
                                                           dec_keys, 
                                                           dec_values, 
                                                           self_attn_mask,
                                                           cached_kv)

        dec_output = residual + dec_output
        if self.use_pre_norm == False:
            dec_output = self.norm_1(dec_output)
            
        residual = dec_output
        if self.use_pre_norm == True:
            dec_output = self.norm_2(dec_output)
        dec_output = self.ffn(dec_output)
        dec_output = residual + dec_output
        if self.use_pre_norm == False:
            dec_output = self.norm_2(dec_output)
            
        outputs = [dec_output, self_attn_scores]
        if cached_kv == True:
            outputs = outputs + [dec_keys, dec_values]
            
        return outputs


    def cache_dec_kv(self, dec_output):
        """
        """
        return self.self_attention.cache_kv(dec_output)


class LMDecoder(nn.Module):
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
                 use_proj_bias=True):
        """
        """
        super(LMDecoder, self).__init__()
        self.trg_vocab_size = trg_vocab_size
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers
        self.share_layer_params = share_layer_params
        self.n_share_across_layers = n_share_across_layers
        self.scale_embedding = scale_embedding
        self.use_pre_norm = use_pre_norm
        self.norm_before_pred = norm_before_pred
        self.norm_after_embedding = norm_after_embedding
        
        if embedding_size is None:
            self.trg_embedding = Embedding(trg_vocab_size, 
                                           d_model, 
                                           scale_embedding)
        else:
            self.trg_embedding = FactorizedEmbedding(trg_vocab_size, 
                                                     embedding_size,
                                                     d_model)
        self.emb_dropout = Dropout(emb_dropout)
        self.pos_embedding = PositionEmbedding(trg_max_len,
                                               d_model,
                                               pos_need_train)
        if self.norm_after_embedding == True:
            self.norm_emb = LayerNorm(self.d_model)
            
        if share_layer_params == False:
            self.layers = nn.ModuleList([
                    LMDecoderLayer(n_heads, 
                                   d_model, 
                                   d_ff, 
                                   d_qk, 
                                   d_v,
                                   dropout=dropout,
                                   attn_dropout=attn_dropout,
                                   ln_eps=ln_eps,
                                   use_pre_norm=use_pre_norm,
                                   activation=activation)
                    for _ in range(n_layers)])
        else:
            layers = []
            for i in range(n_layers):
                if i % n_share_across_layers == 0:
                    layer = LMDecoderLayer(n_heads, 
                                           d_model,
                                           d_ff, 
                                           d_qk,
                                           d_v,
                                           dropout=dropout,
                                           attn_dropout=attn_dropout,
                                           ln_eps=ln_eps,
                                           use_pre_norm=use_pre_norm,
                                           activation=activation)

                    layers.append(layer)
            self.layers = nn.ModuleList(layers)
        
        self.share_emb_out_proj = share_emb_out_proj
        if share_emb_out_proj == False: 
            self.W = nn.Parameter(torch.Tensor(trg_vocab_size, d_model))
        
        self.use_proj_bias = use_proj_bias
        if use_proj_bias == True:
            self.b = nn.Parameter(torch.Tensor(trg_vocab_size))
        
        if self.norm_before_pred == True:
            self.norm = LayerNorm(d_model)
        
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.d_model)
        if self.share_emb_out_proj == False: 
            self.W.data.uniform_(-stdv, stdv)
        if self.use_proj_bias == True:
            self.b.data.zero_()
            
        
    def forward(self, y, self_attn_mask, return_states=False):
        """
        """
        dec_states = []
        
        embeded = self.trg_embedding(y)
            
        dec_output = embeded + self.pos_embedding(y)
        if self.norm_after_embedding == True:
            dec_output = self.norm_emb(dec_output)
        dec_output = self.emb_dropout(dec_output)

        dec_states.append(dec_output)

        self_attn_scores_list = []
        for i in range(self.n_layers):
            if self.share_layer_params == False:
                layer = self.layers[i]
            else:
                layer = self.layers[i // self.n_share_across_layers]
            
            outputs = layer(dec_output, self_attn_mask)
            dec_output, self_attn_scores = outputs[0:2]
            dec_states.append(dec_output)
            self_attn_scores_list.append(self_attn_scores)
            
        
        if self.norm_before_pred == True:
            dec_output = self.norm(dec_output)
            
        if self.share_emb_out_proj == False: 
            W = self.W
        else:
            W = self.trg_embedding.get_embedding()
        logits = F.linear(dec_output, W) 
        if self.use_proj_bias:
            logits = logits + self.b
        
        outputs = [logits]
        if return_states == True:
            outputs.append(dec_states)
            outputs.append(self_attn_scores_list)
            
        return outputs


    def cache_dec_kv(self, dec_output):
        """
        """
        kv_list = []
        for i in range(self.n_layers):
            if self.share_layer_params == False:
                layer = self.layers[i]
            else:
                layer = self.layers[i // self.n_share_across_layers]
            kv_list.append(layer.cache_dec_kv(dec_output))
            
        return kv_list
    
    
    def _step(self, steps, dec_kv_list, y):
        """
        """
        embeded = self.trg_embedding(y)
        
        dec_output = embeded + self.pos_embedding(y, steps)
        if self.norm_after_embedding == True:
            dec_output = self.norm_emb(dec_output)
        
        dec_self_attn_list = []
        
        for i in range(self.n_layers):
            if self.share_layer_params == False:
                layer = self.layers[i]
            else:
                layer = self.layers[i // self.n_share_across_layers]
            
            outputs = layer.forward(dec_output,
                                    None,
                                    True,
                                    dec_kv_list[i][0], 
                                    dec_kv_list[i][1])
            dec_output, self_attn_scores, dec_keys, dec_values = outputs

            dec_kv_list[i][0] = dec_keys
            dec_kv_list[i][1] = dec_values
            dec_self_attn_list.append(self_attn_scores)

        if self.norm_before_pred == True:
            dec_output = self.norm(dec_output)

        if self.share_emb_out_proj == False:
            W = self.W
        else:
            W = self.trg_embedding.get_embedding()
        
        logits = F.linear(dec_output, W) 
        if self.use_proj_bias:
            logits = logits + self.b
        
        logits = logits.view(-1, self.trg_vocab_size)
        
        return dec_kv_list, logits, dec_self_attn_list


if __name__ == "__main__":
    pass