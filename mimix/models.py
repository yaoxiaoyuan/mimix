# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:15:01 2019

@author: Xiaoyuan Yao
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mimix.layers import *
from mimix.utils import real_path

class Transformer(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        """
        """
        super(Transformer, self).__init__()
        self.vocab_size = kwargs.get("vocab_size", None)
        self.max_len = kwargs.get("max_len", None)
        self.n_heads = kwargs["n_heads"] 
        self.d_model = kwargs["d_model"] 
        self.d_ff = kwargs.get("d_ff", 4 * self.d_model) 
        self.d_qk = kwargs.get("d_qk", self.d_model//self.n_heads)
        self.d_v = kwargs.get("d_v", self.d_model//self.n_heads)
        self.n_layers = kwargs["n_layers"]
        self.n_types = kwargs.get("n_types", None)
        self.with_across_attention = kwargs.get("with_across_attention", False)
        self.dropout = kwargs.get("dropout", 0)
        self.attn_dropout = kwargs.get("attn_dropout", 0)
        self.emb_dropout = kwargs.get("emb_dropout", 0)
        self.ln_eps = kwargs.get("ln_eps", 1e-5)
        self.use_rms_norm = kwargs.get("use_rms_norm", False)
        self.use_attention_bias = kwargs.get("use_attention_bias", True)
        self.use_ffn_bias = kwargs.get("use_ffn_bias", True)
        self.n_kv_heads = kwargs.get("n_kv_heads", None)
        self.use_alibi_bias = kwargs.get("use_alibi_bias", False)
        self.max_relative_len = kwargs.get("max_relative_len", -1)
        self.use_rel_pos_value = kwargs.get("use_rel_pos_value", False)
        self.rel_pos_type = kwargs.get("rel_pos_type", "learned")
        self.factorized_size = kwargs.get("factorized_size", None)
        self.share_layer_params = kwargs.get("share_layer_params", False)
        self.n_share_across_layers = kwargs.get("n_share_across_layers", 1)
        self.use_word_embedding = kwargs.get("use_word_embedding", True)
        self.output_next_word_logits = kwargs.get("output_next_word_logits", False)
        self.share_emb_out_proj = kwargs.get("share_emb_out_proj", False)
        self.use_pre_norm = kwargs.get("use_pre_norm", False)
        self.activation = kwargs.get("activation", "relu")
        self.norm_type = kwargs.get("layer_norm_type", "layer_norm")
        self.scale_embedding = kwargs.get("scale_embedding", False)
        self.norm_before_pred = kwargs.get("norm_before_pred", False)
        self.norm_after_embedding = kwargs.get("norm_after_embedding", False)
        self.use_pos_embedding = kwargs.get("use_pos_embedding", True)
        self.pos_type = kwargs.get("pos_type", "learned")
        self.use_output_bias = kwargs.get("use_output_bias", False)
        self.use_talking_attention = kwargs.get("use_talking_attention", False)
        self.use_attention_residual = kwargs.get("use_attention_residual", False)
        self.use_glu = kwargs.get("use_glu", False)
        self.use_moe = kwargs.get("use_moe", False)
        self.use_ln_scale = kwargs.get("use_ln_scale", True)
        self.use_ln_bias = kwargs.get("use_ln_bias", True)
        
        self.use_vit_encoder = kwargs.get("use_vit_encoder", False)
        if self.use_vit_encoder == True:
            img_h,img_w = kwargs["img_h"], kwargs["img_w"]
            ph,pw = kwargs["patch_h"], kwargs["patch_w"]
            self.use_word_embedding = False
            self.max_len = img_h // ph * img_w // pw + 1
            self.patch_embedding = nn.Conv2d(kwargs["n_channels"], self.d_model, kernel_size=(ph, pw), stride=(ph, pw))
            self.cls = nn.Parameter(torch.Tensor(self.d_model))

        if self.use_word_embedding == True:
            if self.factorized_size is None:
                self.word_embedding = Embedding(self.vocab_size, self.d_model)
            else:
                self.word_embedding = FactorizedEmbedding(self.vocab_size, self.d_model, self.factorized_size)
                
        self.emb_dropout = Dropout(self.emb_dropout) if self.emb_dropout else None
        
        if self.use_pos_embedding == True:    
            self.pos_embedding = PositionEmbedding(self.max_len, self.d_model, self.pos_type)
            
        if self.n_types is not None:    
            self.type_embedding = Embedding(self.n_types, self.d_model)
        
        if self.norm_after_embedding == True:       
            if self.norm_type == "layer_norm":
                self.norm_emb = LayerNorm(self.d_model,self.ln_eps,self.use_ln_scale,self.use_ln_bias)
            elif self.norm_type == "rms_norm":
                self.norm_emb = RMSNorm(self.d_model,self.ln_eps,self.use_ln_scale,self.use_ln_bias)
        
        self.layers = nn.ModuleList([TransformerLayer(**kwargs) for i in range(self.n_layers//self.n_share_across_layers)])
   
        if self.output_next_word_logits == True:
            if self.share_emb_out_proj == False: 
                self.W = nn.Parameter(torch.Tensor(self.vocab_size, self.d_model))
            if self.use_output_bias == True:
                self.b = nn.Parameter(torch.Tensor(self.vocab_size))

        if self.norm_before_pred == True:
            if self.norm_type == "layer_norm":
                self.norm = LayerNorm(self.d_model,self.ln_eps,self.use_ln_scale,self.use_ln_bias)
            elif self.norm_type == "rms_norm":
                self.norm = RMSNorm(self.d_model,self.ln_eps,self.use_ln_scale,self.use_ln_bias)
        
        self.reset_parameters()
    
    
    def reset_parameters(self):
        """
        """
        if self.output_next_word_logits == True:
            if self.share_emb_out_proj == False: 
                stdv = 1.0 / np.sqrt(self.vocab_size)
                self.W.data.uniform_(-stdv, stdv)
            if self.use_output_bias == True:
                self.b.data.zero_()
        if self.use_vit_encoder == True:
            self.cls.data.uniform_(0, 1)
            

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore, ids_keep


    def get_mask_image_enc(self, x, mask_ratio):
        """
        """
        embeded = self.patch_embedding(x).flatten(2).transpose(1, 2)
        embeded_masked, mask, ids_restore, ids_keep = self.random_masking(embeded, mask_ratio)
        
        cls = self.cls.repeat(embeded_masked.shape[0], 1, 1)
        cls_pos_ids = torch.zeros([embeded_masked.shape[0], 1], dtype=torch.long, device=ids_keep.device)
        enc_pos_ids = torch.cat([cls_pos_ids, 1+ids_keep], 1)
        embeded_masked = torch.cat([cls, embeded_masked], 1)

        embeded = embeded_masked + self.pos_embedding(enc_pos_ids)
        output = embeded
        self_attention_residual = None
        for i in range(self.n_layers):
            
            if self.share_layer_params == False:
                layer = self.layers[i]
            else:
                layer = self.layers[i // self.n_share_across_layers]
                
            outputs = layer(output,
                            self_pos_ids=enc_pos_ids,
                            past_pos_ids=enc_pos_ids,
                            self_attention_residual=attention_residual if self.use_attention_residual else None)
            
            output = outputs["output"] 
            self_attention_residual = outputs["self_attn_scores"]
            
        if self.norm_before_pred == True:
            output = self.norm(output)
        
        outputs = [output, mask, ids_restore]
        
        return outputs            

        
    def forward(self, 
                x, 
                self_attn_mask=None,
                self_kv_list=None, 
                self_enc_attn_mask=None,
                enc_kv_list=None,
                self_pos_ids=None,
                enc_pos_ids=None,
                past_pos_ids=None,
                cached_kv=False, 
                embedding=None,
                type_ids=None,
                prefix_embeded=None):
        """
        """
        if self.use_vit_encoder == True:
            x = self.patch_embedding(x).flatten(2).transpose(1, 2)
            cls = self.cls.repeat(x.shape[0], 1, 1)
            x = torch.cat([cls, x], 1)

        if self.use_word_embedding == True:
            embedding = self.word_embedding

        embeded = x
        if embedding is not None:
            embeded = embedding(x) 
        
        if prefix_embeded is not None:
            embeded = torch.cat([prefix_embeded, embeded], 1)

        if self.use_pos_embedding == True:   
            embeded = embeded + self.pos_embedding(self_pos_ids)

        if type_ids is not None:
            embeded = embeded + self.type_embedding(type_ids)

        output = embeded
        
        if self.norm_after_embedding == True:
            output = self.norm_emb(output)
        if self.emb_dropout is not None:
            output = self.emb_dropout(output)

        self_attn_weights_list = []
        enc_attn_weights_list = []
        moe_logits_list = []
        moe_weights_list = []
        moe_indices_list = []
        states = [output]
        self_attention_residual = None
        enc_attention_residual = None
        for i in range(self.n_layers):
            
            if self.share_layer_params == False:
                layer = self.layers[i]
            else:
                layer = self.layers[i // self.n_share_across_layers]
                
            outputs = layer(output,
                            self_attn_mask,
                            cached_kv,
                            self_kv_list[i][0] if cached_kv else None,
                            self_kv_list[i][1] if cached_kv else None,
                            enc_kv_list[i][0] if cached_kv and self.with_across_attention else enc_kv_list, 
                            enc_kv_list[i][1] if cached_kv and self.with_across_attention else enc_kv_list, 
                            self_enc_attn_mask,
                            self_pos_ids,
                            enc_pos_ids,
                            past_pos_ids,
                            self_attention_residual if self.use_attention_residual else None,
                            enc_attention_residual if self.use_attention_residual else None)
           

            output = outputs["output"] 
            
            self_attn_weights_list.append(outputs["self_attn_weight"])
            self_attention_residual = outputs["self_attn_scores"]

            if self.with_across_attention == True:
                enc_attn_weights_list.append(outputs["enc_attn_weight"])
                enc_attention_residual = outputs["enc_attn_scores"]

            if self.use_moe == True:
                moe_logits_list.append(outputs["moe_logits"])
                moe_weights_list.append(outputs["moe_weights"])
                moe_indices_list.append(outputs["moe_indices"])

            states.append(output)

            if cached_kv == True:
                self_kv_list[i][0] = outputs["cache_sa_keys"]
                self_kv_list[i][1] = outputs["cache_sa_values"]

        if self.norm_before_pred == True:
            output = self.norm(output)

        outputs = {"output":output}
        if self.output_next_word_logits == True:            
            if self.share_emb_out_proj == False:
                W = self.W
            else:
                W = embedding.get_embedding()
            
            logits = F.linear(output, W)
            if self.use_output_bias == True:
                logits = logits + self.b
        
            outputs["logits"] = logits
        if cached_kv == True:
            outputs["cache_kv"] = self_kv_list

        
        outputs["embeded"] = embeded
        outputs["states"] = states
        outputs["self_attn_weights_list"] = self_attn_weights_list
        outputs["enc_attn_weights_list"] = enc_attn_weights_list
        outputs["moe_weights_list"] = moe_weights_list
        outputs["moe_indices_list"] = moe_indices_list
        outputs["moe_logits_list"] = moe_logits_list
            
        return outputs


    def cache_enc_kv(self, enc_output, pos_ids=None):
        """
        """
        kv_list = []
        for i in range(self.n_layers):
            if not self.share_layer_params:
                layer = self.layers[i]
                cached_kv = layer.cache_enc_kv(enc_output, pos_ids)
            elif i % self.n_share_across_layers == 0:
                layer = self.layers[i // self.n_share_across_layers]
                cached_kv = layer.cache_enc_kv(enc_output, pos_ids)
            else:
                cached_kv = kv_list[-1]
            kv_list.append(cached_kv)
            
        return kv_list
    

    def cache_dec_kv(self, 
                     y=None, 
                     self_attn_mask=None, 
                     enc_kv_list=None, 
                     self_enc_attn_mask=None, 
                     self_pos_ids=None, 
                     enc_pos_ids=None,
                     past_pos_ids=None,
                     trg_embedding=None,
                     prefix_embeded=None):
        """
        """
        self_kv_list = []
        for i in range(self.n_layers):
            self_kv_list.append([None, None])
        if y is None:
            return self_kv_list
        
        if trg_embedding is None:
            trg_embedding = self.word_embedding
        word_embeded = trg_embedding(y)
        if prefix_embeded is not None:
            word_embeded = torch.cat([prefix_embeded, word_embeded], 1)
        if self.use_pos_embedding == True:
            word_embeded = word_embeded + self.pos_embedding(self_pos_ids)
        self_output = word_embeded
        
        if self.norm_after_embedding == True:
            self_output = self.norm_emb(self_output)

        self_attention_residual = None
        enc_attention_residual = None
        for i in range(self.n_layers):
            if self.share_layer_params == False:
                layer = self.layers[i]
            else:
                layer = self.layers[i // self.n_share_across_layers]

            outputs = layer(self_output,
                            self_attn_mask,
                            True,
                            self_kv_list[i][0],
                            self_kv_list[i][1],
                            enc_kv_list[i][0] if self.with_across_attention else None,
                            enc_kv_list[i][1] if self.with_across_attention else None,
                            self_enc_attn_mask,
                            self_pos_ids,
                            enc_pos_ids,
                            past_pos_ids,
                            self_attention_residual if self.use_attention_residual else None,
                            enc_attention_residual if self.use_attention_residual else None
                            )
             
            self_output = outputs["output"]
            self_attention_residual = outputs["self_attn_scores"]
            if self.with_across_attention == True:
                enc_attention_residual = outputs["enc_attn_scores"]

            self_kv_list[i][0] = outputs["cache_sa_keys"]
            self_kv_list[i][1] = outputs["cache_sa_values"]

        return self_kv_list


class TransformerSeq2seq(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        """
        """
        super(TransformerSeq2seq, self).__init__()
        self.PAD = kwargs["symbol2id"]["_pad_"]
        self.BOS = kwargs["symbol2id"]["_bos_"]
        self.EOS = kwargs["symbol2id"]["_eos_"]
        self.UNK = kwargs["symbol2id"]["_unk_"]
        self.SEP = kwargs["symbol2id"]["_sep_"]
        self.CLS = kwargs["symbol2id"]["_cls_"]
        self.MASK = kwargs["symbol2id"]["_mask_"]
        self.MAX_LOGITS = 10000.
        self.MIN_LOGITS = -10000.
                
        self.use_vit_encoder = kwargs.get("use_vit_encoder", False)
        enc_config = kwargs.copy()
        if self.use_vit_encoder == False:
            enc_config["vocab_size"] = kwargs["src_vocab_size"]
            enc_config["max_len"] = kwargs["src_max_len"]
        enc_config["n_layers"] = kwargs["n_enc_layers"]
        self.encoder = Transformer(**enc_config)
        self.src_max_len = self.encoder.max_len

        dec_config = kwargs.copy()
        dec_config["use_vit_encoder"] = False
        dec_config["vocab_size"] = kwargs["trg_vocab_size"]
        dec_config["max_len"] = kwargs["trg_max_len"]
        dec_config["output_next_word_logits"] = True
        dec_config["with_across_attention"] = True
        dec_config["n_layers"] = kwargs["n_dec_layers"]
        if kwargs.get("share_src_trg_emb", False) == True:
            dec_config["use_word_embedding"] = False
        self.decoder = Transformer(**dec_config)
        self.share_src_trg_emb = kwargs.get("share_src_trg_emb", False)
        self.n_dec_layers = kwargs["n_dec_layers"]
        self.trg_vocab_size = kwargs["trg_vocab_size"]
        self.trg_max_len = kwargs["trg_vocab_size"]
        
    
    def get_seq_mask(self, seq, seq_lens=None):
        """
        """
        mask = None
        if seq_lens is None:
            if seq.dtype == torch.long:
                mask = seq.ne(self.PAD)
            return mask
        bz, len_seq = seq.size(0), seq.size(1)
        mask = (torch.arange(len_seq).unsqueeze(0).repeat(bz, 1) < seq_lens).int()
        mask = mask.to(seq.device)
        return mask

    
    def get_attn_mask(self, seq_query, seq_key):
        """
        """
        if seq_query is None or seq_key is None:
            return None
        
        len_query = seq_query.size(1)
        mask = seq_key.eq(self.PAD).byte()
        mask = mask.unsqueeze(1).repeat(1, len_query, 1)
        
        return mask
    
    
    def get_subsequent_mask(self, seq):
        """
        """
        len_seq = seq.size(1)
        mask = torch.triu(torch.ones(len_seq, 
                                     len_seq, 
                                     device=seq.device, 
                                     dtype=torch.uint8), 
                          diagonal=1)

        return mask
        
    
    def forward(self, inputs, targets=None, compute_loss=False):
        """
        """
        x, y = inputs["x"], inputs["y"]
        
        x_mask = self.get_seq_mask(x, inputs.get("x_len", None))
        y_mask = self.get_seq_mask(y)
        
        enc_self_attn_mask = self.get_attn_mask(x_mask, x_mask)
        
        dec_self_attn_mask = self.get_subsequent_mask(y_mask)
        dec_self_attn_mask = dec_self_attn_mask | self.get_attn_mask(y_mask, y_mask)
        
        dec_enc_attn_mask = self.get_attn_mask(y_mask, x_mask)
        
        if self.use_vit_encoder == True:
            enc_pos_ids = torch.arange(self.src_max_len).repeat([x.size(0), 1]).to(x.device)
        else:
            enc_pos_ids = torch.arange(x.size(1)).repeat([x.size(0), 1]).to(x.device)
        dec_pos_ids = y.ne(self.PAD).cumsum(-1) - 1       
                    
        enc_outputs = self.encoder(x, 
                                   self_attn_mask=enc_self_attn_mask,
                                   self_pos_ids=enc_pos_ids,
                                   past_pos_ids=enc_pos_ids)
        
        enc_output = enc_outputs["output"]

        trg_embedding = None
        if self.share_src_trg_emb == True:
            trg_embedding = self.encoder.word_embedding

        dec_outputs = self.decoder(y, 
                                   self_attn_mask=dec_self_attn_mask, 
                                   enc_kv_list=enc_output,
                                   self_enc_attn_mask=dec_enc_attn_mask,
                                   self_pos_ids=dec_pos_ids,
                                   enc_pos_ids=enc_pos_ids,
                                   past_pos_ids=dec_pos_ids,
                                   embedding=trg_embedding)
        
        outputs = dec_outputs
        outputs["enc_outputs"] = enc_outputs
        
        if compute_loss == True:
            loss = self.loss_fn(outputs, targets)
            outputs["loss"] = loss

        return outputs


    def init_search(self, states, inputs):
        """
        """
        x = inputs["x"]
        
        x_mask = self.get_seq_mask(x, inputs.get("x_len", None))
        enc_self_attn_mask = self.get_attn_mask(x_mask, x_mask)
        if self.use_vit_encoder == True:
            enc_pos_ids = torch.arange(self.src_max_len).repeat([x.size(0), 1]).to(x.device)
        else:
            enc_pos_ids = torch.arange(x.size(1)).repeat([x.size(0), 1]).to(x.device)
        enc_outputs = self.encoder(x, 
                                   self_attn_mask=enc_self_attn_mask, 
                                   self_pos_ids=enc_pos_ids,
                                   past_pos_ids=enc_pos_ids)      
        enc_output = enc_outputs["output"]
        enc_kv_list = self.decoder.cache_enc_kv(enc_output, enc_pos_ids)
        
        dec_pos_ids = states[0].ne(self.PAD).cumsum(-1) - 1         
        dec_kv_list = self.decoder.cache_dec_kv()
        past_pos_ids = dec_pos_ids
        
        prefix = inputs.get("y", None)
        if prefix is not None:
            y = prefix[:, :-1].clone()
            states[0] = prefix[:,-1][:,None].clone() 
            states[4] = prefix.clone()
            
            dec_pos_ids = y.ne(self.PAD).cumsum(-1) - 1 
            
            y_mask = self.get_seq_mask(y)
            
            dec_self_attn_mask = self.get_subsequent_mask(y_mask)            
            dec_self_attn_mask = dec_self_attn_mask | self.get_attn_mask(y_mask, y_mask)
            
            dec_enc_attn_mask = self.get_attn_mask(y_mask, x_mask)
            
            trg_embedding = None
            if self.share_src_trg_emb == True:
                trg_embedding = self.encoder.word_embedding
            
            dec_kv_list = self.decoder.cache_dec_kv(y, 
                                                    dec_self_attn_mask, 
                                                    enc_kv_list, 
                                                    dec_enc_attn_mask, 
                                                    dec_pos_ids, 
                                                    enc_pos_ids,
                                                    past_pos_ids,
                                                    trg_embedding)         
            past_pos_ids = dec_pos_ids
            dec_pos_ids = dec_pos_ids[:,-1][:,None] + 1 
            past_pos_ids = torch.cat([past_pos_ids, dec_pos_ids], -1)
        
        y_mask = self.get_seq_mask(states[0])
        dec_enc_attn_mask = self.get_attn_mask(y_mask, x_mask)
        
        cache = [enc_kv_list, dec_kv_list, dec_enc_attn_mask, enc_pos_ids, dec_pos_ids, past_pos_ids]
        
        return states, cache

    
    def step(self, states, cache):
        """
        """
        y = states[0]
        
        enc_kv_list, dec_kv_list, dec_enc_attn_mask, enc_pos_ids, dec_pos_ids, past_pos_ids = cache
        
        trg_embedding = None
        if self.share_src_trg_emb == True:
            trg_embedding = self.encoder.word_embedding
        
        dec_self_attn_mask = self.get_attn_mask(y, states[4]) 
      
        outputs = self.decoder(y,
                               dec_self_attn_mask,
                               dec_kv_list,
                               dec_enc_attn_mask,
                               enc_kv_list,
                               self_pos_ids=dec_pos_ids,
                               enc_pos_ids=enc_pos_ids,
                               past_pos_ids=past_pos_ids,
                               cached_kv=True,
                               embedding=trg_embedding)
                
        logits, dec_kv_list = outputs["logits"], outputs["cache_kv"]
        dec_pos_ids = dec_pos_ids + 1
        past_pos_ids = torch.cat([past_pos_ids, dec_pos_ids], -1)
        
        cache = [enc_kv_list, dec_kv_list, dec_enc_attn_mask, enc_pos_ids, dec_pos_ids, past_pos_ids]
        return logits, cache


    def gather_cache(self, cache, beam_id):
        """
        """
        enc_kv_list, dec_kv_list, dec_enc_attn_mask, enc_pos_ids, dec_pos_ids, past_pos_ids = cache

        for i in range(self.n_dec_layers):
            if enc_kv_list is not None:
                enc_kv_list[i][0] = enc_kv_list[i][0][beam_id]
                enc_kv_list[i][1] = enc_kv_list[i][1][beam_id]

            dec_kv_list[i][0] = dec_kv_list[i][0][beam_id]
            dec_kv_list[i][1] = dec_kv_list[i][1][beam_id]
        
        if dec_enc_attn_mask is not None:
            dec_enc_attn_mask = dec_enc_attn_mask[beam_id]

        enc_pos_ids = enc_pos_ids[beam_id]
        
        dec_pos_ids = dec_pos_ids[beam_id]

        past_pos_ids = past_pos_ids[beam_id]

        cache = [enc_kv_list, dec_kv_list, dec_enc_attn_mask, enc_pos_ids, dec_pos_ids, past_pos_ids]
        return cache


class TransformerBridgeSeq2seq(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        """
        """
        super(TransformerBridgeSeq2seq, self).__init__()
        self.PAD = kwargs["symbol2id"]["_pad_"]
        self.BOS = kwargs["symbol2id"]["_bos_"]
        self.EOS = kwargs["symbol2id"]["_eos_"]
        self.UNK = kwargs["symbol2id"]["_unk_"]
        self.SEP = kwargs["symbol2id"]["_sep_"]
        self.CLS = kwargs["symbol2id"]["_cls_"]
        self.MASK = kwargs["symbol2id"]["_mask_"]
        self.MAX_LOGITS = 10000.
        self.MIN_LOGITS = -10000.

        self.use_vit_encoder = kwargs.get("use_vit_encoder", False)
        enc_config = kwargs.copy()
        if self.use_vit_encoder == False:
            enc_config["vocab_size"] = kwargs["src_vocab_size"]
            enc_config["max_len"] = kwargs["src_max_len"]
        enc_config["n_layers"] = kwargs["n_enc_layers"]
        self.encoder = Transformer(**enc_config)
        self.src_max_len = self.encoder.max_len

        dec_config = kwargs.copy()
        dec_config["use_vit_encoder"] = False
        dec_config["vocab_size"] = kwargs["trg_vocab_size"]
        dec_config["max_len"] = kwargs["trg_max_len"]
        dec_config["output_next_word_logits"] = True
        dec_config["n_layers"] = kwargs["n_dec_layers"]
        if kwargs.get("share_src_trg_emb", False) == True:
            dec_config["use_word_embedding"] = False
        self.decoder = Transformer(**dec_config)
        self.share_src_trg_emb = kwargs.get("share_src_trg_emb", False)
        self.n_dec_layers = kwargs["n_dec_layers"]
        self.trg_vocab_size = kwargs["trg_vocab_size"]
        self.trg_max_len = kwargs["trg_vocab_size"]

        bridge_type = kwargs.get("bridge_type", "mlp")        
        self.n_bridge_layers = kwargs.get("n_bridge_layers", 1)
        if bridge_type == "mlp":
            self.bridge = nn.ModuleList([Linear(self.d_model, self.d_model) for _ in range(self.n_bridge_layers)])


    def get_seq_mask(self, seq, seq_lens=None):
        """
        """
        mask = None
        if seq_lens is None:
            if seq.dtype == torch.long:
                mask = seq.ne(self.PAD)
            return mask
        bz, len_seq = seq.size(0), seq.size(1)
        mask = (torch.arange(len_seq-1, -1, -1).unsqueeze(0).repeat(bz, 1) < seq_lens).int()
        mask = mask.to(seq.device)
        return mask


    def get_attn_mask(self, seq_query, seq_key):
        """
        """
        if seq_query is None or seq_key is None:
            return None

        len_query = seq_query.size(1)
        mask = seq_key.eq(self.PAD).byte()
        mask = mask.unsqueeze(1).repeat(1, len_query, 1)

        return mask


    def get_subsequent_mask(self, seq):
        """
        """
        len_seq = seq.size(1)
        mask = torch.triu(torch.ones(len_seq,
                                     len_seq,
                                     device=seq.device,
                                     dtype=torch.uint8),
                          diagonal=1)

        return mask


    def forward(self, inputs, targets=None, compute_loss=False):
        """
        """
        x, y = inputs["x"], inputs["y"]

        x_mask = self.get_seq_mask(x, inputs.get("x_len", None))
        y_mask = self.get_seq_mask(y)
        xy_mask = torch.cat([x_mask, y_mask], 1)

        enc_self_attn_mask = self.get_attn_mask(x_mask, x_mask)
        dec_self_attn_mask = self.get_subsequent_mask(xy_mask)
        dec_self_attn_mask = dec_self_attn_mask | self.get_attn_mask(y_mask, y_mask)
        
        if self.use_vit_encoder == True:
            enc_pos_ids = torch.arange(self.src_max_len).repeat([x.size(0), 1]).to(x.device)
        else:
            enc_pos_ids = torch.arange(x.size(1)).repeat([x.size(0), 1]).to(x.device)
        dec_pos_ids = xy.ne(self.PAD).cumsum(-1) - 1

        enc_outputs = self.encoder(x,
                                   self_attn_mask=enc_self_attn_mask,
                                   self_pos_ids=enc_pos_ids,
                                   past_pos_ids=enc_pos_ids)
        enc_output = enc_outputs["output"]

        prefix_embeded = enc_output
        for i in range(self.n_bridge_layers):
            prefix_embeded = torch.tanh(self.bridge[i](prefix_embeded))

        trg_embedding = None
        if self.share_src_trg_emb == True:
            trg_embedding = self.encoder.word_embedding

        dec_outputs = self.decoder(y,
                                   self_attn_mask=dec_self_attn_mask,
                                   enc_kv_list=enc_output,
                                   self_enc_attn_mask=dec_enc_attn_mask,
                                   self_pos_ids=dec_pos_ids,
                                   enc_pos_ids=enc_pos_ids,
                                   past_pos_ids=dec_pos_ids,
                                   embedding=trg_embedding,
                                   prefix_embeded=prefix_embeded)

        outputs = dec_outputs
        outputs["logits"] = outputs["logits"][:,-y.size(1):,:]
        outputs["enc_outputs"] = enc_outputs

        if compute_loss == True:
            loss = self.loss_fn(outputs, targets)
            outputs["loss"] = loss

        return outputs        


    def init_search(self, states, inputs):
        """
        """
        x = inputs["x"]

        x_mask = self.get_seq_mask(x, inputs.get("x_len", None))
        enc_self_attn_mask = self.get_attn_mask(x_mask, x_mask)
        if self.use_vit_encoder == True:
            enc_pos_ids = torch.arange(self.src_max_len).repeat([x.size(0), 1]).to(x.device)
        else:
            enc_pos_ids = torch.arange(x.size(1)).repeat([x.size(0), 1]).to(x.device)
        enc_outputs = self.encoder(x,
                                   self_attn_mask=enc_self_attn_mask,
                                   self_pos_ids=enc_pos_ids,
                                   past_pos_ids=enc_pos_ids)
        enc_output = enc_outputs["output"]

        prefix_embeded = enc_output
        for i in range(self.n_bridge_layers):
            prefix_embeded = torch.tanh(self.bridge[i](prefix_embeded))

        x_mask = x_mask.long()

        y = torch.tensor([], dtype=torch.long)
        prefix = inputs.get("y", None)
        if prefix is not None:
            y = prefix[:,:-1].clone()
            states[0] = prefix[:,-1][:,None].clone()
            states[4] = prefix.clone()        
        y = torch.cat([x_mask, y], 1)

        self_pos_ids = y.ne(self.PAD).cumsum(-1) - 1
        dec_self_attn_mask = self.get_subsequent_mask(y)
        dec_self_attn_mask = dec_self_attn_mask | self.get_attn_mask(y, y)

        dec_kv_list = self.decoder.cache_dec_kv(y=y,
                                                self_attn_mask=dec_self_attn_mask,
                                                self_pos_ids=self_pos_ids,
                                                past_pos_ids=past_pos_ids)

        past_pos_ids = self_pos_ids
        self_pos_ids = self_pos_ids[:,-1][:,None] + 1
        past_pos_ids = torch.cat([past_pos_ids, self_pos_ids], -1)

        cache = [dec_kv_list, self_pos_ids, past_pos_ids]

        return states, cache


    def step(self, states, cache):
        """
        """
        dec_kv_list, self_pos_ids, past_pos_ids = cache
        y = states[0]

        dec_self_attn_mask = self.get_attn_mask(y, states[4])
        outputs = self.decoder(y,
                               self_attn_mask=dec_self_attn_mask,
                               self_kv_list=dec_kv_list,
                               self_pos_ids=self_pos_ids,
                               past_pos_ids=past_pos_ids,
                               cached_kv=True)
        logits,dec_kv_list = outputs["logits"], outputs["cache_kv"]

        self_pos_ids = self_pos_ids + 1
        past_pos_ids = torch.cat([past_pos_ids, self_pos_ids], -1)
        cache = [dec_kv_list, self_pos_ids, past_pos_ids]

        return logits, cache


    def gather_cache(self, cache, beam_id):
        """
        """
        dec_kv_list,self_pos_ids,past_pos_ids = cache

        for i in range(self.n_layers):

            dec_kv_list[i][0] = dec_kv_list[i][0][beam_id]
            dec_kv_list[i][1] = dec_kv_list[i][1][beam_id]

        self_pos_ids = self_pos_ids[beam_id]
        past_pos_ids = past_pos_ids[beam_id]

        cache = [dec_kv_list, self_pos_ids, past_pos_ids]

        return cache


class TransformerLM(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        """
        """
        super(TransformerLM, self).__init__()
        self.PAD = kwargs["symbol2id"]["_pad_"]
        self.BOS = kwargs["symbol2id"]["_bos_"]
        self.EOS = kwargs["symbol2id"]["_eos_"]
        self.UNK = kwargs["symbol2id"]["_unk_"]
        self.SEP = kwargs["symbol2id"]["_sep_"]
        self.CLS = kwargs["symbol2id"]["_cls_"]
        self.MASK = kwargs["symbol2id"]["_mask_"]
        self.MAX_LOGITS = 10000.
        self.MIN_LOGITS = -10000.
        self.trg_vocab_size = kwargs["trg_vocab_size"]
        self.n_layers = kwargs["n_dec_layers"]
        dec_config = kwargs.copy()
        dec_config["max_len"] = kwargs["trg_max_len"]
        dec_config["vocab_size"] = kwargs["trg_vocab_size"]
        dec_config["n_layers"] = kwargs["n_dec_layers"]
        dec_config["output_next_word_logits"] = True
        self.decoder = Transformer(**dec_config)   

 
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
        
    
    def forward(self, inputs, targets=None, compute_loss=False):
        """
        """
        y = inputs["y"]
        
        dec_self_attn_mask = inputs.get("attn_mask", None)
        if dec_self_attn_mask is None:
            dec_self_attn_mask = self.get_subsequent_mask(y)
            dec_self_attn_mask = dec_self_attn_mask | self.get_attn_mask(y, y)            
        
        self_pos_ids = inputs.get("pos_ids", None)
        if self_pos_ids is None:
            self_pos_ids = y.ne(self.PAD).cumsum(-1) - 1
        
        dec_outputs = self.decoder(y, 
                                   self_attn_mask=dec_self_attn_mask,
                                   self_pos_ids=self_pos_ids,
                                   past_pos_ids=self_pos_ids)

        outputs = dec_outputs
       
        if compute_loss == True:
            loss = self.loss_fn(outputs, targets)
            outputs["loss"] = loss

        return outputs


    def init_search(self, states, inputs):
        """
        """
        self_pos_ids = states[0].ne(self.PAD).cumsum(-1) - 1
        dec_kv_list = self.decoder.cache_dec_kv()
        past_pos_ids = self_pos_ids
        
        prefix = inputs.get("y", None)
        if prefix is not None:
            y = prefix[:,:-1].clone() 
            states[0] = prefix[:,-1][:,None].clone() 
            states[4] = prefix.clone() 
            
            self_pos_ids = y.ne(self.PAD).cumsum(-1) - 1
            
            dec_self_attn_mask = self.get_subsequent_mask(y)

            dec_self_attn_mask = dec_self_attn_mask | self.get_attn_mask(y, y)
            
            dec_kv_list = self.decoder.cache_dec_kv(y=y,
                                                    self_attn_mask=dec_self_attn_mask,
                                                    self_pos_ids=self_pos_ids,
                                                    past_pos_ids=past_pos_ids)

            past_pos_ids = self_pos_ids
            self_pos_ids = self_pos_ids[:,-1][:,None] + 1
            past_pos_ids = torch.cat([past_pos_ids, self_pos_ids], -1)
            
        cache = [dec_kv_list, self_pos_ids, past_pos_ids]

        return states, cache
    

    def step(self, states, cache):
        """
        """
        dec_kv_list, self_pos_ids, past_pos_ids = cache
        y = states[0]
        
        dec_self_attn_mask = self.get_attn_mask(y, states[4])
        outputs = self.decoder(y, 
                               self_attn_mask=dec_self_attn_mask, 
                               self_kv_list=dec_kv_list,
                               self_pos_ids=self_pos_ids,
                               past_pos_ids=past_pos_ids,
                               cached_kv=True)
        logits,dec_kv_list = outputs["logits"], outputs["cache_kv"]
        
        self_pos_ids = self_pos_ids + 1
        past_pos_ids = torch.cat([past_pos_ids, self_pos_ids], -1)
        cache = [dec_kv_list, self_pos_ids, past_pos_ids]

        return logits, cache


    def gather_cache(self, cache, beam_id):
        """
        """
        dec_kv_list,self_pos_ids,past_pos_ids = cache

        for i in range(self.n_layers):
            
            dec_kv_list[i][0] = dec_kv_list[i][0][beam_id]
            dec_kv_list[i][1] = dec_kv_list[i][1][beam_id]
        
        self_pos_ids = self_pos_ids[beam_id] 
        past_pos_ids = past_pos_ids[beam_id]
               
        cache = [dec_kv_list, self_pos_ids, past_pos_ids]
        
        return cache

    
class TransformerEncoder(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        """
        """
        super(TransformerEncoder, self).__init__()
        self.use_vit_encoder = kwargs.get("use_vit_encoder", False)
        if self.use_vit_encoder == False:
            self.PAD = kwargs["symbol2id"]["_pad_"]
            self.BOS = kwargs["symbol2id"]["_bos_"]
            self.EOS = kwargs["symbol2id"]["_eos_"]
            self.UNK = kwargs["symbol2id"]["_unk_"]
            self.SEP = kwargs["symbol2id"]["_sep_"]
            self.CLS = kwargs["symbol2id"]["_cls_"]
            self.MASK = kwargs["symbol2id"]["_mask_"]
        self.MAX_LOGITS = 10000.
        self.MIN_LOGITS = -10000.
        
        self.src_vocab_size = kwargs.get("src_vocab_size", None)
        self.d_model = kwargs["d_model"]
        
        enc_config = kwargs.copy()
        
        if self.use_vit_encoder == True:
            img_h,img_w = kwargs["img_h"], kwargs["img_w"]
            ph,pw = kwargs["patch_h"], kwargs["patch_w"]
            enc_config["use_word_embedding"] = False
            enc_config["max_len"] = img_h // ph * img_w // pw + 1
            enc_config["n_layers"] = kwargs["n_enc_layers"]
        else:
            enc_config["vocab_size"] = kwargs["src_vocab_size"]
            enc_config["max_len"] = kwargs["src_max_len"]
            enc_config["n_layers"] = kwargs["n_enc_layers"]
        
        self.encoder = Transformer(**enc_config)
        self.src_max_len = self.encoder.max_len

        self.n_types = kwargs.get("n_types", None) 
        self.n_layers = kwargs["n_enc_layers"] 
        self.n_class = kwargs.get("n_class", None)
        
        self.activation = kwargs.get("activation", "relu")
       
        self.W_pool = None
        self.b_pool = None
        self.use_pooling = kwargs.get("use_pooling", False)
        if self.use_pooling == True:
            self.W_pool = nn.Parameter(torch.Tensor(self.d_model, self.d_model))
            self.b_pool = nn.Parameter(torch.zeros(self.d_model))
        
        self.W_cls = None
        self.b_cls = None
        if self.n_class is not None:
            self.W_cls = nn.Parameter(torch.Tensor(self.d_model, self.n_class))
            self.b_cls = nn.Parameter(torch.zeros(self.n_class))

        self.W_out = None
        self.b_out = None        
        self.out_dim = kwargs.get("out_dim", None)
        if self.out_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(self.d_model, self.out_dim))
            self.b_out = nn.Parameter(torch.zeros(self.out_dim))
        
        self.crf = None
        if self.n_class is not None and kwargs.get("use_crf", False) == True:
            self.crf = CRF(self.n_class)
        
        self.W_mlm = None
        self.b_mlm = None
        self.W_out_mlm = None
        self.b_out_mlm = None
        self.with_mlm = kwargs.get("with_mlm", False)
        self.share_emb_out_proj = kwargs.get("share_emb_out_proj", False)
        if self.with_mlm == True:
            self.norm_type = kwargs.get("layer_norm_type", "layer_norm")
            self.ln_eps = kwargs.get("ln_eps", 1e-5)
            self.use_ln_scale = kwargs.get("use_ln_scale", True)
            self.use_ln_bias = kwargs.get("use_ln_bias", True)
            self.W_mlm = nn.Parameter(torch.Tensor(self.d_model, self.d_model))
            self.b_mlm = nn.Parameter(torch.Tensor(self.d_model))
            if self.norm_type == "layer_norm":
                self.norm_mlm = LayerNorm(self.d_model,self.ln_eps,self.use_ln_scale,self.use_ln_bias)
            elif self.norm_type == "rms_norm":
                self.norm_mlm = RMSNorm(self.d_model,self.ln_eps,self.use_ln_scale,self.use_ln_bias)
                
            self.share_emb_out_proj = kwargs.get("share_emb_out_proj", False)
            if self.share_emb_out_proj == False:
                self.W_out_mlm = nn.Parameter(torch.Tensor(self.d_model, self.src_vocab_size))
            self.b_out_mlm = nn.Parameter(torch.Tensor(self.src_vocab_size))

        self.reset_parameters()
    
    
    def reset_parameters(self):
        """
        """
        stdv = 1.0 / np.sqrt(self.d_model)
        for weight in [self.W_pool, self.W_cls, self.W_out, self.W_mlm, self.W_out_mlm]:
            if weight is not None:
                weight.data.uniform_(-math.sqrt(3)*stdv, math.sqrt(3)*stdv)
        for weight in [self.b_pool, self.b_cls, self.b_out, self.b_mlm, self.b_out_mlm]:
            if weight is not None:
                weight.data.zero_()
                

    def get_seq_mask(self, seq, seq_lens=None):
        """
        """
        mask = None
        if seq_lens is None:
            if seq.dtype == torch.long:
                mask = seq.ne(self.PAD)
            return mask
        bz, len_seq = seq.size(0), seq.size(1)
        mask = (torch.arange(len_seq).unsqueeze(0).repeat(bz, 1) < seq_lens).int()
        mask = mask.to(seq.device)
        return mask

                
    def get_attn_mask(self, seq_query, seq_key):
        """
        """
        if seq_query is None or seq_key is None:
            return None
        
        len_query = seq_query.size(1)
        mask = seq_key.eq(self.PAD)
        mask = mask.unsqueeze(1).repeat(1, len_query, 1)
        
        return mask
    
    
    def get_emission(self, x):
        """
        """
        x_mask = self.get_seq_mask(x, inputs.get("x_len", None))
        enc_self_attn_mask = self.get_attn_mask(x_mask, x_mask)
        if self.use_vit_encoder == True:
            enc_pos_ids = torch.arange(self.src_max_len).repeat([x.size(0), 1]).to(x.device)
        else:
            enc_pos_ids = torch.arange(x.size(1)).repeat([x.size(0), 1]).to(x.device)
        
        enc_outputs = self.encoder(x, enc_self_attn_mask, self_pos_ids=self_pos_ids)
        enc_output = enc_outputs["output"]
        
        if self.use_pooling == True:
            enc_output = torch.tanh(torch.matmul(enc_output, self.W_pool) + self.b_pool)
            
        enc_output = torch.matmul(enc_output, self.W_cls) + self.b_cls

        return enc_output     

    
    def forward(self, inputs, targets=None, compute_loss=False):
        """
        """
        x = inputs["x"]
        type_ids = None
        if self.n_types is not None:
            type_ids = inputs["type_ids"]

        x_mask = self.get_seq_mask(x, inputs.get("x_len", None))
        enc_self_attn_mask = self.get_attn_mask(x_mask, x_mask)
        if self.use_vit_encoder == True:
            enc_pos_ids = torch.arange(self.src_max_len).repeat([x.size(0), 1]).to(x.device)
        else:
            enc_pos_ids = torch.arange(x.size(1)).repeat([x.size(0), 1]).to(x.device)
        enc_outputs = self.encoder(x, 
                                   enc_self_attn_mask,
                                   self_pos_ids=enc_pos_ids, 
                                   past_pos_ids=enc_pos_ids, 
                                   type_ids=type_ids) 

        enc_output = enc_outputs["output"]   
        mlm_enc_output = enc_output
        
        outputs = enc_outputs
        if self.use_pooling == True:
            
            enc_output = torch.tanh(torch.matmul(enc_output, self.W_pool) + self.b_pool)
            
            outputs["output"] = enc_output[:,0,:]
            
        if self.out_dim is not None:
            enc_output = torch.matmul(enc_output, self.W_out) + self.b_out

            outputs["output"] = enc_output[:,0,:]
        
        if self.n_class is not None:
            enc_output = torch.matmul(enc_output, self.W_cls) + self.b_cls

            outputs["cls_output"] = enc_output
            outputs["cls_logits"] = enc_output[:,0,:] 
                
        if self.with_mlm == True:
            enc_output = F.linear(mlm_enc_output, self.W_mlm) + self.b_mlm
            
            enc_output = act2fn[self.activation](enc_output)
            
            enc_output = self.norm_mlm(enc_output)

            if self.share_emb_out_proj == False:
                W = self.W_out_mlm
            else:
                W = self.encoder.word_embedding.get_embedding().T
            
            logits = torch.matmul(enc_output, W) + self.b_out_mlm

            outputs["mlm_logits"] = logits
        
        if compute_loss == True:
            
            if self.crf is not None:
                mask = x.ne(self.PAD).to(enc_output.dtype)
                nlg = self.crf(enc_output, targets["labels"], mask)
                outputs["crf_nlg"] = nlg
            
            loss = self.loss_fn(outputs, targets)
            outputs["loss"] = loss
        
        return outputs        


class CLIP(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        """
        """
        super(CLIP, self).__init__()

        self.PAD = kwargs["symbol2id"]["_pad_"]
        self.BOS = kwargs["symbol2id"]["_bos_"]
        self.EOS = kwargs["symbol2id"]["_eos_"]
        self.UNK = kwargs["symbol2id"]["_unk_"]
        self.SEP = kwargs["symbol2id"]["_sep_"]
        self.CLS = kwargs["symbol2id"]["_cls_"]
        self.MASK = kwargs["symbol2id"]["_mask_"]

        img_config = {}
        for k in kwargs:
            if k.startswith("image_") or k in ["symbols", "symbol2id"]:
                img_config[k.replace("image_", "")] = kwargs[k]
        img_config["use_vit_encoder"] = True
        
        self.img_encoder = TransformerEncoder(**img_config)
        text_config = {}
        for k in kwargs:
            if k.startswith("text_") or k in ["symbols", "symbol2id"]:
                text_config[k.replace("text_", "")] = kwargs[k]
        text_config["use_vit_encoder"] = False
        self.text_encoder = TransformerEncoder(**text_config)


    def forward(self, inputs, targets=None, compute_loss=False):
        """
        """
        img, text = inputs["img"], inputs["text"]
              
        img_outputs = self.img_encoder({"x":img})
        text_outputs = self.text_encoder({"x":text})

        outputs = {}
        outputs["img_outputs"] = img_outputs
        outputs["text_outputs"] = text_outputs

        if compute_loss == True:            
            loss = self.loss_fn(outputs, targets)
            outputs["loss"] = loss

        return outputs


class MAE(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        """
        """
        super(MAE, self).__init__()

        img_h,img_w = kwargs["enc_img_h"], kwargs["enc_img_w"]
        ph,pw = kwargs["enc_patch_h"], kwargs["enc_patch_w"]
        self.img_h = img_h
        self.img_w = img_w
        self.ph = ph
        self.pw = pw
        self.n_channels = kwargs["enc_n_channels"]
        
        self.max_len = img_h // ph * img_w // pw + 1
        
        enc_config = {}
        for k in kwargs:
            if k.startswith("enc_"):
                enc_config[k[4:]] = kwargs[k]
        enc_config["max_len"] = self.max_len
        enc_config["use_word_embedding"] = False
        self.encoder = Transformer(**enc_config)
        
        self.dec_embedding = Linear(kwargs["enc_d_model"], kwargs["dec_d_model"])
        self.mask = nn.Parameter(torch.zeros(kwargs["dec_d_model"]))
        dec_config = {}
        for k in kwargs:
            if k.startswith("dec_"):
                dec_config[k[4:]] = kwargs[k]
        dec_config["max_len"] = self.max_len
        dec_config["use_word_embedding"] = False
        self.decoder = Transformer(**dec_config)

        self.out_proj = Linear(kwargs["dec_d_model"], self.ph*self.pw*self.n_channels)
        

    def forward(self, inputs, targets=None, compute_loss=False):
        """
        """
        x, mask_ratio = inputs["x"],inputs["mask_ratio"]
        
        enc_output, mask, ids_restore = self.encoder.get_mask_image_enc(x, mask_ratio)
        
        dec_embeded = self.dec_embedding(enc_output)
        
        mask_tokens = self.mask.repeat(dec_embeded.shape[0], ids_restore.shape[1] + 1 - dec_embeded.shape[1], 1)
        
        dec_embeded_ = torch.cat([dec_embeded[:, 1:, :], mask_tokens], dim=1)  
        dec_embeded_ = torch.gather(dec_embeded_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, dec_embeded_.shape[2]))  
        dec_embeded = torch.cat([dec_embeded[:, :1, :], dec_embeded_], dim=1)  
        
        dec_pos_ids = torch.arange(dec_embeded.shape[1], device=dec_embeded.device).repeat([dec_embeded.shape[0],1])
        dec_output = self.decoder(dec_embeded, 
                                  self_pos_ids=dec_pos_ids,
                                  past_pos_ids=dec_pos_ids)
        
        dec_output = self.out_proj(dec_output["output"][:,1:,:])
        
        h = self.img_h // self.ph
        w = self.img_w // self.pw
        reconstruct = dec_output.reshape(shape=(dec_output.shape[0], h, w, self.ph, self.pw, self.n_channels))
        reconstruct = torch.einsum('nhwpqc->nchpwq', reconstruct)
        reconstruct = reconstruct.reshape(shape=(dec_output.shape[0], self.n_channels,self.img_h, self.img_w))
        
        patchify_x = x.reshape(shape=(x.shape[0], 3, h, self.ph, w, self.pw))
        patchify_x = torch.einsum('nchpwq->nhwpqc', patchify_x)
        patchify_x = patchify_x.reshape(shape=(patchify_x.shape[0], h * w, self.ph * self.pw * 3))
        
        outputs = {
            "output": dec_output, 
            "reconstruct": reconstruct, 
            "mask": mask, 
            "patchify_x": patchify_x
            }
        
        if compute_loss == True:            
            loss = self.loss_fn(outputs, targets)
            outputs["loss"] = loss

        return outputs
        

def load_model_weights(model, weights_path):
    """
    """
    model_path = real_path(weights_path)
    state_dict = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    state_dict = {k.replace("module.",""):state_dict[k] for k in state_dict}
    param_dict = {}
    for k,v in model.named_parameters():
        if k in state_dict:
            param_dict[k] = state_dict[k] 
        else:
            print("warn: weight %s not found in model file" % k)

    model.load_state_dict(param_dict, False)
    
    return model


def build_model(config, load_model_path=None):
    """
    """
    if config["task"] in ["enc_dec", "image2text"]:
        if config["model"] == "transformer":
            model = TransformerSeq2seq(**config)
        else:
            raise ValueError("model not correct!")
    elif config["task"] == "lm":
        if config["model"] == "transformer":
            model = TransformerLM(**config)
        else:
            raise ValueError("model not correct!")
    elif config["task"] in ["cls", "match", "mlm", "seqcls"]:
        if config["model"] == "transformer":
            model = TransformerEncoder(**config)
        else:
            raise ValueError("model not correct!")
    elif config["task"] in ["image_classification"]:
        if config["model"] == "transformer":
            model = TransformerEncoder(**config)
        else:
            raise ValueError("model not correct!")              
    elif config["task"] in ["image_text_match"]:
        if config["model"] == "transformer":
            model = CLIP(**config)
        else:
            raise ValueError("model not correct!")  
    elif config["task"] in ["masked_auto_encoder"]:
        if config["model"] == "transformer":
            model = MAE(**config)
        else:
            raise ValueError("model not correct!")  
    else:
        raise ValueError("model not correct!")  
        
    if load_model_path is not None:
        model = load_model_weights(model, load_model_path)
     
    return model

