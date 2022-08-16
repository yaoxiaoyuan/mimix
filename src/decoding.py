# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:59:36 2019

@author: Xiaoyuan Yao
"""
import math
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import nested_to_cuda

def init_search(model, batch_size):
    """
    """
    vocab_size = model.trg_vocab_size
    y = torch.zeros(batch_size, 1, dtype=torch.long) + model.BOS
    
    log_probs = torch.zeros(batch_size, 1, dtype=torch.float)
    finished = torch.zeros(batch_size, 1, dtype=torch.uint8)
    hypothesis = torch.ones(batch_size, 0, dtype=torch.long)
    history_log_probs = torch.zeros(batch_size, 0, dtype=torch.float)
    
    mask_finished = torch.tensor([model.MIN_LOGITS] * vocab_size, 
                                 dtype=torch.float)
    mask_finished[model.PAD] = model.MAX_LOGITS
    
    states = [y, 
              log_probs,
              finished, 
              mask_finished, 
              hypothesis, 
              history_log_probs]
    
    return states


def beam_step(enc_dec_model, 
              steps, 
              enc_states,
              dec_enc_attn_mask, 
              dec_states, 
              y, 
              last_log_probs,
              mask_finished, 
              finished,
              batch_size,
              beam_size,
              vocab_size, 
              expand, 
              hypothesis, 
              history_log_probs,
              eos):
    """
    """        
    outputs = enc_dec_model.step(steps, 
                                 enc_states, 
                                 dec_enc_attn_mask, 
                                 dec_states, 
                                 y)
    
    dec_states, logits = outputs[0:2]

    cur_log_probs = F.log_softmax(logits, 1) 

    mask = finished.type(torch.float)
    masked_log_probs = cur_log_probs * (1 - mask) + mask_finished * mask
    log_probs = last_log_probs + masked_log_probs

    if expand == True:
        log_probs, indice = log_probs.topk(beam_size)
            
        log_probs = log_probs.view(-1, 1)
        beam_id = torch.arange(0, batch_size).unsqueeze(1)
        beam_id = beam_id.repeat(1, beam_size).view(-1)
        y = indice.view(-1, 1)
        
        cur_log_probs = torch.gather(cur_log_probs, 1, indice)
        
        dec_states, dec_enc_attn_mask, enc_states = \
        enc_dec_model.gather_beam_states(dec_states, 
                                         dec_enc_attn_mask,
                                         beam_id,
                                         enc_states)
    else:
        log_probs = log_probs.view(-1, beam_size * vocab_size)
        log_probs, indice = log_probs.topk(beam_size)
            
        log_probs = log_probs.view(-1, 1)
        base_id = torch.arange(0, batch_size, device = y.device) * beam_size 
        beam_id = (base_id.view(-1, 1) + indice // vocab_size).view(-1)
        y = (indice % vocab_size).view(-1, 1)
        
        cur_log_probs = cur_log_probs.view(-1, beam_size * vocab_size)
        cur_log_probs = torch.gather(cur_log_probs, 1, indice)
        
        dec_states, dec_enc_attn_mask, _ = \
        enc_dec_model.gather_beam_states(dec_states, 
                                         dec_enc_attn_mask,
                                         beam_id)

    log_probs -= finished[beam_id].type(torch.float) * enc_dec_model.MAX_LOGITS
    finished = (finished[beam_id,:] | y.eq(eos).byte())
    
    hypothesis = torch.cat([hypothesis[beam_id,:], y.view(-1, 1)], 1)
    history_log_probs = torch.cat([history_log_probs[beam_id,:], 
                                   cur_log_probs.view(-1, 1)], 1)
    
    return enc_states, dec_enc_attn_mask, dec_states, \
        y, log_probs, finished, hypothesis, history_log_probs


def normalize_log_probs(log_probs, hyp_len, gamma, normalize):
    """
    """
    normalized_score = log_probs
    
    if normalize == "linear":
        normalizer = hyp_len
        normalized_score = log_probs / normalizer.unsqueeze(-1)
    elif normalize == "gnmt":
        normalizer = torch.pow(5 + hyp_len, gamma) / math.pow(5 + 1, gamma)
        normalized_score = log_probs / normalizer.unsqueeze(-1)

    return normalized_score


def get_sort_idx(normalized_score):
    """
    """
    batch_size, beam_size = normalized_score.size()
    
    normalized_score, indice = normalized_score.topk(beam_size)
    
    sort_id = torch.arange(0, batch_size, device=normalized_score.device) 
    sort_id = sort_id.view(batch_size, 1) * beam_size + indice
    sort_id = sort_id.view(-1)
    
    return sort_id


def beam_search(enc_dec_model, 
                x, 
                beam_size, 
                max_decode_steps, 
                gamma,
                normalize="none", 
                sort=True, 
                eos=None):
    """
    """
    if eos is None:
        eos = enc_dec_model.EOS

    batch_size = x.size(0)
    enc_states, dec_states, dec_enc_attn_mask = enc_dec_model.init_search(x)

    search_states = init_search(enc_dec_model, batch_size)    
    if x.is_cuda:
        search_states = nested_to_cuda(search_states, x.device)
    
    y = search_states[0]
    log_probs = search_states[1]
    finished = search_states[2]
    mask_finished = search_states[3] 
    hypothesis = search_states[4]
    history_log_probs = search_states[5]
    
    steps = 0
    trg_seq_len = 0
    vocab_size = enc_dec_model.trg_vocab_size
    max_decode_steps = min(max_decode_steps, 
                           enc_dec_model.trg_max_len - trg_seq_len)
    while not finished.all() and steps < max_decode_steps: 

        if steps == 0:
            expand = True
        else:
            expand = False

        enc_states, dec_enc_attn_mask, dec_states, \
        y, log_probs, finished, hypothesis, history_log_probs = \
        beam_step(enc_dec_model, 
                  steps, 
                  enc_states, 
                  dec_enc_attn_mask, 
                  dec_states, 
                  y, 
                  log_probs, 
                  mask_finished,
                  finished, 
                  batch_size,
                  beam_size, 
                  vocab_size, 
                  expand, 
                  hypothesis,
                  history_log_probs,
                  eos)
        steps += 1
    
    hyp_len = torch.sum(hypothesis.ne(enc_dec_model.PAD).float(), 1)
    normalized_score = \
        normalize_log_probs(log_probs, hyp_len, gamma, normalize)
    if sort == True:
        sort_idx = get_sort_idx(normalized_score.view(batch_size, beam_size))
        
        hypothesis = hypothesis[sort_idx, :]
        normalized_score = normalized_score[sort_idx, :]
        history_log_probs = history_log_probs[sort_idx, :]
    
    outputs = [hypothesis, normalized_score]
    
    return outputs


def sample_step(enc_dec_model, 
                steps, 
                enc_states,
                dec_enc_attn_mask, 
                dec_states, 
                y, 
                last_log_probs,
                mask_finished, 
                finished,
                batch_size,
                sample_size,
                vocab_size, 
                expand, 
                hypothesis, 
                history_log_probs,
                temp,
                eos):
    """
    """
    outputs = enc_dec_model.step(steps, 
                                 enc_states, 
                                 dec_enc_attn_mask, 
                                 dec_states, 
                                 y)
    
    dec_states, logits = outputs[0:2]
    
    mask = finished.type(torch.float)
    cur_log_probs = F.log_softmax(logits, 1)
    
    mask_logits = logits * (1 - mask) + mask_finished * mask
    
    probs = F.softmax(temp * mask_logits, 1)
    
    if expand == True:
        indice = torch.multinomial(probs, sample_size, replacement=True)
    
        y = (indice % vocab_size).view(-1, 1)
        
        repeat_id = torch.arange(0, batch_size).unsqueeze(1)
        repeat_id = repeat_id.repeat(1, sample_size).view(-1)
        
        finished = finished[repeat_id]
        mask = mask[repeat_id]
        hypothesis = hypothesis[repeat_id]
        last_log_probs = last_log_probs[repeat_id]
        history_log_probs = history_log_probs[repeat_id]

        dec_states, dec_enc_attn_mask, enc_states = \
        enc_dec_model.gather_beam_states(dec_states, 
                                         dec_enc_attn_mask,
                                         repeat_id,
                                         enc_states)
        
        finished = (finished | y.eq(enc_dec_model.EOS).byte())
    
        hypothesis = torch.cat([hypothesis, y], 1)
        cur_log_probs = torch.gather(cur_log_probs, 1, indice).view(-1, 1)
        log_probs = last_log_probs + cur_log_probs * (1 - mask)
        history_log_probs = torch.cat([history_log_probs, cur_log_probs], 1)
    else:
        indice = torch.multinomial(probs, 1)
    
        y = (indice % vocab_size).view(-1, 1)
        
        finished = (finished | y.eq(enc_dec_model.EOS).byte())
    
        hypothesis = torch.cat([hypothesis, y], 1)
        cur_log_probs = torch.gather(cur_log_probs, 1, indice)
        log_probs = last_log_probs + cur_log_probs * (1 - mask)
        history_log_probs = torch.cat([history_log_probs, cur_log_probs], 1)
    
    return enc_states, dec_enc_attn_mask, dec_states, y, log_probs, finished,\
            hypothesis, history_log_probs


def sample(enc_dec_model, 
           x, 
           sample_size,
           max_decode_steps,
           sample_temp=1, 
           normalize="none", 
           gamma=1, 
           eos=None):
    """
    """  
    if eos is None:
        eos = enc_dec_model.EOS
    
    batch_size = x.size(0)
    enc_states, dec_states, dec_enc_attn_mask = enc_dec_model.init_search(x)

    search_states = init_search(enc_dec_model, batch_size)    
    if x.is_cuda:
        search_states = nested_to_cuda(search_states, x.device)
    
    y = search_states[0]
    log_probs = search_states[1]
    finished = search_states[2]
    mask_finished = search_states[3] 
    hypothesis = search_states[4]
    history_log_probs = search_states[5]
    
    steps = 0
    trg_seq_len = 0
    vocab_size = enc_dec_model.trg_vocab_size
    max_decode_steps = min(max_decode_steps, 
                           enc_dec_model.trg_max_len - trg_seq_len)
    while not finished.all() and steps < max_decode_steps:
        if steps == 0:
            expand = True
        else:
            expand = False
        enc_states, dec_enc_attn_mask, dec_states, y, log_probs, finished,\
            hypothesis, history_log_probs = sample_step(enc_dec_model, 
                                                        steps, 
                                                        enc_states,
                                                        dec_enc_attn_mask, 
                                                        dec_states, 
                                                        y, 
                                                        log_probs,
                                                        mask_finished, 
                                                        finished,
                                                        batch_size,
                                                        sample_size,
                                                        vocab_size, 
                                                        expand, 
                                                        hypothesis, 
                                                        history_log_probs,
                                                        sample_temp,
                                                        eos)
        
        steps += 1
        trg_seq_len += 1
    
    hyp_len = torch.sum(hypothesis.ne(enc_dec_model.PAD).float(), 1)
    normalized_score = \
        normalize_log_probs(log_probs, hyp_len, gamma, normalize)
    
    outputs = [hypothesis, normalized_score]
    
    return outputs


def lm_sample(lm_model, 
              max_decode_steps, 
              use_cuda,
              device,
              batch_size=1,
              normalize="none", 
              gamma=1, 
              temp=1, 
              eos=None):
    """
    """  
    if eos is None:
        eos = lm_model.EOS
    
    dec_states = lm_model.init_search()
    search_states = init_search(lm_model, batch_size)    
    if use_cuda == True:
        search_states = nested_to_cuda(search_states, device)
    
    y = search_states[0]
    log_probs = search_states[1]
    finished = search_states[2]
    mask_finished = search_states[3] 
    hypothesis = search_states[4]
    history_log_probs = search_states[5]
    
    steps = 0
    trg_seq_len = 0
    vocab_size = lm_model.trg_vocab_size
    while not finished.all() and steps < max_decode_steps and trg_seq_len < max_decode_steps: 
        outputs = lm_model._step(steps, dec_states, y)
        dec_states, logits = outputs[0:2]
        
        mask = finished.type(torch.float)
        cur_log_probs = F.log_softmax(logits, 1)
        mask_logits = logits * (1 - mask) + mask_finished * mask
        
        probs = F.softmax(temp * mask_logits, 1)

        indice = torch.multinomial(probs, 1)
        
        y = (indice % vocab_size).view(-1, 1)
        
        finished = (finished | y.eq(eos).byte())
        
        hypothesis = torch.cat([hypothesis, y], 1)
        cur_log_probs = torch.gather(cur_log_probs, 1, indice)
        log_probs = log_probs + cur_log_probs * (1 - mask)
        history_log_probs = torch.cat([history_log_probs, cur_log_probs], 1)
        trg_seq_len += 1
        steps += 1
    
    hyp_len = torch.sum(hypothesis.ne(lm_model.PAD).float(), 1)
    normalized_score = \
        normalize_log_probs(log_probs, hyp_len, gamma, normalize)
    
    outputs = [hypothesis, normalized_score]
    
    return outputs


def filter_top_k(logits, top_k, fill_value):
    """
    """
    top_k = min(top_k, logits.size(-1))
    _logits,_indice = torch.topk(logits, top_k)
    logits[logits < _logits[:, -1, None]] = fill_value
    return logits


def filter_top_p(logits, top_p, fill_value):
    """
    """
    _logits, _indices = torch.sort(logits, descending=True)
    cumulative_logits = torch.cumsum(torch.softmax(_logits, -1), dim=-1)
    need_filter =  (cumulative_logits > top_p)
    
    need_filter[:, 1:] = need_filter[:, :-1].clone()
    need_filter[:, 0] = 0
    
    filter_indice = need_filter.scatter(1, _indices, need_filter)
    logits[filter_indice] = fill_value
        
    return logits


def top_k_sampling(logits, n_samples, top_k):
    """
    """
    fill_value = -10000
        
    logits = filter_top_k(logits, top_k, fill_value)
    
    probs = torch.softmax(logits, -1)
    
    return torch.multinomial(probs, n_samples, replacement=True)


def top_k_top_p_sampling(logits, 
                         top_k, 
                         top_p, 
                         temp=1, 
                         n_samples=1, 
                         replacement=True):
    """
    """
    fill_value = -10000
    
    if top_k > 0:
        logits = filter_top_k(logits, top_k, fill_value)
    if top_p > 0:
        logits = filter_top_p(logits, top_p, fill_value)
    
    logits[logits > fill_value] = (logits[logits > fill_value] * temp)
    
    probs = torch.softmax(logits, -1)

    return torch.multinomial(probs, n_samples, replacement=replacement)


def get_single_token_mask(vocab_size, mask_id, penalty):
    """
    """
    mask_unk = torch.zeros([vocab_size], dtype=torch.float)
    mask_unk[mask_id] = penalty
    
    return mask_unk


def get_multi_token_mask(words, 
                         vocab_size,
                         start, 
                         end, 
                         penalty, 
                         penalty_beta,
                         penalty_vocab_start=-1, 
                         penalty_vocab_end=-1):
    """
    """
    mask = torch.zeros(words.shape[0]*words.shape[1], 
                       vocab_size, 
                       device = words.device)
    mask.scatter_(1, words.view(-1, 1), 1)
    mask = mask.view(words.shape[0], words.shape[1], vocab_size)

    mask[:, :start, :] = 0
    mask[:, end:, :] = 0
    
    mask = torch.sum(mask, 1)
    
    mask[mask > 0] = penalty + penalty_beta * mask[mask > 0]

    if penalty_vocab_start > 0:
        mask[:, :penalty_vocab_start] = 0

    if penalty_vocab_end > 0:
        mask[:, penalty_vocab_end:] = 0

    return mask


def get_prefix_states(enc_dec_model, 
                      steps, 
                      enc_states,
                      dec_enc_attn_mask, 
                      dec_states, 
                      y, 
                      last_log_probs,
                      finished,
                      hypothesis, 
                      history_log_probs,
                      eos,
                      prefix):
    """
    """
    steps = 0
    log_probs = last_log_probs
    while not finished.all() and steps < prefix.size(1): 
        outputs = enc_dec_model.step(steps, 
                                     enc_states, 
                                     dec_enc_attn_mask, 
                                     dec_states, 
                                     y)
        dec_states, logits = outputs[0:2]
        
        y = prefix[:, steps:steps+1]
        
        cur_log_probs = torch.gather(torch.log_softmax(logits, -1), 1, y)
        
        finished = (finished | y.eq(eos).byte())
        log_probs = log_probs + cur_log_probs
        history_log_probs = torch.cat([history_log_probs, cur_log_probs], 1)
        
        steps += 1
        
    return dec_states, y, log_probs, finished, history_log_probs


def beam_step_with_constraints(enc_dec_model, 
                               steps, 
                               enc_states,
                               dec_enc_attn_mask, 
                               dec_states, 
                               y, 
                               last_log_probs, 
                               mask_finished, 
                               finished,
                               batch_size, 
                               beam_size, 
                               vocab_size, 
                               expand, 
                               hypothesis, 
                               history_log_probs,
                               repeat_penalty,
                               history_penalty,
                               mask_unk,
                               group_sampling,
                               temp,
                               top_k,
                               diverse_rate,
                               eos):
    """
    """
    outputs = enc_dec_model.step(steps, 
                                 enc_states, 
                                 dec_enc_attn_mask,
                                 dec_states, 
                                 y)
    dec_states, logits = outputs[0:2]

    if repeat_penalty is not None:
        logits += repeat_penalty

    if history_penalty is not None:
        logits += history_penalty
    
    if mask_unk is not None:
        logits += mask_unk

    cur_log_probs = F.log_softmax(temp * logits, 1) 

    masked_log_probs = cur_log_probs 
    if diverse_rate > 0 and expand == False:
        diverse_penalty = diverse_rate * torch.arange(0, vocab_size, 1, dtype=torch.float, device=cur_log_probs.device)
    
        sort_idx = cur_log_probs.topk(vocab_size, -1)[1]
        
        diverse_penalty = torch.gather(diverse_penalty, -1, sort_idx.view(-1))
        
        diverse_penalty = diverse_penalty.view(-1, vocab_size)
        
        masked_log_probs -= diverse_penalty
    
    mask = finished.type(torch.float)
    masked_log_probs = masked_log_probs * (1 - mask) + mask_finished * mask
    
    log_probs = last_log_probs + masked_log_probs

    if expand == True:
        if group_sampling == False or top_k < 0:
            log_probs, indice = log_probs.topk(beam_size)
        else:
            indice = top_k_sampling(logits, beam_size, top_k)
            log_probs = torch.gather(log_probs, 1, indice)
            
        log_probs = log_probs.view(-1, 1)
        beam_id = torch.arange(0, batch_size).unsqueeze(1)
        beam_id = beam_id.repeat(1, beam_size).view(-1)
        y = indice.view(-1, 1)
        
        cur_log_probs = torch.gather(cur_log_probs, 1, indice)
        
        dec_states, dec_enc_attn_mask, enc_states = \
        enc_dec_model.gather_beam_states(dec_states, 
                                         dec_enc_attn_mask,
                                         beam_id,
                                         enc_states)
    else:
        log_probs = log_probs.view(-1, beam_size * vocab_size)
        log_probs, indice = log_probs.topk(beam_size)
            
        log_probs = log_probs.view(-1, 1)
        base_id = torch.arange(0, batch_size, device = y.device) * beam_size 
        beam_id = (base_id.view(-1, 1) + indice // vocab_size).view(-1)
        y = (indice % vocab_size).view(-1, 1)
        
        cur_log_probs = cur_log_probs.view(-1, beam_size * vocab_size)
        cur_log_probs = torch.gather(cur_log_probs, 1, indice)
        
        dec_states, dec_enc_attn_mask, _ = \
        enc_dec_model.gather_beam_states(dec_states, 
                                         dec_enc_attn_mask,
                                         beam_id)
    
    log_probs -= finished[beam_id].type(torch.float) * enc_dec_model.MAX_LOGITS
    finished = (finished[beam_id,:] | y.eq(eos).byte())
    
    hypothesis = torch.cat([hypothesis[beam_id,:], y.view(-1, 1)], 1)
    history_log_probs = torch.cat([history_log_probs[beam_id,:], 
                                   cur_log_probs.view(-1, 1)], 1)

    return enc_states, dec_enc_attn_mask, dec_states, \
        y, log_probs, finished, hypothesis, history_log_probs


def beam_search_with_constraints(enc_dec_model, 
                                 x, 
                                 beam_size,
                                 search_strategy,
                                 max_decode_steps, 
                                 group_size=None, 
                                 top_k=-1,
                                 diverse_rate=0,
                                 history_penalty=0,
                                 history_penalty_beta=0,
                                 repeat_penalty=0,
                                 penalty_vocab_start=-1,
                                 penalty_vocab_end=-1,
                                 alpha_0=1,
                                 alpha=1,
                                 beta=0,
                                 need_mask_unk=False,
                                 eos=None,
                                 prefix_y=None,
                                 normalize="none",
                                 gamma=0,
                                 sort=True,
                                 init_states=None,
                                 return_states=False):
    """
    """    
    if eos is None:
        eos = enc_dec_model.EOS

    cur_beam_size = beam_size
    if "group" in search_strategy:
        cur_beam_size = group_size

    steps = 0
    trg_seq_len = 0
    if init_states is None:
        
        use_cuda = x.is_cuda
    
        batch_size = x.size(0)
        enc_states, dec_states, dec_enc_attn_mask = enc_dec_model.init_search(x)

        search_states = init_search(enc_dec_model, batch_size)    
        if x.is_cuda:
            search_states = nested_to_cuda(search_states, x.device)
    
        y = search_states[0]
        log_probs = search_states[1]
        finished = search_states[2]
        mask_finished = search_states[3] 
        hypothesis = search_states[4]
        history_log_probs = search_states[5]
        
        if prefix_y is not None:
            dec_states, y, log_probs, finished, history_log_probs = \
                get_prefix_states(enc_dec_model, 
                                  steps, 
                                  enc_states,
                                  dec_enc_attn_mask, 
                                  dec_states, 
                                  y, 
                                  log_probs, 
                                  finished,
                                  hypothesis, 
                                  history_log_probs,
                                  eos,
                                  prefix_y)
            
            hypothesis = prefix_y
            trg_seq_len = prefix_y.size(1)
    else:        
        enc_states, dec_states, y, log_probs, finished, mask_finished, \
        hypothesis, history_log_probs, dec_enc_attn_mask = init_states
        trg_seq_len = hypothesis.size(1)
        use_cuda = y.is_cuda
        batch_size = y.size(0)
    mask_unk = None
    if need_mask_unk == True:
        mask_unk = get_single_token_mask(enc_dec_model.trg_vocab_size,
                                         enc_dec_model.UNK,
                                         enc_dec_model.MIN_LOGITS)
    if use_cuda == True:
        mask_unk = nested_to_cuda(mask_unk, x.device)
        
    vocab_size = enc_dec_model.trg_vocab_size
    max_decode_steps = min(max_decode_steps, 
                           enc_dec_model.trg_max_len - trg_seq_len)
    while not finished.all() and steps < max_decode_steps:

        cur_repeat_penalty = None
        cur_history_penalty = None

        if trg_seq_len > 1 and repeat_penalty < 0:
            cur_repeat_penalty = get_multi_token_mask(hypothesis, 
                                                      vocab_size, 
                                                      -2, 
                                                      steps, 
                                                      repeat_penalty, 
                                                      0, 
                                                      penalty_vocab_start,
                                                      penalty_vocab_end)

        if trg_seq_len > 2 and history_penalty < 0:
            cur_history_penalty = get_multi_token_mask(hypothesis, 
                                                       vocab_size, 
                                                       0, 
                                                       -2, 
                                                       history_penalty, 
                                                       history_penalty_beta,
                                                       penalty_vocab_start, 
                                                       penalty_vocab_end)
        
        if steps == 0:
            cur_batch_size = batch_size
            cur_beam_size = beam_size
            if "group" in search_strategy:
                cur_beam_size = group_size
            expand = True
            temp = alpha_0
            group_sampling = False
            if "random" in search_strategy:
                group_sampling = True
        elif steps == 1 and "group" in search_strategy:
            cur_batch_size = batch_size*group_size
            cur_beam_size = beam_size // group_size
            temp = alpha
            expand = True
            group_sampling = False
        else:
            cur_batch_size = batch_size
            cur_beam_size = beam_size
            if "group" in search_strategy:
                cur_batch_size = batch_size*group_size
                cur_beam_size = beam_size // group_size
            temp = alpha + beta * steps
            expand = False
            group_sampling = False
        
        enc_states, dec_enc_attn_mask, dec_states, \
        y, log_probs, finished, hypothesis, history_log_probs = \
        beam_step_with_constraints(enc_dec_model, 
                                   trg_seq_len, 
                                   enc_states,
                                   dec_enc_attn_mask, 
                                   dec_states, 
                                   y, 
                                   log_probs, 
                                   mask_finished, 
                                   finished,
                                   cur_batch_size, 
                                   cur_beam_size, 
                                   vocab_size, 
                                   expand, 
                                   hypothesis, 
                                   history_log_probs,
                                   cur_repeat_penalty,
                                   cur_history_penalty,
                                   mask_unk,
                                   group_sampling,
                                   temp,
                                   top_k,
                                   diverse_rate,
                                   eos)
        steps += 1
        trg_seq_len += 1

    hyp_len = torch.sum(hypothesis.ne(enc_dec_model.PAD).float(), 1)
    normalized_score = \
        normalize_log_probs(log_probs, hyp_len, gamma, normalize)
    if sort == True:
        sort_idx = get_sort_idx(normalized_score.view(batch_size, beam_size))
        
        hypothesis = hypothesis[sort_idx, :]
        normalized_score = normalized_score[sort_idx, :]
        history_log_probs = history_log_probs[sort_idx, :]
    
    outputs = [hypothesis, normalized_score]
    if return_states == True:
        outputs = [hypothesis, 
                   normalized_score, 
                   history_log_probs,
                   enc_states, 
                   dec_states, 
                   y, 
                   log_probs, 
                   finished, 
                   mask_finished, 
                   dec_enc_attn_mask]
    
    return outputs


def sample_step_with_constraints(enc_dec_model, 
                                 steps, 
                                 enc_states,
                                 dec_enc_attn_mask, 
                                 dec_states, 
                                 y, 
                                 last_log_probs,
                                 mask_finished, 
                                 finished,
                                 batch_size,
                                 sample_size,
                                 vocab_size, 
                                 expand, 
                                 hypothesis, 
                                 history_log_probs,
                                 top_k,
                                 top_p,
                                 temp,
                                 repeat_penalty,
                                 history_penalty,
                                 mask_unk,
                                 prefix,
                                 eos):
    """
    """
    outputs = enc_dec_model.step(steps, 
                                 enc_states, 
                                 dec_enc_attn_mask, 
                                 dec_states, 
                                 y)
    
    dec_states, logits = outputs[0:2]
    
    mask = finished.type(torch.float)
    
    if repeat_penalty is not None:
        logits = logits + repeat_penalty
        
    if history_penalty is not None:
        logits = logits + history_penalty
    
    if mask_unk is not None:
        logits = logits + mask_unk
    
    if prefix is not None:
        prefix_mask = torch.zeros_like(logits)
        prefix_mask = prefix_mask.scatter_(1, prefix, enc_dec_model.MAX_LOGITS)
        prefix_mask = prefix_mask * prefix.ne(enc_dec_model.PAD).float()
        logits = logits + prefix_mask
    
    _log_probs = F.log_softmax(logits, 1)
    
    mask_logits = logits * (1 - mask) + mask_finished * mask
    
    probs = F.softmax(temp * mask_logits, 1)

    if expand == True:
        indice = top_k_top_p_sampling(mask_logits, 
                                      top_k, 
                                      top_p, 
                                      temp, 
                                      sample_size, 
                                      replacement=True)
        
        y = (indice % vocab_size).view(-1, 1)
        
        repeat_id = torch.arange(0, batch_size).unsqueeze(1)
        repeat_id = repeat_id.repeat(1, sample_size).view(-1)
        
        finished = finished[repeat_id]
        mask = mask[repeat_id]
        hypothesis = hypothesis[repeat_id]
        last_log_probs = last_log_probs[repeat_id]
        history_log_probs = history_log_probs[repeat_id]

        dec_states, dec_enc_attn_mask, enc_states = \
        enc_dec_model.gather_beam_states(dec_states, 
                                         dec_enc_attn_mask,
                                         repeat_id,
                                         enc_states)
        
        finished = (finished | y.eq(eos).byte())

        hypothesis = torch.cat([hypothesis, y], 1)

        _log_probs = torch.gather(_log_probs, 1, indice).view(-1, 1)

        log_probs = last_log_probs + _log_probs * (1 - mask)
        history_log_probs = torch.cat([history_log_probs, _log_probs], 1)
    else:
        indice = top_k_top_p_sampling(mask_logits, 
                                      top_k, 
                                      top_p, 
                                      temp)
    
        y = (indice % vocab_size).view(-1, 1)
        
        finished = (finished | y.eq(eos).byte())
    
        hypothesis = torch.cat([hypothesis, y], 1)
        _log_probs = torch.gather(_log_probs, 1, indice)

        log_probs = last_log_probs + _log_probs * (1 - mask)
        history_log_probs = torch.cat([history_log_probs, _log_probs], 1)

    return enc_states, dec_enc_attn_mask, dec_states, y, log_probs, finished,\
            hypothesis, history_log_probs


def sample_with_constraints(enc_dec_model, 
                            x, 
                            sample_size,
                            max_decode_steps,
                            repeat_penalty = 0,
                            history_penalty = 0,
                            history_penalty_beta = 0,
                            penalty_vocab_start = -1,
                            penalty_vocab_end = -1,
                            need_mask_unk=False,
                            sample_alpha_0=1,
                            sample_alpha=1,
                            sample_beta=0,
                            sample_top_k=-1,
                            sample_top_k0=-1,
                            sample_top_p=-1,
                            sample_top_p0=-1,
                            normalize="none", 
                            gamma=1, 
                            prefix_y=None,
                            eos=None,
                            return_states=False,
                            early_stop=False,
                            sort=True):
    """
    """  
    if eos is None:
        eos = enc_dec_model.EOS
    
    batch_size = x.size(0)
    enc_states, dec_states, dec_enc_attn_mask = enc_dec_model.init_search(x)

    search_states = init_search(enc_dec_model, batch_size)    
    if x.is_cuda:
        search_states = nested_to_cuda(search_states, x.device)
    
    y = search_states[0]
    log_probs = search_states[1]
    finished = search_states[2]
    mask_finished = search_states[3] 
    hypothesis = search_states[4]
    history_log_probs = search_states[5]

    mask_unk = None
    if need_mask_unk == True:
        mask_unk = get_single_token_mask(enc_dec_model.trg_vocab_size,
                                         enc_dec_model.UNK,
                                         enc_dec_model.MIN_LOGITS)
    if x.is_cuda == True:
        mask_unk = nested_to_cuda(mask_unk, x.device)
    
    steps = 0
    trg_seq_len = 0
    vocab_size = enc_dec_model.trg_vocab_size
    max_decode_steps = min(max_decode_steps, 
                           enc_dec_model.trg_max_len - trg_seq_len)
    while not finished.all() and steps < max_decode_steps:
        if early_stop == True and finished.any():
            break
        if steps == 0:
            expand = True
        else:
            expand = False

        cur_repeat_penalty = None
        cur_history_penalty = None

        if trg_seq_len > 1 and repeat_penalty < 0:
            cur_repeat_penalty = get_multi_token_mask(hypothesis, 
                                                      vocab_size, 
                                                      -2, 
                                                      steps, 
                                                      repeat_penalty, 
                                                      0, 
                                                      penalty_vocab_start,
                                                      penalty_vocab_end)

        if trg_seq_len > 2 and history_penalty < 0:
            cur_history_penalty = get_multi_token_mask(hypothesis, 
                                                       vocab_size, 
                                                       0, 
                                                       -2, 
                                                       history_penalty, 
                                                       history_penalty_beta,
                                                       penalty_vocab_start, 
                                                       penalty_vocab_end)

        temp = sample_alpha_0
        if steps == 0:
            _top_k,_top_p = sample_top_k0, sample_top_p0
        else:
            _top_k,_top_p = sample_top_k, sample_top_p
            temp = sample_alpha + steps * sample_beta
        
        prefix = None
        if prefix_y is not None and trg_seq_len < prefix_y.size(1):
            prefix = prefix_y[:, trg_seq_len:trg_seq_len+1]
            if steps > 0:
                prefix = prefix.repeat_interleave(sample_size, 1).view(-1, 1)

        enc_states, dec_enc_attn_mask, dec_states, y, log_probs, finished,\
            hypothesis, history_log_probs = sample_step_with_constraints(
                    enc_dec_model, 
                    trg_seq_len, 
                    enc_states,
                    dec_enc_attn_mask, 
                    dec_states, 
                    y, 
                    log_probs,
                    mask_finished, 
                    finished,
                    batch_size,
                    sample_size,
                    vocab_size, 
                    expand, 
                    hypothesis, 
                    history_log_probs,
                    _top_k,
                    _top_p,
                    temp,
                    cur_repeat_penalty,
                    cur_history_penalty,
                    mask_unk,
                    prefix,
                    eos)
        
        steps += 1
        trg_seq_len += 1

    hyp_len = torch.sum(hypothesis.ne(enc_dec_model.PAD).float(), 1)
    normalized_score = \
        normalize_log_probs(log_probs, hyp_len, gamma, normalize)

    if sort == True:
        sort_idx = get_sort_idx(normalized_score.view(batch_size, sample_size))
        
        hypothesis = hypothesis[sort_idx, :]
        normalized_score = normalized_score[sort_idx, :]
        history_log_probs = history_log_probs[sort_idx, :]
    
    outputs = [hypothesis, normalized_score]
    
    if return_states == True:
        outputs = [hypothesis, 
                   normalized_score,
                   enc_states,
                   dec_states, 
                   y, 
                   log_probs, 
                   finished, 
                   mask_finished, 
                   history_log_probs, 
                   dec_enc_attn_mask
               ]
    
    return outputs


def lm_sample_with_constraints(lm_model, 
                               max_decode_steps, 
                               use_cuda, 
                               device,
                               batch_size=1,
                               alpha_0=1,
                               alpha=1,
                               beta=0,
                               repeat_penalty=0,
                               history_penalty=0,
                               history_penalty_beta=0,
                               penalty_vocab_start=-1,
                               penalty_vocab_end=-1,
                               prefix=None,
                               gamma=1,
                               normalize="none",
                               top_k=-1,
                               top_k0=-1,
                               top_p=-1,
                               top_p0=-1,
                               eos=None,
                               need_mask_unk=True,
                               return_states=False):
    """
    """
    if eos is None:
        eos = lm_model.EOS

    dec_states = lm_model.init_search()
    search_states = init_search(lm_model, batch_size)    
    if use_cuda == True:
        search_states = nested_to_cuda(search_states, device)
    
    y = search_states[0]
    log_probs = search_states[1]
    finished = search_states[2]
    mask_finished = search_states[3] 
    hypothesis = search_states[4]
    history_log_probs = search_states[5]
    
    gamma = torch.tensor(gamma, dtype=torch.float, device=y.device)
    mask_unk = None
    if need_mask_unk == True:
        mask_unk = get_single_token_mask(lm_model.trg_vocab_size,
                                         lm_model.UNK,
                                         lm_model.MIN_LOGITS)
    if use_cuda == True:
        mask_unk = nested_to_cuda(mask_unk, device)
    
    steps = 0
    trg_seq_len = 0
    vocab_size = lm_model.trg_vocab_size
    max_decode_steps = min(max_decode_steps, lm_model.trg_max_len - trg_seq_len)
    while not finished.all() and steps < max_decode_steps: 
        outputs = lm_model.decoder._step(steps, 
                                         dec_states, 
                                         y)
        dec_states, logits = outputs[0:2]
        
        if mask_unk is not None:
            logits += mask_unk

        if steps > 1 and repeat_penalty < 0:
            logits += get_multi_token_mask(hypothesis,
                                           vocab_size, 
                                           -2, 
                                           steps, 
                                           repeat_penalty, 0, 
                                           penalty_vocab_start,
                                           penalty_vocab_end)
            
        if steps > 2 and history_penalty < 0:
            logits += get_multi_token_mask(hypothesis,
                                           vocab_size, 
                                           0, 
                                           -2, 
                                           history_penalty, 
                                           history_penalty_beta,
                                           penalty_vocab_start, 
                                           penalty_vocab_end)
        
        mask = finished.type(torch.float)
        mask_logits = logits * (1 - mask) + mask_finished * mask
        _log_probs = F.log_softmax(logits, 1)
        
        temp = alpha_0
        if steps > 0:
            temp = alpha + steps * beta

        if prefix is not None and steps < prefix.size(1):
            is_prefix = (prefix[:,steps:steps+1]).ne(lm_model.PAD).float()
            prefix_mask = torch.zeros_like(mask_logits)
            prefix_mask.scatter_(1, prefix[:, steps:steps+1], 
                                 lm_model.MAX_LOGITS)
            mask_logits += (prefix_mask * is_prefix)
            
            indice = top_k_top_p_sampling(mask_logits, -1, -1)
        elif steps == 0:
            indice = top_k_top_p_sampling(mask_logits, top_k0, top_p0, temp)
        else:
            indice = top_k_top_p_sampling(mask_logits, top_k, top_p, temp)
        
        y = (indice % vocab_size).view(-1, 1)
        
        finished = (finished | y.eq(eos).byte())
        
        hypothesis = torch.cat([hypothesis, y], 1)
        _log_probs = torch.gather(_log_probs, 1, indice)
        log_probs = log_probs + _log_probs * (1 - mask)
        history_log_probs = torch.cat([history_log_probs, _log_probs], 1)
        steps += 1
        trg_seq_len += 1
    
    hyp_len = torch.sum(hypothesis.ne(lm_model.PAD).float(), 1)
    normalized_score = \
        normalize_log_probs(log_probs, hyp_len, gamma, normalize)
    
    outputs = [hypothesis, normalized_score]
    if return_states == True:
        outputs = [hypothesis, 
                   normalized_score, 
                   history_log_probs,
                   dec_states, 
                   y, 
                   log_probs, 
                   finished, 
                   mask_finished]
        
    return outputs


def crf_decoding(crf, emission, mask=None, pad_tag=0):
    """
    """
    batch_size, seq_len, n_labels = emission.size()
    
    if mask is not None:
        end_mask = crf.get_end_mask(mask)
    
    scores = crf.start_trans + emission[:, 0, :]
    if mask is not None:
        scores = scores + end_mask[:, 0:1] * crf.end_trans
    path_table = torch.zeros(batch_size, seq_len-1, n_labels, dtype=torch.long)
    
    for i in range(1, seq_len):
        all_scores = scores.unsqueeze(2) + emission[:, i, :].unsqueeze(1) + crf.trans.unsqueeze(0)

        if mask is not None:
            all_scores = all_scores + (end_mask[:, i:i+1] * crf.end_trans).unsqueeze(1)
        
        next_scores,indices = torch.max(all_scores, 1)

        next_scores = mask[:,i:i+1] * next_scores + (1 - mask[:,i:i+1]) * scores
            
        path_table[:, i-1, :] = indices
    
        scores = next_scores
    
    best_scores,end_tag = torch.max(scores, 1)
    end_tag = end_tag.unsqueeze(-1)
    
    indice = end_tag
    best_path = indice
    for i in range(seq_len-2, -1, -1):
        indice = torch.gather(path_table[:,i,:], -1, indice)

        if end_mask is not None:
            _mask = end_mask[:,i].unsqueeze(-1)
            
            indice = indice * (1 - _mask.long()) + end_tag * _mask.long() 
        
        best_path = torch.cat([indice, best_path], 1)
    
    if mask is not None:
        best_path[mask<1] = pad_tag
    
    return best_path


def crf_model_decoding(model, x):
    """
    """
    emission = model.get_emission(x)
    
    mask = x.ne(model.PAD)
    
    return crf_decoding(model.crf, emission, mask, model.PAD)


