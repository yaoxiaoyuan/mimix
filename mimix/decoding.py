# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:59:36 2019

@author: Xiaoyuan Yao
"""
import torch

def init_search(model, batch_size, use_cuda):
    """
    """
    vocab_size = model.trg_vocab_size
    y = torch.zeros(batch_size, 1, dtype=torch.long) + model.BOS

    log_probs = torch.zeros(batch_size, 1, dtype=torch.float)
    finished = torch.zeros(batch_size, 1, dtype=torch.uint8)
    hypothesis = torch.zeros(batch_size, 1, dtype=torch.long) + model.BOS
    history_probs = torch.zeros(batch_size, 0, dtype=torch.float)

    mask_finished = torch.tensor([model.MIN_LOGITS] * vocab_size,
                                 dtype=torch.float)
    mask_finished[model.PAD] = model.MAX_LOGITS
                     
    if use_cuda == False:
        states = [y,
                  log_probs,
                  finished,
                  mask_finished,
                  hypothesis,
                  history_probs]
    else:
        states = [y.cuda(),
                  log_probs.cuda(),
                  finished.cuda(),
                  mask_finished.cuda(),
                  hypothesis.cuda(),
                  history_probs.cuda()]        
    return states


def process_logits(model, logits, states, repetition_penalty):
    """ 
    """
    mask_unk = torch.zeros_like(logits)
    mask_unk[:,model.UNK] = model.MIN_LOGITS
    logits = logits + mask_unk
    
    if repetition_penalty < 0:
        y, log_probs, finished, mask_finished, hypothesis, history_probs = states
        mask = torch.zeros(hypothesis.shape[0]*hypothesis.shape[1], 
                           model.trg_vocab_size, 
                           device = hypothesis.device)
        mask.scatter_(1, hypothesis.view(-1, 1), 1)
        mask = mask.view(hypothesis.shape[0], hypothesis.shape[1], model.trg_vocab_size)

        mask = torch.sum(mask, 1)
    
        logits = logits + mask * repetition_penalty
    
    return logits
    

def top_k_top_p_sampling(logits,
                         top_k=-1,
                         top_p=-1,
                         temperature=1,
                         n_samples=1,
                         replacement=True):
    """
    """
    logits /= temperature
    probs = torch.softmax(logits, -1)
    
    if top_k > 0 or top_p > 0:
        _logits, _indices = torch.sort(logits, descending=True)
    
        if top_k > 0:
            probs[logits < _logits[:, top_k, None]] = 0 

        if top_p > 0:
            cumulative_logits = torch.cumsum(torch.softmax(_logits, -1), dim=-1)
            need_filter =  (cumulative_logits > top_p)

            need_filter[:, 1:] = need_filter[:, :-1].clone()
            need_filter[:, 0] = 0

            filter_indice = need_filter.scatter(1, _indices, need_filter)
            probs[filter_indice] = 0

        probs /= torch.sum(probs, dim=-1, keepdim=True) 

    samples = torch.multinomial(probs, n_samples, replacement=replacement)
    probs = torch.gather(probs, 1, samples)

    return samples, probs


def search(model, 
           beam_size, 
           inputs=None,
           use_cuda=False,
           strategy="beam_search",
           top_k=-1,
           top_p=-1,
           temperature=1,
           eos=None,
           group_size=-1, 
           repetition_penalty=0,
           use_mask_unk=False,
           max_decode_steps=None):
    """
    """ 
    batch_size = 1
    if len(inputs) > 0 and inputs[0] is not None:
        batch_size = inputs[0].size(0)
    
    states = init_search(model, batch_size, use_cuda)

    states, cache = model.init_search(states, inputs)
    
    steps = 0
    last_beam_size = 1
    cur_batch_size = batch_size
    cur_beam_size = beam_size
    if group_size > 0:
        cur_beam_size = group_size
    
    while True:
        
        y, log_probs, finished, mask_finished, hypothesis, history_probs = states
        
        
        logits, cache = model.step(states, cache)  
        
        vocab_size = logits.size(-1)
        logits = logits.view(-1, vocab_size)
        
        #logits: (B x last_beam_size) x V
        #probs: (B x last_beam_size) x V
        logits = process_logits(model, logits, states, repetition_penalty)
        probs = torch.softmax(logits, -1)  
        
        if strategy == "beam_search":
            
            #log_probs: B x last_beam_size
            #finished: (B x last_beam_size) x 1
            #mask_finished: vocab_size      
            #cur_log_probs: (B x last_beam_size) x V
            masked_logits = logits * (1 - finished.float()) + mask_finished * finished.float()
            cur_log_probs = log_probs.view(-1, 1) + torch.log_softmax(masked_logits, -1)
            
            #topk_log_probs: B x cur_beam_size
            #topk_ids: B x cur_beam_size    
            #y: (B x cur_beam_size) x 1
            #probs: (B x cur_beam_size) x 1
            topk_log_probs, topk_ids = cur_log_probs.view(cur_batch_size, (last_beam_size * vocab_size)).topk(cur_beam_size)
            y = (topk_ids % vocab_size).view(-1, 1)            
            probs = torch.gather(probs.view(cur_batch_size, (last_beam_size * vocab_size)), 1, topk_ids).view(-1, 1)
            
            #base_id: B
            #beam_id: (B x cur_beam_size)
            base_id = torch.arange(0, cur_batch_size, device = y.device) * last_beam_size
            beam_id = (base_id.view(-1, 1) + topk_ids // vocab_size).view(-1)
            
            cur_log_probs = topk_log_probs.view(-1)
            
            cache = model.gather_cache(cache, beam_id)    
        else:
            replacement = not (group_size > 0 and steps == 0)
            logits = logits * (1 - finished.float()) + mask_finished * finished.float()
            y,probs = top_k_top_p_sampling(logits,
                                           top_k,
                                           top_p,
                                           temperature,
                                           n_samples=cur_beam_size,
                                           replacement=replacement)
            
            base_id = torch.arange(0, cur_batch_size, device = y.device) * last_beam_size
            beam_id = (base_id.view(-1, 1) + y // vocab_size).view(-1)

            y = y.view(-1, 1)
            probs = probs.view(-1, 1)
            
            cur_log_probs = log_probs[beam_id] + torch.log(probs)
            
            cache = model.gather_cache(cache, beam_id) 
            
        if strategy == "beam_search" or last_beam_size != cur_beam_size:
            finished = finished[beam_id,:]
            hypothesis = hypothesis[beam_id,:]
            history_probs = history_probs[beam_id, :] 
                
        finished = (finished | y.eq(eos).byte())        
        hypothesis = torch.cat([hypothesis, y], 1)
        history_probs = torch.cat([history_probs, probs], 1)

        if strategy == "beam_search":
            if group_size > 0:
                if steps == 0:
                    cur_batch_size = batch_size * group_size
                    cur_beam_size = beam_size // group_size
                else:
                    last_beam_size = cur_beam_size                
            else:
                last_beam_size = cur_beam_size
                cur_batch_size = batch_size  
                cur_beam_size = beam_size
        elif strategy == "sample":
            if group_size > 0:
                if steps == 0:
                    cur_batch_size = batch_size * group_size
                    cur_beam_size = beam_size // group_size
                else:                    
                    cur_batch_size = batch_size * beam_size
                    cur_beam_size = 1
            else:
                cur_batch_size = batch_size * beam_size
                cur_beam_size = 1

        states = [y, cur_log_probs, finished, mask_finished, hypothesis, history_probs]
        steps += 1
        if finished.all() or (max_decode_steps is not None and steps >= max_decode_steps):
            break

    return states, cache
    

def crf_model_decoding(model, x):
    """
    """
    emission = model.get_emission(x)
    
    mask = x.ne(model.PAD).float()
    
    crf, emission, mask, pad_tag = model.crf, emission, mask, model.PAD

    batch_size, seq_len, n_labels = emission.size()
    
    if mask is not None:
        end_mask = crf.get_end_mask(mask).float()
    
    scores = crf.start_trans + emission[:, 0, :]
    if mask is not None:
        scores = scores + end_mask[:, 0:1] * crf.end_trans
    path_table = torch.zeros(batch_size, seq_len-1, n_labels, dtype=torch.long, device=x.device)
    
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