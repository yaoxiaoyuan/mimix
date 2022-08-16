# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:17:36 2019

@author: Xiaoyuan Yao
"""
import numpy as np
import torch
import torch.nn.functional as F
from utils import nested_to_cuda
from decoding import crf_model_decoding

def eval_acc(trainer, dataset="val"):
    """
    """
    trainer.model.eval()
    with torch.no_grad():
        shot_count = 0
        total_count = 0
        for inputs,targets in trainer.val_dataset():
            inputs = nested_to_cuda(inputs, trainer.device)
            targets = nested_to_cuda(targets, trainer.device)
                
            outputs = trainer.model(inputs)
            pred = outputs[0]
            shot = torch.sum(pred.argmax(1) == targets[0].view(-1))
                
            shot_count = shot_count + shot.item()
            total_count = total_count + targets[0].size(0)
        
    acc = shot_count / total_count
    trainer.logger.info("acc:%f" % acc)
    return acc


def eval_perplexity(trainer, dataset="val"):
    """
    """
    trainer.model.eval()
    with torch.no_grad():
        sum_log_p = 0
        sum_len = 0
        for inputs,targets in trainer.val_dataset():
            inputs = nested_to_cuda(inputs, trainer.device)
            targets = nested_to_cuda(targets, trainer.device)            
            outputs = trainer.model(inputs)
            logits = outputs[0]

            log_probs = torch.gather(F.log_softmax(logits, 2), 
                                     2, 
                                     targets[0].unsqueeze(-1))
            
            mask = (inputs[0] != trainer.model.PAD).float()
            seq_len = torch.sum(mask)
            log_probs = torch.sum(mask * log_probs.squeeze(-1))
            sum_log_p = sum_log_p + log_probs.item()
            sum_len = sum_len + seq_len.item()

    perplexity = np.exp(-sum_log_p / sum_len)
    trainer.logger.info("ppl:%f" % perplexity)
    return perplexity


def eval_sequence_labeling_acc(trainer, dataset="val"):
    """
    """
    trainer.model.eval()
    with torch.no_grad():
        shot_count = 0
        total_count = 0
        for inputs,targets in trainer.val_dataset():
            inputs = nested_to_cuda(inputs, trainer.device)
            targets = nested_to_cuda(targets, trainer.device)
            x = inputs[0]
            if trainer.model.use_crf == True:
                pred = crf_model_decoding(trainer.model, x)
            else:
                pred = trainer.model(inputs)[0].argmax(-1)
            shot = torch.sum((pred == targets[0]) * (x != trainer.model.PAD))
            
            shot_count = shot_count + shot.item()
            total_count = total_count + torch.sum(x != trainer.model.PAD).item()
        
    acc = shot_count / total_count
    trainer.logger.info("acc:%f" % acc)
    return acc


def build_eval_fn(model_config):
    """
    """
    if model_config["task"] in ["lm"]:
        return lambda trainer:eval_perplexity(trainer)
    elif model_config["task"] in ["classify"]:
        return lambda trainer:eval_acc(trainer)
    elif model_config["task"] in ["sequence_labeling"]:
        return lambda trainer:eval_sequence_labeling_acc(trainer)
    else:
        return None

