# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 09:38:03 2020

@author: Xiaoyuan Yao
"""
import torch
import torch.nn.functional as F

def cross_entropy_with_smoothing(logits, target, eps, pad):
    """
    """
    n_class = logits.size(1)
    one_hot = torch.zeros_like(logits).scatter(1, target.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_probs = F.log_softmax(logits, dim=1)
    loss = -torch.sum(one_hot * log_probs, dim=1)

    if pad is not None:
        mask = target.view(-1).ne(pad)
        loss = torch.sum(loss * mask.float()) / torch.sum(mask.float())
    else:
        loss = torch.mean(loss)
    
    return loss
    

def seq_cross_entropy(logits, target, eps, pad):
    """
    """
    if logits.dim() == 3:
        logits = torch.flatten(logits, start_dim=0, end_dim=1)
    if eps > 0:
        loss = cross_entropy_with_smoothing(logits, target, eps, pad)
    else:
        loss = F.cross_entropy(logits, 
                               target.view(-1), 
                               ignore_index=pad)
    return loss


def classify_loss(logits, target, eps):
    """
    """
    if eps > 0:
        loss = cross_entropy_with_smoothing(logits, target, eps, None)
    else:
        loss = F.cross_entropy(logits, target.view(-1))
            
    return loss


def kl_loss(logits, soft_target, target, pad, temperature):
    """
    """
    if logits.dim() == 3:
        logits = logits.view(-1, logits.size(-1))
    if soft_target.dim() == 3:
        soft_target = soft_target.view(-1, soft_target.size(-1))
        
    kl_loss = F.kl_div(F.log_softmax(logits/temperature, dim=1),
                         F.softmax(soft_target/temperature, dim=1),
                         reduction="none")
        
    mask = target.ne(pad).view(-1, 1).float()

    kl_loss = torch.sum(kl_loss * mask) / torch.sum(mask)
    kl_loss = kl_loss * temperature * temperature

    return kl_loss


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    """
    """
    kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                - torch.pow(prior_mu - recog_mu, 2) / torch.exp(prior_logvar)
                - torch.exp(recog_logvar) / torch.exp(prior_logvar),
                1)
    return kld


def build_lm_loss(model_config, train_config):
    """
    """
    eps = train_config.get("eps", 0)
    pad = model_config["symbol2id"]["_pad_"]
    return lambda x,y:seq_cross_entropy(x[0], y[0], eps, pad)
 

def build_classify_loss(model_config, train_config):
    """
    """
    eps = train_config.get("eps", 0)
    return lambda x,y:classify_loss(x[0], y[0], eps)

                
def build_sequence_labeling_loss(model_config, train_config):
    """
    """
    eps = train_config.get("eps", 0)
    pad = model_config["symbol2id"]["_pad_"]
    if "crf" in model_config["model"]:
        return lambda x,y:x[0]
    else:
        return lambda x,y:seq_cross_entropy(x[0], y[0], eps, pad)


def build_match_loss(model_config, train_config):
    """
    """
    eps = train_config.get("eps", 0)
    pad = model_config["symbol2id"]["_pad_"]
    return lambda x,y:seq_cross_entropy(train_config["sim_alpha"] * x[0], y[0], eps, pad)


loss_builder_dict = {
     "enc_dec": build_lm_loss,
     "lm": build_lm_loss,
     "bi_lm": build_lm_loss,
     "sequence_labeling": build_sequence_labeling_loss,
     "classify": build_classify_loss,
     "match": build_match_loss
}

def build_loss_fn(model_config, train_config):
    """
    """
    if model_config["task"] in loss_builder_dict:
        return loss_builder_dict[model_config["task"]](model_config, train_config)
    else:
        raise ValueError("model not correct!")
