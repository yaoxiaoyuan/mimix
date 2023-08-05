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
    one_hot = torch.eye(n_class, device=target.device)[target]
    target = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    loss = F.cross_entropy(logits, target, reduction="none")

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


def contrastive_loss(vec, target, sim_alpha):
    """
    """
    norm_vec = F.normalize(vec, p=2, dim=1)
    sim = torch.mm(norm_vec, norm_vec.T)
    sim = torch.masked_fill(sim, torch.eye(sim.shape[0], device=sim.device).bool(), -1000)
    loss = F.cross_entropy(sim_alpha * sim, target)

    return loss

def classify_loss(logits, target, eps):
    """
    """
    if eps > 0:
        loss = cross_entropy_with_smoothing(logits, target, eps, None)
    else:
        loss = F.cross_entropy(logits, target)
            
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

