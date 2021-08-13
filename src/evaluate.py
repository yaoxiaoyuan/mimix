# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:17:36 2019

@author: lyn
"""
import collections
import math
import sys
import numpy as np
import torch
import torch.nn.functional as F
from utils import real_path, parse_args, load_config, nested_to_cuda
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
        for inputs,target in trainer.val_dataset():
            inputs = nested_to_cuda(inputs, trainer.device)
            targets = nested_to_cuda(targets, trainer.device)            
            outputs = trainer.model(inputs)
            logits = outputs[0]

            log_probs = torch.gather(F.log_softmax(logits, 2), 
                                     2, 
                                     target.unsqueeze(-1))
            
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


def _get_ngrams(segment, max_order):
    """Extracts all n-grams up to a given maximum order from an input segment.
    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.
    Returns:
      The Counter containing all n-grams up to max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
      for i in range(0, len(segment) - order + 1):
        ngram = tuple(segment[i:i + order])
        ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus,
                 translation_corpus,
                 max_order=4,
                 use_bp=True):
    """Computes BLEU score of translated segments against one or more references.
    Args:
      reference_corpus: list of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      use_bp: boolean, whether to apply brevity penalty.
    Returns:
      BLEU score.
    """
    reference_length = 0
    translation_length = 0
    bp = 1.0
    geo_mean = 0
    
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    precisions = []
    
    for (references, translations) in zip(reference_corpus, translation_corpus):
      reference_length += len(references)
      translation_length += len(translations)
      ref_ngram_counts = _get_ngrams(references, max_order)
      translation_ngram_counts = _get_ngrams(translations, max_order)
    
      overlap = dict((ngram,
                      min(count, translation_ngram_counts[ngram]))
                     for ngram, count in ref_ngram_counts.items())
    
      for ngram in overlap:
        matches_by_order[len(ngram) - 1] += overlap[ngram]
      for ngram in translation_ngram_counts:
        possible_matches_by_order[len(ngram)-1] += translation_ngram_counts[ngram]
    precisions = [0] * max_order
    smooth = 1.0
    for i in range(0, max_order):
      if possible_matches_by_order[i] > 0:
        precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
        if matches_by_order[i] > 0:
          precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
        else:
          smooth *= 2
          precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
      else:
        precisions[i] = 0.0
    
    if max(precisions) > 0:
      p_log_sum = sum(math.log(p) for p in precisions if p)
      geo_mean = math.exp(p_log_sum/max_order)
    
    if use_bp:
      if not reference_length:
        bp = 1.0
      else:
        ratio = translation_length / reference_length
        if ratio <= 0.0:
          bp = 0.0
        elif ratio >= 1.0:
          bp = 1.0
        else:
          bp = math.exp(1 - 1. / ratio)
    bleu = geo_mean * bp
    return np.float32(bleu)


def run_bleu(ref_file, hyp_file):
    """
    """
    src_list = []
    ref_tokens = {}
    for line in open(ref_file, "r", encoding="utf-8"):
        src,trg = line.strip().split("\t")[0:2]
        if src not in ref_tokens:
            ref_tokens[src] = []
        ref_tokens[src].append(trg)
        src_list.append(src)
    
    hyp_tokens = {}
    for line in open(hyp_file, "r", encoding="utf-8"):
        if line.startswith("#"):
            continue
        src,trg = line.strip().split("\t")[0:2]
        hyp_tokens[src] = [trg]
    
    ref_tokens = [ref_tokens[src] for src in src_list]
    hyp_tokens = [hyp_tokens[src] for src in src_list]
    
    assert len(ref_tokens) == len(hyp_tokens)
    
    bleu_1 = compute_bleu(ref_tokens, hyp_tokens, 1)
    print("bleu_1:", bleu_1)
    
    bleu_2 = compute_bleu(ref_tokens, hyp_tokens, 2)
    print("bleu_2:", bleu_2)
    
    bleu_3 = compute_bleu(ref_tokens, hyp_tokens, 3)
    print("bleu_3:", bleu_3)
    
    bleu_4 = compute_bleu(ref_tokens, hyp_tokens, 4)
    print("bleu_4:", bleu_4)


def run_evaluate():
    """
    """
    usage = "usage: run_evaluate --conf <conf_file>"
    
    options = parse_args(usage)
    config_file = options.config
    params = load_config(real_path(config_file))
    
    ref_file = params["test_in"]
    hyp_file = params["test_out"]

    run_bleu(ref_file, hyp_file)


if __name__ == "__main__":
    run_evaluate()
    