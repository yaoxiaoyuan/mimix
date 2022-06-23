# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 10:39:14 2019

@author: lyn
"""
import abc
import os
import random
import numpy as np
import torch
from tokenization import build_tokenizer
from utils import real_path, load_vocab

class DataProcessor():
    """
    """
    def __init__(self):
        """
        """
        self.custom_parse_fn = None
        
    @abc.abstractmethod
    def parse(line):
        """
        """
        return None

    @abc.abstractmethod
    def preprocess(line):
        """
        """
        return None

    
    def __call__(self, line):
        """
        """
        if self.custom_parse_fn is None:
            parsed = self.parse(line)
        else:
            parsed = self.custom_parse_fn(line)
        
        processed = None
        if parsed is not None:
            processed = self.preprocess(parsed)
        
        return processed


class S2SDataProcessor(DataProcessor):
    """
    """
    def __init__(self, 
                 src_max_len, 
                 trg_max_len, 
                 src_tokenizer, 
                 trg_tokenizer, 
                 symbol2id):
        """
        """
        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        
        self.src_tokenizer = src_tokenizer 
        self.trg_tokenizer = trg_tokenizer 
        
        self.custom_parse_fn = None
        
        
    def parse(self, line):
        """
        """        
        try:
            arr = line.strip().split("\t")
            src,trg = arr[0],arr[1]
        except:
            return None
        
        return [src, trg]


    def preprocess(self, data):
        """
        """        
        src,trg = data
        x = self.src_tokenizer.tokenize_to_ids(src)
        y = self.trg_tokenizer.tokenize_to_ids(trg)

        x = x[:self.src_max_len]
        y = y[:self.trg_max_len - 1]
                        
        y = [self.BOS] + y + [self.EOS]
            
        return {"src":x, "trg":y}

    
class LMDataProcessor(DataProcessor):
    """
    """
    def __init__(self, 
                 trg_max_len, 
                 trg_tokenizer, 
                 symbol2id):
        """
        """
        self.trg_max_len = trg_max_len
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        
        self.trg_tokenizer = trg_tokenizer
        self.custom_parse_fn = None
        

    def parse(self, line):
        """
        """
        trg = line.strip()
        if len(trg) == 0:
            return None
        return [trg]


    def preprocess(self, data):
        """
        """
        trg = data[0]
        y = self.trg_tokenizer.tokenize_to_ids(trg)
            
        y = y[:self.trg_max_len - 1]
        y = [self.BOS] + y + [self.EOS]
        
        return {"trg":y}


class ClassifyDataProcessor(DataProcessor):
    """
    """
    def __init__(self, 
                 src_max_len, 
                 label2id, 
                 n_class, 
                 symbol2id, 
                 src_tokenizer):
        """
        """
        self.src_max_len = src_max_len
        self.label2id = label2id
        self.num_class = n_class
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        self.src_tokenizer = src_tokenizer
        self.custom_parse_fn = None
        

    def parse(self, line):
        """
        """
        try:
            src,label = line.strip().split("\t")[:2]
        except:
            return None
        
        return [src, label]


    def preprocess(self, data):
        """
        """        
        src,label = data
        x = self.src_tokenizer.tokenize_to_ids(src)

        x = [self.CLS] + x[:self.src_max_len - 1]
        if self.label2id is not None:
            y = self.label2id[label]
        else:
            y = int(label)
        
        return {"src":x, "label":y}


class SequenceLabelingDataProcessor(DataProcessor):
    """
    """
    def __init__(self, 
                 src_max_len, 
                 label2id, 
                 n_labels, 
                 symbol2id, 
                 src_tokenizer):
        """
        """
        self.src_max_len = src_max_len
        self.label2id = label2id
        self.n_labels = n_labels
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        self.src_tokenizer = src_tokenizer
        self.custom_parse_fn = None
        
 
    def parse(self, line):
        """
        """
        try:
            src,label = line.strip().split("\t")[:2]
        except:
            return None
        return [src, label]


    def preprocess(self, data):
        """
        """        
        src, label = data
        x = self.src_tokenizer.tokenize_to_ids(src)

        x = x[:self.src_max_len]
        y = [self.label2id[s] for s in label.split()][:self.src_max_len]
        
        return {"src":x, "label":y}
        
    
class BiLMDataProcessor(DataProcessor):
    """
    """
    def __init__(self, 
                 trg_max_len, 
                 symbol2id, 
                 trg_tokenizer, 
                 mask_rate):
        """
        """
        self.trg_max_len = trg_max_len
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        self.mask_rate = mask_rate
        self.trg_tokenizer = trg_tokenizer
        self.custom_parse_fn = None
        

    def parse(self, line):
        """
        """
        trg = line.strip()
        if len(trg) == 0:
            return None
        return [trg]


    def preprocess(self, data):
        """
        """
        trg = data[0]
        
        y = self.trg_tokenizer.tokenize_to_ids(trg)
        
        return {"trg":y}


def build_data_processor(train_config, model_config):
    """
    """
    if model_config["task"] == "enc_dec":
        
        src_tokenizer = build_tokenizer(
                tokenizer=model_config["src_tokenizer"],
                vocab_file=model_config["src_vocab"], 
                pre_tokenized=train_config.get("pre_tokenized",False),  
                pre_vectorized=train_config.get("pre_vectorized",False))
        trg_tokenizer = build_tokenizer(
                tokenizer=model_config["trg_tokenizer"],
                vocab_file=model_config["trg_vocab"], 
                pre_tokenized=train_config.get("pre_tokenized",False),  
                pre_vectorized=train_config.get("pre_vectorized",False))
        
        return S2SDataProcessor(
                model_config["src_max_len"], 
                model_config["trg_max_len"], 
                src_tokenizer, 
                trg_tokenizer,
                model_config["symbol2id"])
        
    elif model_config["task"] == "lm":
        trg_tokenizer = build_tokenizer(
                tokenizer=model_config["trg_tokenizer"],
                vocab_file=model_config["trg_vocab"], 
                pre_tokenized=train_config.get("pre_tokenized",False),  
                pre_vectorized=train_config.get("pre_vectorized",False))
        
        return LMDataProcessor(
                model_config["trg_max_len"], 
                trg_tokenizer,
                model_config["symbol2id"])
        
    elif model_config["task"] == "classify":
        src_tokenizer = build_tokenizer(
                tokenizer=model_config["src_tokenizer"],
                vocab_file=model_config["src_vocab"], 
                pre_tokenized=train_config.get("pre_tokenized",False),  
                pre_vectorized=train_config.get("pre_vectorized",False))
        
        label2id = None
        if "label2id" in model_config:
            label2id = load_vocab(real_path(model_config["label2id"]))
        
        return ClassifyDataProcessor(
                model_config["src_max_len"], 
                label2id, 
                model_config["n_class"],
                model_config["symbol2id"], 
                src_tokenizer)

    elif model_config["task"] == "bi-lm":
        trg_tokenizer = build_tokenizer(
                tokenizer=model_config["trg_tokenizer"],
                vocab_file=model_config["trg_vocab"], 
                pre_tokenized=train_config.get("pre_tokenized",False),  
                pre_vectorized=train_config.get("pre_vectorized",False))
        
        return BiLMDataProcessor(
                model_config["trg_max_len"], 
                trg_tokenizer, 
                model_config["symbol2id"], 
                model_config["mask_rate"])

    elif model_config["task"] == "sequence_labeling":
        src_tokenizer = build_tokenizer(
                tokenizer=model_config["src_tokenizer"],
                vocab_file=model_config["src_vocab"], 
                pre_tokenized=train_config.get("pre_tokenized",False),  
                pre_vectorized=train_config.get("pre_vectorized",False))

        label2id = load_vocab(real_path(model_config["label2id"]))
        
        return SequenceLabelingDataProcessor(
                model_config["src_max_len"], 
                label2id, 
                model_config["n_labels"],
                model_config["symbol2id"], 
                src_tokenizer)
