# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 10:39:14 2019

@author: lyn
"""
import os
import random
import numpy as np
import torch
from tokenization import build_tokenizer
from utils import real_path, load_vocab

class Dataset():
    """
    """
    def __init__(self):
        """
        """
        pass
    
    
    def vectorize(self, batch_data):
        """
        """
        return batch_data
    
    
    def preprocess(line):
        """
        """
        return [line]
    
    
    def __call__(self, start_steps=0):
        """
        """
        data = []
        files = os.listdir(self.data_dir)
        files.sort()
        
        steps = 1
        for fi in files:
            fi = os.path.join(self.data_dir, fi)
            for line in open(fi, "r", encoding="utf-8", errors="ignore"):
                steps += 1
                if steps < start_steps * self.batch_size:
                    continue
                
                processed = self.preprocess(line)
                if len(processed) > 0:
                    data.append(processed)
                if len(data) % (20 * self.batch_size) == 0:
                    batch_data = data[:self.batch_size]
                    data = data[self.batch_size:]
                    yield self.vectorize(batch_data)
                
        while len(data) > 0:
            batch_data = data[:self.batch_size]
            yield self.vectorize(batch_data)              
            data = data[self.batch_size:]


def word_dropout(word_list, rate, replace_token):
    """
    """
    if rate > 0:
        tmp = []
        
        for word in word_list:
            if random.random() < rate:
                tmp.append(replace_token)
            else:
                tmp.append(word)
        
        word_list = tmp
        
    return word_list


class S2SDataset(Dataset):
    """
    """
    def __init__(self, 
                 data_dir,
                 batch_size,
                 src_max_len, 
                 trg_max_len, 
                 src_tokenizer, 
                 trg_tokenizer, 
                 symbol2id, 
                 word_dropout):
        """
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
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
        
        self.word_dropout = word_dropout
        
    
    def vectorize(self, batch_data):
        """
        """
        batch_size = len(batch_data)
        x = self.PAD + np.zeros((batch_size, self.src_max_len), dtype=np.long)
        y = self.PAD + np.zeros((batch_size, self.trg_max_len), dtype=np.long)
        y_target = self.PAD + np.zeros((batch_size, self.trg_max_len), 
                                       dtype=np.long)
        
        for i, (xx, yy, src, trg) in enumerate(batch_data):
            x[i, :len(xx)] = xx
            y[i, :len(yy) - 1] = word_dropout(
                    yy[:-1], self.word_dropout, self.UNK)
            y_target[i, :len(yy) - 1] = yy[1:]
        
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        y_target = torch.tensor(y_target, dtype=torch.long)
        
        return [x, y], [y_target]
        
    
    def preprocess(self, line):
        """
        """
        try:
            arr = line.strip().split("\t")
            src,trg = arr[0],arr[1]
        except:
            return []
        
        x = self.src_tokenizer.tokenize_to_ids(src)
        y = self.trg_tokenizer.tokenize_to_ids(trg)

        x = x[:self.src_max_len]
        y = y[:self.trg_max_len - 1]
                        
        y = [self.BOS] + y + [self.EOS]
            
        return [x, y, src, trg]


class LMDataset(Dataset):
    """
    """
    def __init__(self, 
                 data_dir,
                 batch_size,
                 trg_max_len, 
                 trg_tokenizer, 
                 symbol2id,
                 word_dropout):
        """
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.trg_max_len = trg_max_len
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        
        self.trg_tokenizer = trg_tokenizer
        self.word_dropout = word_dropout

        
    def vectorize(self, batch_data):
        """
        """
        batch_size = len(batch_data)
        y = self.PAD + np.zeros((batch_size, self.trg_max_len), dtype=np.long)
        y_target = self.PAD + np.zeros((batch_size, self.trg_max_len), 
                                       dtype=np.long)
        
        for i, (yy, trg) in enumerate(batch_data):
            y[i, :len(yy) - 1] = word_dropout(
                    yy[:-1], self.word_dropout, self.UNK)
            y_target[i, :len(yy) - 1] = yy[1:]
            
        y = torch.tensor(y, dtype=torch.long)
        y_target = torch.tensor(y_target, dtype=torch.long)
        
        return [y], [y_target]
        
    
    def preprocess(self, line):
        """
        """
        trg = line.strip()

        y = self.trg_tokenizer.tokenize_to_ids(trg)
            
        y = y[:self.trg_max_len - 1]
        y = [self.BOS] + y + [self.EOS]
        
        return [y, trg]


class ClassifyDataset(Dataset):
    """
    """
    def __init__(self, 
                 data_dir,
                 batch_size,
                 src_max_len, 
                 label2id, 
                 n_class, 
                 symbol2id, 
                 src_tokenizer, 
                 word_dropout):
        """
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
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
        self.word_dropout = word_dropout
        
        
    def vectorize(self, batch_data):
        """
        """
        batch_size = len(batch_data)
        x = self.PAD + np.zeros((batch_size, self.src_max_len), dtype=np.long)
        
        y = np.zeros((batch_size, 1), dtype=np.long)
        
        for i, (xx, yy, src, trg) in enumerate(batch_data):
            x[i, :len(xx)] = word_dropout(xx, self.word_dropout, self.UNK)
            y[i] = yy
        
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        
        return [x], [y]
        
    
    def preprocess(self, line):
        """
        """
        try:
            src,label = line.strip().split("\t")[:2]
        except:
            return []
        
        x = self.src_tokenizer.tokenize_to_ids(src)

        x = [self.CLS] + x[:self.src_max_len - 1]
        if self.label2id is not None:
            y = self.label2id[label]
        else:
            y = int(label)
        
        return [x, y, src, label]


class SequenceLabelingDataset(Dataset):
    """
    """
    def __init__(self, 
                 data_dir,
                 batch_size,
                 src_max_len, 
                 label2id, 
                 n_labels, 
                 symbol2id, 
                 src_tokenizer):
        """
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
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
        
        
    def vectorize(self, batch_data):
        """
        """
        batch_size = len(batch_data)
        x = self.PAD + np.zeros((batch_size, self.src_max_len), dtype=np.long)
        
        y = self.PAD + np.zeros((batch_size, self.src_max_len), dtype=np.long)
        
        for i, (xx, yy, src, trg) in enumerate(batch_data):
            x[i, :len(xx)] = xx
            y[i, :len(yy)] = yy
        
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        
        return [x, y], [y]
        
    
    def preprocess(self, line):
        """
        """
        try:
            src,label = line.strip().split("\t")[:2]
        except:
            return []
        
        x = self.src_tokenizer.tokenize_to_ids(src)

        x = x[:self.src_max_len]
        y = [self.label2id[s] for s in label.split()][:self.src_max_len]
        
        return [x, y, src, label]


class BiLMDataset(Dataset):
    """
    """
    def __init__(self, 
                 data_dir,
                 batch_size,
                 trg_max_len, 
                 symbol2id, 
                 trg_tokenizer, 
                 mask_rate):
        """
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
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
        
        
    def vectorize(self, batch_data):
        """
        """
        batch_size = len(batch_data)
        y = self.PAD + np.zeros((batch_size, self.trg_max_len), dtype=np.long)
        y_target = self.PAD + np.zeros((batch_size, self.trg_max_len), 
                                       dtype=np.long)
        
        for i, (yy,tt) in enumerate(batch_data):
            for j, w in enumerate(yy):
                if random.random() < self.mask_rate:
                    y[i, j] = self.MASK
                    y_target[i, j] = w
                else:
                    y[i, j] = w
                    y_target[i, j] = self.PAD
        
        y_target = y_target.reshape(-1)
        y_target = y_target[y_target != self.PAD]
        
        y = torch.tensor(y, dtype=torch.long)
        y_target = torch.tensor(y_target, dtype=torch.long)
        
        return [y], [y_target]
        
    
    def preprocess(self, line):
        """
        """
        trg = line.strip()
        
        y = self.trg_tokenizer.tokenize_to_ids(trg)
        
        return y,trg


def build_dataset(train_config, model_config, dataset="train"):
    """
    """
    if dataset == "train":
        data_dir = train_config["train_dir"]
        batch_size = train_config["batch_size"]
    elif dataset == "val":
        data_dir = train_config["val_dir"]
        batch_size = train_config["test_batch_size"]
    elif dataset == "test":
        data_dir = train_config["test_dir"]
        batch_size = train_config["test_batch_size"]
        
    if model_config["task"] == "enc_dec":
        
        src_tokenizer = build_tokenizer(
                tokenizer=model_config["src_tokenizer"],
                vocab_file=model_config["src_vocab"], 
                pre_tokenized=model_config.get("pre_tokenized",False),  
                pre_vectorized=model_config.get("pre_vectorized",False))
        trg_tokenizer = build_tokenizer(
                tokenizer=model_config["trg_tokenizer"],
                vocab_file=model_config["trg_vocab"], 
                pre_tokenized=model_config.get("pre_tokenized",False),  
                pre_vectorized=model_config.get("pre_vectorized",False))
        
        return S2SDataset(
                data_dir,
                batch_size,
                model_config["src_max_len"], 
                model_config["trg_max_len"], 
                src_tokenizer, 
                trg_tokenizer,
                model_config["symbol2id"], 
                model_config.get("word_dropout", 0))
        
    elif model_config["task"] == "lm":
        trg_tokenizer = build_tokenizer(
                tokenizer=model_config["trg_tokenizer"],
                vocab_file=model_config["trg_vocab"], 
                pre_tokenized=model_config.get("pre_tokenized",False),  
                pre_vectorized=model_config.get("pre_vectorized",False))
        
        return LMDataset(
                data_dir,
                batch_size,
                model_config["trg_max_len"], 
                trg_tokenizer,
                model_config["symbol2id"], 
                model_config.get("word_dropout", 0))
        
    elif model_config["task"] == "classify":
        src_tokenizer = build_tokenizer(
                tokenizer=model_config["src_tokenizer"],
                vocab_file=model_config["src_vocab"], 
                pre_tokenized=model_config.get("pre_tokenized",False),  
                pre_vectorized=model_config.get("pre_vectorized",False))
        
        label2id = None
        if "label2id" in model_config:
            label2id = load_vocab(real_path(model_config["label2id"]))
        
        return ClassifyDataset(
                data_dir,
                batch_size,
                model_config["src_max_len"], 
                label2id, 
                model_config["n_class"],
                model_config["symbol2id"], 
                src_tokenizer, 
                model_config.get("word_dropout", 0))

    elif model_config["task"] == "bi-lm":
        trg_tokenizer = build_tokenizer(
                tokenizer=model_config["trg_tokenizer"],
                vocab_file=model_config["trg_vocab"], 
                pre_tokenized=model_config.get("pre_tokenized",False),  
                pre_vectorized=model_config.get("pre_vectorized",False))
        
        return BiLMDataset(
                data_dir,
                batch_size,
                model_config["trg_max_len"], 
                trg_tokenizer, 
                model_config["symbol2id"], 
                model_config["mask_rate"],
                model_config.get("word_dropout", 0))

    elif model_config["task"] == "sequence_labeling":
        src_tokenizer = build_tokenizer(
                tokenizer=model_config["src_tokenizer"],
                vocab_file=model_config["src_vocab"], 
                pre_tokenized=model_config.get("pre_tokenized",False),  
                pre_vectorized=model_config.get("pre_vectorized",False))

        label2id = load_vocab(real_path(model_config["label2id"]))
        
        return SequenceLabelingDataset(
                data_dir,
                batch_size,
                model_config["src_max_len"], 
                label2id, 
                model_config["n_labels"],
                model_config["symbol2id"], 
                src_tokenizer)


def build_train_dataset(train_config, model_config):
    """
    """
    return build_dataset(train_config, model_config, "train")


def build_val_dataset(train_config, model_config):
    """
    """
    return build_dataset(train_config, model_config, "val")


def build_test_dataset(train_config, model_config):
    """
    """
    return build_dataset(train_config, model_config, "test")
