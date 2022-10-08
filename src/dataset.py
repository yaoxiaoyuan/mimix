# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 10:39:14 2019

@author: Xiaoyuan Yao
"""
import os
import json
import random
import numpy as np
from abc import ABC, abstractmethod
import torch
from utils import real_path, load_vocab

class Dataset(ABC):
    """
    """
    def __init__(self, rank=0, world_size=1):
        """
        """
        self.rank = rank
        self.world_size = world_size
    

    @abstractmethod
    def vectorize(self, batch_data):
        """
        """
        pass

    
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
                if steps % self.world_size != self.rank:
                    continue
                data.append(json.loads(line))
                if len(data) % (20 * self.batch_size) == 0:
                    batch_data = data[:self.batch_size]
                    data = data[self.batch_size:]
                    yield self.vectorize(batch_data)
                
        while len(data) > 0:
            batch_data = data[:self.batch_size]
            yield self.vectorize(batch_data)              
            data = data[self.batch_size:]


class S2SDataset(Dataset):
    """
    """
    def __init__(self, 
                 data_dir,
                 batch_size,
                 symbol2id,
                 rank=0,
                 world_size=1):
        """
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        self.rank = rank
        self.world_size = world_size        
        
    
    def vectorize(self, batch_data):
        """
        """
        batch_size = len(batch_data)
        src_max_len = max(len(x["src"]) for x in batch_data)
        trg_max_len = max(len(x["trg"]) - 1 for x in batch_data)
        x = self.PAD + np.zeros((batch_size, src_max_len), dtype=np.long)
        y = self.PAD + np.zeros((batch_size, trg_max_len), dtype=np.long)
        y_target = self.PAD + np.zeros((batch_size, trg_max_len), 
                                       dtype=np.long)
        
        for i, d in enumerate(batch_data):
            xx,yy = d["src"],d["trg"]
            x[i, :len(xx)] = xx
            y[i, :len(yy) - 1] = yy[:-1]
            y_target[i, :len(yy) - 1] = yy[1:]
        
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        y_target = torch.tensor(y_target, dtype=torch.long)
        
        return [x, y], [y_target]


class LMDataset(Dataset):
    """
    """
    def __init__(self, 
                 data_dir,
                 batch_size, 
                 symbol2id,
                 rank=0,
                 world_size=1):
        """
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        self.rank = rank
        self.world_size = world_size        
        
        
    def vectorize(self, batch_data):
        """
        """
        batch_size = len(batch_data)
        trg_max_len = max(len(x["trg"]) - 1 for x in batch_data)
        y = self.PAD + np.zeros((batch_size, trg_max_len), dtype=np.long)
        y_target = self.PAD + np.zeros((batch_size, trg_max_len), 
                                       dtype=np.long)
        
        for i, d in enumerate(batch_data):
            yy = d["trg"]
            y[i, :len(yy) - 1] = yy[:-1]
            y_target[i, :len(yy) - 1] = yy[1:]
            
        y = torch.tensor(y, dtype=torch.long)
        y_target = torch.tensor(y_target, dtype=torch.long)
        
        return [y], [y_target]


class ClassifyDataset(Dataset):
    """
    """
    def __init__(self, 
                 data_dir,
                 batch_size,
                 symbol2id,
                 rank=0,
                 world_size=1):
        """
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        self.rank = rank
        self.world_size = world_size        
                
        
    def vectorize(self, batch_data):
        """
        """
        batch_size = len(batch_data)
        src_max_len = max(len(x["src"]) for x in batch_data)
        x = self.PAD + np.zeros((batch_size, src_max_len), dtype=np.long)
        
        y = np.zeros((batch_size, 1), dtype=np.long)
        
        for i, d in enumerate(batch_data):
            xx,yy = d["src"],d["label"]
            x[i, :len(xx)] = xx
            y[i] = yy
        
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        
        return [x], [y]


class SequenceLabelingDataset(Dataset):
    """
    """
    def __init__(self, 
                 data_dir,
                 batch_size,
                 symbol2id,
                 rank=0,
                 world_size=1):
        """
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        self.rank = rank
        self.world_size = world_size        
        
       
    def vectorize(self, batch_data):
        """
        """
        batch_size = len(batch_data)
        src_max_len = max(len(x["src"]) for x in batch_data)
        x = self.PAD + np.zeros((batch_size, src_max_len), dtype=np.long)
        y = self.PAD + np.zeros((batch_size, src_max_len), dtype=np.long)
        
        for i, d in enumerate(batch_data):
            xx,yy = d["src"],d["label"]
            x[i, :len(xx)] = xx
            y[i, :len(yy)] = yy
        
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        
        return [x, y], [y]
        
    
class BiLMDataset(Dataset):
    """
    """
    def __init__(self, 
                 data_dir,
                 batch_size,
                 symbol2id, 
                 mask_rate,
                 rank=0,
                 world_size=1):
        """
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        self.mask_rate = mask_rate
        self.rank = rank
        self.world_size = world_size        
        
        
    def vectorize(self, batch_data):
        """
        """
        batch_size = len(batch_data)
        trg_max_len = max(len(x["trg"]) for x in batch_data)
        y = self.PAD + np.zeros((batch_size, trg_max_len), dtype=np.long)
        y_target = self.PAD + np.zeros((batch_size, trg_max_len), 
                                       dtype=np.long)
        
        for i, d in enumerate(batch_data):
            yy = d["trg"]
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


class MatchDataset(Dataset):
    """
    """
    def __init__(self,
                 data_dir,
                 batch_size,
                 symbol2id,
                 rank=0,
                 world_size=1):
        """
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.PAD = symbol2id["_pad_"]
        self.BOS = symbol2id["_bos_"]
        self.EOS = symbol2id["_eos_"]
        self.UNK = symbol2id["_unk_"]
        self.SEP = symbol2id["_sep_"]
        self.CLS = symbol2id["_cls_"]
        self.MASK = symbol2id["_mask_"]
        self.rank = rank
        self.world_size = world_size        
        

    def vectorize(self, batch_data):
        """
        """
        batch_size = len(batch_data)
        src_max_len = max(max(len(xx) for xx in x["src_list"]) for x in batch_data)
        y = self.PAD + np.zeros((2 * batch_size, src_max_len), dtype=np.long)
        y_target = np.zeros((2 * batch_size), dtype=np.long)
        for i, d in enumerate(batch_data):
            y1,y2 = random.sample(d["src_list"], 2)
            y[2*i, :len(y1)] = y1
            y[2*i+1, :len(y2)] = y2
            y_target[2*i] = 2*i+1
            y_target[2*i+1] = 2*i
        y = torch.tensor(y, dtype=torch.long)
        y_target = torch.tensor(y_target, dtype=torch.long)

        return [y], [y_target]


def build_enc_dec_dataset(data_dir, batch_size, symbo2id, train_config, rank, world_size):
    """
    """
    return S2SDataset(
                real_path(data_dir), 
                batch_size, 
                symbo2id,
                rank, 
                world_size)


def build_lm_dataset(data_dir, batch_size, symbol2id, train_config, rank, world_size):
    """
    """
    return LMDataset(
                real_path(data_dir),
                batch_size,
                symbol2id,
                rank, 
                world_size)


def build_classify_dataset(data_dir, batch_size, symbol2id, train_config, rank, world_size):
    """
    """
    return ClassifyDataset(
                real_path(data_dir),
                batch_size,
                symbol2id,
                rank, 
                world_size)

 
def build_bi_lm_dataset(data_dir, batch_size, symbol2id, train_config, rank, world_size):
    """
    """
    return BiLMDataset(
                real_path(data_dir),
                batch_size,
                symbol2id,
                train_config["mask_rate"],
                rank, 
                world_size)


def build_sequence_labeling_dataset(data_dir, batch_size, symbol2id, train_config, rank, world_size):
    """
    """
    return SequenceLabelingDataset(
                real_path(data_dir),
                batch_size,
                symbol2id,
                rank, 
                world_size)


def build_match_dataset(data_dir, batch_size, symbol2id, train_config, rank, world_size):
    """
    """
    return MatchDataset(
                real_path(data_dir),
                batch_size,
                symbol2id,
                rank, 
                world_size)


dataset_builder_dict = {
        "enc_dec": build_enc_dec_dataset,
        "lm": build_lm_dataset,
        "classify": build_classify_dataset,
        "bi_lm": build_bi_lm_dataset,
        "sequence_labeling": build_sequence_labeling_dataset,
        "match": build_match_dataset,
}


def build_dataset(train_config, model_config, dataset="train", rank=0, world_size=1):
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

    if model_config["task"] in dataset_builder_dict:
        dataset_builder_fn = dataset_builder_dict[model_config["task"]]
        return dataset_builder_fn(data_dir, 
                                  batch_size, 
                                  model_config["symbol2id"],
                                  train_config,
                                  rank, 
                                  world_size)
    else:
        raise ValueError("model not correct!")


def build_train_dataset(train_config, model_config, rank=0, world_size=1):
    """
    """
    return build_dataset(train_config, model_config, "train", rank, world_size)


def build_val_dataset(train_config, model_config, rank=0, world_size=1):
    """
    """
    return build_dataset(train_config, model_config, "val", rank, world_size)


def build_test_dataset(train_config, model_config, rank=0, world_size=1):
    """
    """
    return build_dataset(train_config, model_config, "test", rank, world_size)


