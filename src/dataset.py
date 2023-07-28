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
from utils import real_path


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


def nested_to_device(nested_tensor, device):
    """
    """
    res = nested_tensor
    if isinstance(nested_tensor, list) == True or isinstance(nested_tensor, tuple) == True:
        res = []
        for elem in nested_tensor:
            res.append(nested_to_device(elem, device))
    elif isinstance(nested_tensor, torch.Tensor) == True:
        res = nested_tensor.to(device)
    return res


class Dataset(ABC):
    """
    """
    def __init__(self, device="cpu", rank=0, world_size=1):
        """
        """
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.sort_key_fn = None
    

    @abstractmethod
    def vectorize(self, batch_data):
        """
        """
        pass

    
    def local_shuffle(self):
        """
        """
        for f in os.listdir(self.data_dir):
            lines = [line for line in open(os.path.join(self.data_dir, f), "r", encoding="utf-8")]
            random.shuffle(lines)
            if self.sort_key_fn is not None:
                lines = [[line, self.sort_key_fn(json.loads(line))] for line in lines]
                lines.sort(key=lambda x:x[1])
                lines = [x[0] for x in lines]
            fo = open(os.path.join(self.data_dir, f), "w", encoding="utf-8")
            for line in lines:
                fo.write(line)
            fo.close()
    
    
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
                    yield nested_to_device(self.vectorize(batch_data), self.device)
                
        while len(data) > 0:
            batch_data = data[:self.batch_size]
            yield nested_to_device(self.vectorize(batch_data), self.device)           
            data = data[self.batch_size:]


class S2SDataset(Dataset):
    """
    """
    def __init__(self, 
                 data_dir,
                 batch_size,
                 symbol2id,
                 device="cpu",
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
        self.device = device
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
                 device="cpu",
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
        self.device = device
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
                 device="cpu",
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
        self.device = device
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
                 device="cpu",
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
        self.device = device
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
            xx,yy = d["src"],d["seq_label"]
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
                 device="cpu",
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
        self.device = device
        self.mask_rate = mask_rate
        self.rank = rank
        self.world_size = world_size        
        
        
    def vectorize(self, batch_data):
        """
        """
        batch_size = len(batch_data)
        trg_max_len = max(len(x["src"]) for x in batch_data)
        y = self.PAD + np.zeros((batch_size, trg_max_len), dtype=np.long)
        y_target = self.PAD + np.zeros((batch_size, trg_max_len), 
                                       dtype=np.long)
        
        for i, d in enumerate(batch_data):
            yy = d["src"]
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
                 device="cpu",
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
        self.device = device
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
