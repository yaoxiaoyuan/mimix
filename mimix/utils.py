# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:39:56 2019

@author: Xiaoyuan Yao
"""
import sys
import os
import configparser
import json
import random
import numpy as np
from abc import ABC, abstractmethod

home_dir = os.path.abspath(os.getcwd())


SYMBOLS = {"PAD_TOK" : "_pad_",
           "BOS_TOK" : "_bos_",
           "EOS_TOK" : "_eos_",
           "UNK_TOK" : "_unk_",
           "SEP_TOK" : "_sep_",
           "CLS_TOK" : "_cls_",
           "MASK_TOK" : "_mask_"}
           
SYMBOL2ID = {"_pad_":0,
             "_bos_":1,
             "_eos_":2,
             "_unk_":3,
             "_sep_":4,
             "_cls_":5,
             "_mask_":6}


def real_path(path, base_dir=None):
    """
    get real path
    """
    if path is None:
        return None
    if os.path.isabs(path) == True:
        return path
    if base_dir is None:
        base_dir = home_dir
    return os.path.join(base_dir, path)


def load_config(config_file):
    """
    load config
    """
    config = configparser.RawConfigParser()
    config.optionxform = str 

    config_file = real_path(config_file)
    if not os.path.exists(config_file):
        print("config file %s not exist!" % config_file)
        sys.exit(0)
        
    config.read(config_file, encoding='utf-8')
    
    loaded_config = {}
    
    for dtype in config.sections():
        if dtype not in ["str", "int", "float", "bool"]:
            continue
        for k,v in config.items(dtype):
            if dtype == "str":
                loaded_config[k] = str(v)
            elif dtype == "int":
                loaded_config[k] = int(v)
            elif dtype == "float":
                loaded_config[k] = float(v)                 
            elif dtype == "bool":
                if v.lower() == "false":
                    loaded_config[k] = False
                elif v.lower() == "true":
                    loaded_config[k] = True
    return loaded_config


def load_model_config(config_file):
    """
    load config
    """
    loaded_config = load_config(config_file)
    
    loaded_config["symbols"] = SYMBOLS
    loaded_config["symbol2id"] = SYMBOL2ID
    
    for symbol in SYMBOLS:
        if symbol + "2tok" in loaded_config:
            loaded_config["symbols"][symbol] = loaded_config[symbol + "2tok"]
    
    for symbol in SYMBOL2ID:
        if symbol + "2id" in loaded_config:
            loaded_config["symbol2id"][symbol] = loaded_config[symbol + "2id"]   

    return loaded_config


def load_vocab(vocab_path):
    """
    """
    vocab = {}
    for i,line in enumerate(open(real_path(vocab_path), "rb")):
        line = line.decode("utf-8").strip()
        if "\t" in line:
            word, word_id = line.split("\t")
        else:
            word, word_id = line, i
        vocab[word] = int(word_id)
    
    return vocab


def invert_dict(dic):
    """
    """
    return {dic[k]:k for k in dic}


def cut_and_pad_seq(seq, max_len, pad, left=False):
    """
    """
    if left == True:
        return [pad] * (max_len - len(seq)) + seq[:max_len]
    return seq[:max_len] + [pad] * (max_len - len(seq))


def cut_and_pad_seq_list(seq_list, max_len, pad, auto=False, pad_left=False):
    """
    """
    if auto == True:
        max_len = min(max(len(seq) for seq in seq_list), max_len)
        
    x = []
    for seq in seq_list:
        x.append(cut_and_pad_seq(seq, max_len, pad, pad_left))

    return x


def derange(xs):
    for a in range(1, len(xs)):
        b = random.randint(0, a-1)
        xs[a], xs[b] = xs[b], xs[a]
    return xs


def nested_to_device(nested_tensor, device):
    """
    """
    res = nested_tensor
    if isinstance(nested_tensor, list) == True or isinstance(nested_tensor, tuple) == True:
        res = []
        for elem in nested_tensor:
            res.append(nested_to_device(elem, device))
    else:
        res = nested_tensor.to(device)
    return res


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


class SimpleDataset(ABC):
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


if __name__ == "__main__":
    pass
    


