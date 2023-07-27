# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:39:56 2019

@author: Xiaoyuan Yao
"""
import sys
import os
import configparser
import random

home_dir = os.path.dirname(os.path.abspath(__file__))


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


if __name__ == "__main__":
    pass
    


