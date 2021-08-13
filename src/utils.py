# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:39:56 2019

@author: lyn
"""
import sys
import os
import configparser
from optparse import OptionParser
import random
import torch
import json
from constants import symbols, symbol2id
from models import build_model

home_dir = os.path.split(os.path.realpath(sys.argv[0]))[0]

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


def parse_args(usage):
    """
    parse arguments
    """
    parser = OptionParser(usage)
    parser.add_option("--conf", action="store", type="string", 
                      dest="config")
    
    (options, args) = parser.parse_args(sys.argv)

    if not options.config:
        print(usage)
        sys.exit(0)
    return options


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
    
    train_config = {}
    
    for dtype in config.sections():
        for k,v in config.items(dtype):
            if dtype == "bool":
                train_config[k] = eval(v)
            else:
                train_config[k] = eval(dtype + '("' + v + '")')
    
    train_config["symbols"] = symbols
    train_config["symbol2id"] = symbol2id
    
    for symbol in symbols:
        if symbol in train_config:
            train_config["symbols"][symbol] = train_config[symbol]
    
    for symbol in symbols:
        if symbol in train_config:
            train_config["symbol2id"][symbol] = train_config[symbol]   
    
    return train_config


def shuffle_data(data_dir, dest_dir, fast_shuffle=False, num_shards=20):
    """
    Shuffle data
    """
    if fast_shuffle == False:
        data_files = [f for f in os.listdir(data_dir)]

        fo_list = []
        for f in range(num_shards):
            fo_list.append(open(os.path.join(dest_dir, str(f)), "wb"))
    
        for fi in data_files:
            for line in open(os.path.join(data_dir, fi), "rb"):
                fo = random.choice(fo_list)
                fo.write(line)

        for fo in fo_list:
            fo.close()

    for f in range(num_shards):
        lines = [line for line in open(os.path.join(dest_dir, str(f)), "rb")]
        random.shuffle(lines)
        fo = open(os.path.join(dest_dir, str(f)), "wb")
        for line in lines:
            fo.write(line)
        fo.close()


def pretty_print(res):
    """
    Assume res = [{k1:v1,k2:v2,...,kn:[[s1, score1], [s2, score2]]}, {}, ... ]
    """
    for dic in res:
        info = [[k,dic[k]] for k in dic if not isinstance(dic[k], list)]
        info = " ".join("%s:%s" % (k,v) for k,v in info)
        if len(info) > 0:
            print(info)
        print("--------------------")
        for k in dic:
            if isinstance(dic[k], list):
                for a in dic[k]:
                    info = " ".join([str(x) for x in a])
                    print(info)


def load_vocab(vocab_path):
    """
    """
    vocab = {}
    for line in open(vocab_path, "rb"):
        line = line.decode("utf-8").strip()
        word, word_id = line.split("\t")
        vocab[word] = int(word_id)
    
    return vocab


def invert_dict(dic):
    """
    """
    return {dic[k]:k for k in dic}


def convert_conf_file(conf_file, json_file):
    """
    """
    config = load_config(conf_file)
    fo = open(json_file, "w")
    config_str = json.dumps(config, indent=2)
    fo.write(config_str)
    fo.close()


def load_model(config):
    """
    """
    model = build_model(config)
    
    model_path = real_path(config["load_model"])
    state_dict = torch.load(model_path,
                            map_location=lambda storage, loc: storage)

    param_dict = {}
    for k,v in model.named_parameters():
        if k in state_dict:
            param_dict[k] = state_dict[k] 

    model.load_state_dict(param_dict, False)

    use_cuda = config.get("use_cuda", False)
    if use_cuda == True:
        device = torch.device('cuda:%s' % config.get("device_id", "0"))
        model = model.to(device)
    
    return model


def nested_to_cuda(nested_tensor, device):
    """
    """
    res = nested_tensor
    if isinstance(nested_tensor, list) == True:
        res = []
        for elem in nested_tensor:
            res.append(nested_to_cuda(elem, device))
    elif isinstance(nested_tensor, torch.Tensor) == True:
        res = nested_tensor.to(device)
    
    return res


def cut_and_pad_seq(seq, max_len, pad):
    """
    """
    return seq[:max_len] + [pad] * (max_len - len(seq))


def cut_and_pad_seq_list(seq_list, max_len, pad, auto=False):
    """
    """
    if auto == True:
        max_len = min(max(len(seq) for seq in seq_list), max_len)
        
    x = []
    for seq in seq_list:
        x.append(cut_and_pad_seq(seq, max_len, pad))

    return x


def derange(xs):
    for a in range(1, len(xs)):
        b = random.randint(0, a-1)
        xs[a], xs[b] = xs[b], xs[a]
    return xs


if __name__ == "__main__":
    pass
    


