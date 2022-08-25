# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:39:56 2019

@author: Xiaoyuan Yao
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

home_dir = os.path.dirname(os.path.abspath(__file__))

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


def parse_train_args(usage):
    """
    parse arguments
    """
    parser = OptionParser(usage)

    parser.add_option("--local_rank", action="store", type="int",
                      dest="local_rank")

    parser.add_option("--train_conf", action="store", type="string",
                      dest="train_config")

    parser.add_option("--model_conf", action="store", type="string", 
                      dest="model_config")

    (options, args) = parser.parse_args(sys.argv)

    if not options.train_config or not options.model_config:
        print(usage)
        sys.exit(0)
    return options


def parse_test_args(usage):
    """
    parse arguments
    """
    parser = OptionParser(usage)

    parser.add_option("--model_conf", action="store", type="string",
                      dest="model_config")

    (options, args) = parser.parse_args(sys.argv)

    if not options.model_config:
        print(usage)
        sys.exit(0)
    return options


def load_config(config_file, add_symbol=False):
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
    
    if add_symbol == True:
        train_config["symbols"] = symbols
        train_config["symbol2id"] = symbol2id
    
        for symbol in symbols:
            if symbol + "2tok" in train_config:
                train_config["symbols"][symbol] = train_config[symbol + "2tok"]
    
        for symbol in symbol2id:
            if symbol + "2id" in train_config:
                train_config["symbol2id"][symbol] = train_config[symbol + "2id"]   

    return train_config


def shuffle_data(data_dir, 
                 dest_dir, 
                 fast_shuffle=False, 
                 num_shards=20, 
                 data_preprocessor=None,
                 sort_key_fn=None):
    """
    Shuffle data
    """
    if fast_shuffle == False:
        data_files = [f for f in os.listdir(data_dir)]

        fo_list = []
        for f in range(num_shards):
            fo_list.append(open(os.path.join(dest_dir, str(f)), "w", encoding="utf-8"))
    
        for fi in data_files:
            for line in open(os.path.join(data_dir, fi), "r", encoding="utf-8"):
                fo = random.choice(fo_list)
                if data_preprocessor is not None:
                    data = data_preprocessor(line)
                    line = json.dumps(data, ensure_ascii=False) + "\n"
                fo.write(line)

        for fo in fo_list:
            fo.close()

    for f in range(num_shards):
        lines = [line for line in open(os.path.join(dest_dir, str(f)), "r", encoding="utf-8")]
        random.shuffle(lines)
        if sort_key_fn is not None:
            lines = [[line, sort_key_fn(json.loads(line))] for line in lines]
            lines.sort(key=lambda x:x[1])
            lines = [x[0] for x in lines]
        fo = open(os.path.join(dest_dir, str(f)), "w", encoding="utf-8")
        for line in lines:
            fo.write(line)
        fo.close()


def preprocess_data(data_dir, 
                    dest_dir, 
                    data_preprocessor
        ):
    """
    """
    data_files = [f for f in os.listdir(data_dir)]
    for fi in data_files:
        fo = open(os.path.join(dest_dir, str(fi)), "w", encoding="utf-8")
        for line in open(os.path.join(data_dir, fi), "r", encoding="utf-8"):
            if data_preprocessor is not None:
                data = data_preprocessor(line)
                line = json.dumps(data, ensure_ascii=False) + "\n"
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
        else:
            print("warn: weight %s not found in model file" % k)

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
    


