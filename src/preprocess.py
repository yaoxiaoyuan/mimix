# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 10:39:14 2019

@author: Xiaoyuan Yao
"""
import sys
from optparse import OptionParser
import json
import os
import random
from tokenization import build_tokenizer
from utils import real_path, load_vocab, load_config, load_model_config

class TextProcessor():
    """
    """
    def __init__(self, **kwargs):
        """
        """
        self.src_max_len = kwargs.get("src_max_len", None)
        self.trg_max_len = kwargs.get("trg_max_len", None)
        self.PAD = kwargs["symbol2id"]["_pad_"]
        self.BOS = kwargs["symbol2id"]["_bos_"]
        self.EOS = kwargs["symbol2id"]["_eos_"]
        self.UNK = kwargs["symbol2id"]["_unk_"]
        self.SEP = kwargs["symbol2id"]["_sep_"]
        self.CLS = kwargs["symbol2id"]["_cls_"]
        self.MASK = kwargs["symbol2id"]["_mask_"]
        self.src_tokenizer = None
        self.src_tokenizer = build_tokenizer(
                tokenizer=kwargs["src_tokenizer"],
                vocab_file=real_path(kwargs["src_vocab"])) 
        self.trg_tokenizer = None
        self.trg_tokenizer = build_tokenizer(
                tokenizer=kwargs["trg_tokenizer"],
                vocab_file=real_path(kwargs["trg_vocab"])) 
        self.label2id = None
        if kwargs.get("label2id", None) is not None:
            load_vocab(real_path(kwargs["label2id"]))
        self.task = kwargs["task"]
        
        
    def parse(self, line):
        """
        """        
        try:
            data = json.loads(line)
            return data
        except:
            return None
        

    def preprocess(self, data):
        """
        """        
        src = data.get("src", None)
        trg = data.get("trg", None)
        label = data.get("label", None)
        seq_label = data.get("seq_label", None)
        
        data = {}
        if src:
            src = self.src_tokenizer.tokenize_to_ids(src)
            src = src[:self.src_max_len]
            if self.task == "classify" or self.task == "match":
                src = [self.CLS] + src[:self.src_max_len - 1]  
            data["src"] = src
        
        if trg:
            trg = self.trg_tokenizer.tokenize_to_ids(trg)
            trg = trg[:self.trg_max_len - 1]
            trg = [self.BOS] + trg + [self.EOS]
            data["trg"] = trg

        if label:
            if self.label2id is not None:
                label = self.label2id[label]
            else:
                label = int(label)
            data["label"] = label

        if seq_label:
            if self.label2id is not None:
                seq_label = [self.label2id[s] for s in seq_label]
            else:
                seq_label = [int(s) for s in seq_label]
            data["seq_label"] = seq_label
            
        return data


    def __call__(self, line):
        """
        """
        parsed = self.parse(line)
        processed = None
        if parsed is not None:
            processed = self.preprocess(parsed)

        return processed


def preprocess(data_dir, 
               dest_dir, 
               num_shards=1, 
               data_preprocessor=None,
               sort_key_fn=None):
    """
    Shuffle data
    """
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


def run_preprocess():
    """
    """
    usage = "usage: preprocess.py --model_conf <file>"
    
    parser = OptionParser(usage)

    parser.add_option("--train_conf", action="store", type="string",
                      dest="train_config")

    parser.add_option("--model_conf", action="store", type="string", 
                      dest="model_config")

    (options, args) = parser.parse_args(sys.argv)

    if not options.train_config or not options.model_config:
        print(usage)
        sys.exit(0)
    
    model_config = load_model_config(real_path(options.model_config))
    train_config = load_config(real_path(options.train_config))
    
    processor = TextProcessor(**model_config)
    
    preprocess(train_config["train_dir"], 
                 os.path.join(real_path(train_config["tmp_dir"]), "train"), 
                 num_shards=train_config.get("num_shards", 1), 
                 data_preprocessor=processor,
                 sort_key_fn=None)
    if train_config.get("val_dir", None) is not None:
        preprocess(train_config["val_dir"], 
                     os.path.join(real_path(train_config["tmp_dir"]), "val"), 
                     num_shards=1, 
                     data_preprocessor=processor,
                     sort_key_fn=None)
    if train_config.get("test_dir", None) is not None:
        preprocess(train_config["test_dir"], 
                     os.path.join(real_path(train_config["tmp_dir"]), "test"), 
                     num_shards=1, 
                     data_preprocessor=processor,
                     sort_key_fn=None)    
        
if __name__ == "__main__":
    run_preprocess()





