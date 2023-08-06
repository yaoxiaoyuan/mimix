# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 22:06:35 2023

@author: Xiaoyuan Yao
"""
import sys
import os
from argparse import ArgumentParser
import numpy as np
import torch
from mimix.models import build_encoder_model
from mimix.predictor import load_model_weights
from mimix.optimizer import build_optimizer
from mimix.scheduler import build_scheduler
from mimix.loss import seq_cross_entropy
from mimix.utils import real_path, load_config, load_model_config, SimpleDataset
from mimix.train import train


class MatchDataset(SimpleDataset):
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
        self.sort_key_fn = None
                

    def vectorize(self, batch_data):
        """
        """
        batch_size = len(batch_data)
        src_max_len = max(max(len(xx) for xx in x["src_list"]) for x in batch_data)
        y = self.PAD + np.zeros((2 * batch_size, src_max_len), dtype=np.int64)
        y_target = np.zeros((2 * batch_size), dtype=np.int64)
        for i, d in enumerate(batch_data):
            y1,y2 = random.sample(d["src_list"], 2)
            y[2*i, :len(y1)] = y1
            y[2*i+1, :len(y2)] = y2
            y_target[2*i] = 2*i+1
            y_target[2*i+1] = 2*i
        y = torch.tensor(y, dtype=torch.long)
        y_target = torch.tensor(y_target, dtype=torch.long)

        return [y], [y_target]


def main(model_config, train_config):
    """
    """
    model = build_encoder_model(model_config)
    if train_config.get("reload_model", None) is not None:
        model = load_model_weights(model, real_path(train_config["reload_model"]))
    
    device = "cpu"
    if train_config["use_cuda"] == True:
        device_id = train_config.get("device_id", "0")
        device = 'cuda:%s' % device_id
    
    model = model.to(device)
    eps = train_config.get("eps", 0)
    model.loss_fn = lambda x,y:seq_cross_entropy(x[0], y[0], eps, model.PAD)
    symbol2id = model_config["symbol2id"]
    train_dir = os.path.join(train_config["tmp_dir"], "train")
    batch_size = train_config["batch_size"]
    train_dataset = MatchDataset(train_dir, batch_size, symbol2id, device)
    val_dataset = None
    if train_config.get("val_dir", None) is not None:
        val_dir = train_config["val_dir"]
        test_batch_size = train_config["test_batch_size"]
        val_dataset = MatchDataset(val_dir, test_batch_size, symbol2id, device)
    test_dataset = None
    if train_config.get("test_dir", None) is not None:
        test_dir = train_config["test_dir"]
        test_batch_size = train_config["test_batch_size"]
        test_dataset = MatchDataset(test_dir, test_batch_size, symbol2id, device)  
        
    optimizer = build_optimizer(model, train_config)
    lr_scheduler = build_scheduler(train_config, optimizer)
    eval_fn_list = []
    train(model, 
          optimizer,
          train_config,
          train_dataset, 
          val_dataset, 
          test_dataset, 
          eval_fn_list,
          lr_scheduler)


def run_train():
    """
    """
    parser = ArgumentParser()

    parser.add_argument("--model_conf", type=str)
    parser.add_argument("--train_conf", type=str)
    
    args = parser.parse_args(sys.argv[1:])

    model_config = load_model_config(real_path(args.model_conf))
    train_config = load_config(real_path(args.train_conf))

    main(model_config, train_config)


if __name__ == "__main__":
    run_train()
    