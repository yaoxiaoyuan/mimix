# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 22:06:35 2023

@author: 1
"""
import sys
import os
from optparse import OptionParser
from models import build_lm_model
from dataset import LMDataset
from predictor import load_model_weights
from optimizer import build_optimizer
from scheduler import build_scheduler
from util import real_path, load_config
from train import train

def main(model_config, train_config):
    """
    """
    model = build_lm_model(**model_config)
    model = load_model_weights(model, real_path(train_config["reload_model"]))
    
    device = "cpu"
    if train_config["use_cuda"] == True:
        device_id = train_config.get("device_id", "0")
        device = 'cuda:%s' % device_id
    
    model = model.to(device)
    
    symbol2id = model_config["symbol2id"]
    train_dir = os.path.join(train_config["tmp_dir"], "train")
    batch_size = train_config["batch_size"]
    train_dataset = LMDataset(train_dir, batch_size, symbol2id, device)
    val_dataset = None
    if train_config.get("val_dir", "") is not None:
        val_dir = train_config["val_dir"]
        test_batch_size = train_config["test_batch_size"]
        val_dataset = LMDataset(val_dir, test_batch_size, symbol2id, device)
    test_dataset = None
    if train_config.get("test_dir", "") is not None:
        test_dir = train_config["test_dir"]
        test_batch_size = train_config["test_batch_size"]
        test_dataset = LMDataset(test_dir, test_batch_size, symbol2id, device)  
        
    optimizer = build_optimizer(model, train_config["optimizer"], train_config["lr"])
    lr_scheduler = build_scheduler(train_config, optimizer)
    
    train(model, 
          optimizer,
          train_dataset, 
          val_dataset=val_dataset, 
          test_dataset=test_dataset, 
          eval_fn_list=None,
          lr_scheduler=lr_scheduler,
          **train_config)


def run_train():
    """
    """
    usage = "usage: interact.py --model_conf <file>"
    
    parser = OptionParser(usage)

    parser.add_option("--train_conf", action="store", type="string",
                      dest="train_config")

    parser.add_option("--model_conf", action="store", type="string", 
                      dest="model_config")

    (options, args) = parser.parse_args(sys.argv)

    if not options.train_config or not options.model_config:
        print(usage)
        sys.exit(0)
    
    model_config = load_config(real_path(options.model_config))
    train_config = load_config(real_path(options.train_config))

    main(model_config, train_config)
