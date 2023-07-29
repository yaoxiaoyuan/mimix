# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 22:06:35 2023

@author: Xiaoyuan Yao
"""
import sys
import os
from optparse import OptionParser
from models import build_enc_dec_model
from dataset import S2SDataset
from predictor import load_model_weights
from optimizer import build_optimizer
from scheduler import build_scheduler
from loss import seq_cross_entropy
from utils import real_path, load_config, load_model_config
from train import train

def main(model_config, train_config):
    """
    """
    model = build_enc_dec_model(model_config)
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
    train_dataset = S2SDataset(train_dir, batch_size, symbol2id, device)
    val_dataset = None
    if train_config.get("val_dir", None) is not None:
        val_dir = train_config["val_dir"]
        test_batch_size = train_config["test_batch_size"]
        val_dataset = S2SDataset(val_dir, test_batch_size, symbol2id, device)
    test_dataset = None
    if train_config.get("test_dir", None) is not None:
        test_dir = train_config["test_dir"]
        test_batch_size = train_config["test_batch_size"]
        test_dataset = S2SDataset(test_dir, test_batch_size, symbol2id, device)  
        
    optimizer = build_optimizer(model, train_config["optimizer"], train_config["lr"])
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
    usage = "usage: exapmle_train_seq2seq.py --model_conf <file> --train_conf <file>"
    
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

    main(model_config, train_config)


if __name__ == "__main__":
    run_train()
    