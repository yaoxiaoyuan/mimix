# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 22:06:35 2023

@author: Xiaoyuan Yao
"""
import sys
import os
from optparse import OptionParser
from models import build_vit_model
from predictor import load_model_weights
from optimizer import build_optimizer
from scheduler import build_scheduler
from loss import classify_loss
from utils import real_path, load_config, load_model_config
from train import train
from evaluate import eval_acc
import torch
import numpy as np
import random
from torchvision import datasets, transforms

class MNIST():
    """
    """
    def __init__(self, data, batch_size, device):
        """
        """
        self.data = data
        self.batch_size = batch_size
        self.device = device
        self.idx = list(range(0, len(self.data)))
        random.shuffle(self.idx)
        
    
    def local_shuffle(self):
        """
        """
        random.shuffle(self.idx)


    def __call__(self, steps=0):
        """
        """
        i = steps * self.batch_size
        while i < len(self.data):   
            x = torch.cat([self.data[j][0][0].unsqueeze(0) for j in self.idx[i:i+self.batch_size]])
            x = x.float().unsqueeze(1).to(self.device)
            y = torch.tensor([self.data[j][1] for j in self.idx[i:i+self.batch_size]])
            y = y.long().unsqueeze(1).to(self.device)
            yield [x], [y]
            i += self.batch_size


def main(model_config, train_config):
    """
    """
    model = build_vit_model(model_config)
    if train_config.get("reload_model", None) is not None:
        model = load_model_weights(model, real_path(train_config["reload_model"]))
    
    device = "cpu"
    if train_config["use_cuda"] == True:
        device_id = train_config.get("device_id", "0")
        device = 'cuda:%s' % device_id
    
    model = model.to(device)
    eps = train_config.get("eps", 0)
    model.loss_fn = lambda x,y:classify_loss(x[0], y[0], eps)
    symbol2id = model_config["symbol2id"]
    batch_size = train_config["batch_size"]
    
    train_dataset = MNIST(
                          datasets.MNIST("../test_data/mnist-data", train=True, download=True, transform=transforms.ToTensor()),
                          #datasets.FashionMNIST("../test_data/mnist-data", train=True, download=True, transform=transforms.ToTensor()),
                          train_config["batch_size"], 
                          device)
    val_dataset = None
    test_dataset = MNIST(
                         datasets.MNIST("../test_data/mnist-data", train=False, download=True, transform=transforms.ToTensor()), 
                         #datasets.FashionMNIST("../test_data/mnist-data", train=True, download=True, transform=transforms.ToTensor()),
                         train_config["batch_size"],
                         device)

    optimizer = build_optimizer(model, train_config["optimizer"], train_config["lr"])
    lr_scheduler = build_scheduler(train_config, optimizer)
    eval_fn_list = [eval_acc]
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
    usage = "usage: exapmle_train_vit.py --model_conf <file> --train_conf <file>"
    
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
    
