# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 22:06:35 2023

@author: Xiaoyuan Yao
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from argparse import ArgumentParser
from mimix.models import build_model, load_model_weights
from mimix.optimizer import build_optimizer
from mimix.scheduler import build_scheduler
from mimix.loss import seq_cross_entropy
from mimix.utils import real_path, load_config, load_model_config
from mimix.train import train
from mimix.evaluate import eval_acc
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
            y = torch.tensor([[1, 18, 8 + self.data[j][1]] for j in self.idx[i:i+self.batch_size]])
            y = y.long().to(self.device)
            y_target = torch.tensor([[18, 8 + self.data[j][1], 2] for j in self.idx[i:i+self.batch_size]])
            y_target = y_target.long().to(self.device)
            yield [x, y], [y_target]
            i += self.batch_size


def main(model_config, train_config):
    """
    """
    model = build_model(model_config)
    if train_config.get("reload_model", None) is not None:
        model = load_model_weights(model, real_path(train_config["reload_model"]))
    
    device = "cpu"
    if train_config["use_cuda"] == True:
        device_id = train_config.get("device_id", "0")
        device = 'cuda:%s' % device_id
    
    model = model.to(device)
    eps = train_config.get("eps", 0)
    model.loss_fn = lambda x,y:seq_cross_entropy(x["logits"], y[0], eps, model.PAD)
    symbol2id = model_config["symbol2id"]
    batch_size = train_config["batch_size"]
    
    transform = transforms.Compose([
                transforms.Resize((model_config["img_w"], model_config["img_h"])),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ])
    
    train_dataset = MNIST(
                          datasets.MNIST("test_data/mnist-data", train=True, download=True, transform=transform),
                          #datasets.FashionMNIST("test_data/mnist-data", train=True, download=True, transform=transforms.ToTensor()),
                          train_config["batch_size"], 
                          device)
    val_dataset = None
    test_dataset = MNIST(
                         datasets.MNIST("test_data/mnist-data", train=False, download=True, transform=transform), 
                         #datasets.FashionMNIST("test_data/mnist-data", train=True, download=True, transform=transforms.ToTensor()),
                         train_config["batch_size"],
                         device)

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
    