# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 22:06:35 2023

@author: Xiaoyuan Yao
"""
import sys
import os
from argparse import ArgumentParser
from mimix.models import build_vit_model
from mimix.predictor import load_model_weights
from mimix.optimizer import build_optimizer
from mimix.scheduler import build_scheduler
from mimix.loss import classify_loss
from mimix.utils import real_path, load_config, load_model_config, nested_to_device
from mimix.train import train
from mimix.evaluate import eval_acc
import torch
import numpy as np
import random
import gzip
import tarfile
from PIL import Image
import io
from torchvision import transforms
import torch
from torch.utils.data import Dataset,DataLoader,get_worker_info

class CFNDataset(Dataset):
    """
    """
    def __init__(self, 
                 archive, 
                 paths, 
                 img_w, 
                 img_h, 
                 use_randaug=False): 
        """
        """
      
        worker = get_worker_info()
        worker = worker.id if worker else None
        self.tar_obj = {worker: tarfile.open(archive)}
        self.archive = archive

        self.paths = paths
        self.img_w = img_w
        self.img_h = img_h
        if use_randaug == True:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((img_w, img_h)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5, 0.5),
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_w, img_h)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5, 0.5),
                ])


    def __len__(self):
        """
        """
        return len(self.paths)


    def __getitem__(self, index):
        """
        """
        worker = get_worker_info()
        worker = worker.id if worker else None

        if worker not in self.tar_obj:
            self.tar_obj[worker] = tarfile.open(self.archive)

        image = Image.open(io.BytesIO(self.tar_obj[worker].extractfile(self.paths[index][0]).read()))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        x = self.transform(image).unsqueeze(0)
        y = torch.tensor([self.paths[index][1]], dtype=torch.long)
        image.close()
        return x,y


    def __del__(self):
        """
        """
        for o in self.tar_obj.values():
            o.close()


    def __getstate__(self):
        """
        """
        state = dict(self.__dict__)
        state['tar_obj'] = {}
        return state

 
def read_and_split(archive):
    """
    """
    zfile = tarfile.open(archive, 'r')

    test_labels = {}
    for line in zfile.extractfile("release_data/test_truth_list.txt").read().decode("utf-8").split("\n"):
        if len(line.strip()) == 0:
            continue
        f,label = line.strip().split()
        test_labels[f] = int(label)

    train_data, val_data, test_data = [], [], []
    file_paths = zfile.getnames()
    for f in file_paths:
        if not f.endswith("jpg"):
            continue
        label = f.split("/")[2]
        if "train" in f:
            train_data.append([f, int(label)])
        if "val" in f:
            val_data.append([f, int(label)])
        if "test" in f:
            test_data.append([f, int(test_labels[f.split("/")[-1]])])

    random.shuffle(train_data)

    return train_data, val_data, test_data


def collate_batch(batch_list, use_mixup=False, alpha=0.2, n_class=None):
    x = torch.cat([item[0] for item in batch_list])
    y = torch.cat([item[1] for item in batch_list])
    
    if use_mixup == True:
        lam = np.random.beta(alpha, alpha)

        batch_size = x.shape[0]
        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        mixed_y = lam * torch.eye(n_class,device=y.device)[y_a] + (1 - lam) * torch.eye(n_class,device=y.device)[y_b]

        x = mixed_x
        y = mixed_y

    return [x], [y]


class Scheduler():
    """
    """
    def __init__(self, optimizer, train_config):
        """
        """
        self.lr = train_config["lr"]
        self.optimizer = optimizer
        self.steps = 0


    def step(self):
        """
        """
        self.steps += 1
        if self.steps % 7500 == 0:
            self.lr = max(5e-7, self.lr/10)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr


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

    archive = "data/dataset_release.tar"

    train_paths, val_paths, test_paths = read_and_split(archive)
    print(len(train_paths), len(val_paths), len(test_paths))

    train_dataset = CFNDataset(archive,
                               train_paths,
                               model_config["img_w"],
                               model_config["img_h"],
                               True)

    val_dataset = CFNDataset(archive,
                             val_paths,
                             model_config["img_w"],
                             model_config["img_h"])

    test_dataset = CFNDataset(archive,
                              test_paths,
                              model_config["img_w"],
                              model_config["img_h"])
   
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  num_workers=6, 
                                  collate_fn=lambda x:collate_batch(x, use_mixup=True, alpha=0.2, n_class=model_config["n_class"]))
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=batch_size, 
                                num_workers=6, 
                                collate_fn=lambda x:collate_batch(x, use_mixup=False, alpha=None, n_class=model_config["n_class"]))
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=batch_size, 
                                 num_workers=6, 
                                 collate_fn=lambda x:collate_batch(x, use_mixup=False, alpha=None, n_class=model_config["n_class"]))

    def data_generator(dataloader):
        for inputs,targets in dataloader:
            yield nested_to_device(inputs, device), nested_to_device(targets, device)

    optimizer = build_optimizer(model, train_config)
    lr_scheduler = Scheduler(optimizer, train_config)
    eval_fn_list = [eval_acc]
    train(model, 
          optimizer,
          train_config,
          lambda:data_generator(train_dataloader), 
          lambda:data_generator(val_dataloader), 
          lambda:data_generator(test_dataloader), 
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
    
