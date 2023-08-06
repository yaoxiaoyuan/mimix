# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 22:06:35 2023

@author: Xiaoyuan Yao
"""
import sys
import os
from argparse import ArgumentParser
from models import build_lm_model
from predictor import load_model_weights
from optimizer import build_optimizer
from scheduler import build_scheduler
from loss import seq_cross_entropy
from utils import real_path, load_config, load_model_config, SimpleDataset
from train import train


class SequenceLabelingDataset(SimpleDataset):
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
        src_max_len = max(len(x["src"]) for x in batch_data)
        x = self.PAD + np.zeros((batch_size, src_max_len), dtype=np.int64)
        y = self.PAD + np.zeros((batch_size, src_max_len), dtype=np.int64)
        
        for i, d in enumerate(batch_data):
            xx,yy = d["src"],d["seq_label"]
            x[i, :len(xx)] = xx
            y[i, :len(yy)] = yy
        
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        
        return [x, y], [y]


def main(model_config, train_config):
    """
    """
    model = build_lm_model(model_config)
    if train_config.get("reload_model", None) is not None:
        model = load_model_weights(model, real_path(train_config["reload_model"]))
    
    device = "cpu"
    if train_config["use_cuda"] == True:
        device_id = train_config.get("device_id", "0")
        device = 'cuda:%s' % device_id
    
    model = model.to(device)
    eps = train_config.get("eps", 0)
    if model_cofig["use_crf"] == True:
        model.loss_fn = lambda x,y:seq_cross_entropy(x[0], y[0], eps, model.PAD)
    else:
        model.loss_fn = lambda x,y:seq_cross_entropy(x[0], y[0], eps, model.PAD)
    symbol2id = model_config["symbol2id"]
    train_dir = os.path.join(train_config["tmp_dir"], "train")
    batch_size = train_config["batch_size"]
    train_dataset = LMDataset(train_dir, batch_size, symbol2id, device)
    val_dataset = None
    if train_config.get("val_dir", None) is not None:
        val_dir = train_config["val_dir"]
        test_batch_size = train_config["test_batch_size"]
        val_dataset = SequenceLabelingDataset(val_dir, test_batch_size, symbol2id, device)
    test_dataset = None
    if train_config.get("test_dir", None) is not None:
        test_dir = train_config["test_dir"]
        test_batch_size = train_config["test_batch_size"]
        test_dataset = SequenceLabelingDataset(test_dir, test_batch_size, symbol2id, device)  
        
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
    