# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:16:38 2019

@author: lyn
"""

import os
from datetime import datetime
import logging
import torch
from utils import shuffle_data, real_path, parse_args, load_config
from dataset import build_train_dataset, build_val_dataset, build_test_dataset
from models import build_model
from scheduler import build_lr_scheduler
from loss import build_loss_fn
from evaluate import build_eval_fn
from optimizer import build_optimizer
from utils import nested_to_cuda

LOG_DIR = "../logger"

if not os.path.exists(real_path(LOG_DIR)):
    os.mkdir(real_path(LOG_DIR))

class Trainer():
    """
    """
    def __init__(self, 
                 train_config,
                 model=None,
                 optimizer=None,
                 lr_scheduler=None,
                 train_dataset=None,
                 val_dataset=None,
                 test_dataset=None,
                 loss_fn=None,
                 eval_fn=None,
                 step_callback_fn_list=[]):
        """
        """
        self.train_config = train_config
        self.model_dir = real_path(train_config["model_dir"])
        self.model_name = train_config["model_name"]

        self.use_cuda = train_config["use_cuda"]
        self.device = torch.device("cpu")
        
        if self.use_cuda == True:
            device_id = train_config.get("device_id", "0")
            self.device = torch.device('cuda:%s' % device_id)
        
        self.num_shards = train_config["num_shards"]
        
        self.logger = self.build_logger()
        
        self.accumulate_steps = train_config.get("accumulate_steps", 1)
        
        self.epoch = 1
        self.steps = 1
        self.total_steps = 1
        
        self.make_model_dir()
        
        self.max_epoch = train_config["max_epoch"] + 1
        
        self.save_steps = train_config.get("save_steps", 100000)  
        self.tmp_save_steps = train_config.get("tmp_save_steps", 10000) 
        self.reload = train_config.get("reload", False)
        self.reload_model = real_path(train_config.get("reload_model", ""))
        self.recover_checkpoint = train_config.get("recover_checkpoint", False)
        self.pre_shuffle = train_config.get("pre_shuffle", False)
        
        self.eval_model = train_config.get("eval_model", False)
        
        self.model_config = load_config(real_path(train_config["model_conf"]))
        
        self.data_dir = train_config.get("data_dir")
        self.train_dir = train_config.get("train_dir")
        self.val_dir = train_config.get("val_dir", None)
        self.test_dir = train_config.get("test_dir", None)        
        
        self.batch_size = train_config["batch_size"]
        self.test_batch_size = train_config["test_batch_size"]
        
        self.grad_clip = train_config.get("grad_clip", None)

        self.build_all(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                loss_fn=loss_fn,
                eval_fn=eval_fn)
        
        self.step_callback_fn_list = step_callback_fn_list
        

    def build_logger(self):
        """
        """
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        filename = datetime.today().strftime('../logger/%Y-%m-%d-%H-%M-%S.log')
        logging.basicConfig(filename=real_path(filename), 
                            level=logging.INFO,
                            format=format_str)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        
        formatter = logging.Formatter(format_str, "%Y-%m-%d %H:%M:%S")
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        logger = logging.getLogger(__name__)
        
        return logger

        
    def build_all(self, 
                  model=None,
                  optimizer=None,
                  lr_scheduler=None,
                  train_dataset=None,
                  val_dataset=None,
                  test_dataset=None,
                  loss_fn=None,
                  eval_fn=None):
        """
        """
        self.model = model
        if model is None:
            self.model = build_model(self.model_config)

        self.model = self.model.to(self.device)
            
        self.optimizer = optimizer
        if optimizer is None:
            self.optimizer = build_optimizer(self.model, 
                                             self.train_config["optimizer"],
                                             self.train_config["lr"])
        
        self.lr_scheduler = lr_scheduler
        if lr_scheduler is None:
            self.lr_scheduler = build_lr_scheduler(self.train_config, 
                                                   self.optimizer)
        
        self.train_dataset = train_dataset
        if train_dataset is None:
            self.train_dataset = build_train_dataset(self.train_config,
                                                     self.model_config)
        
        self.val_dataset = val_dataset
        if val_dataset is None and self.val_dir is not None:
            self.val_dataset = build_val_dataset(self.train_config,
                                                 self.model_config)

        self.test_dataset = test_dataset
        if test_dataset is None and self.test_dir is not None:
            self.test_dataset = build_test_dataset(self.train_config,
                                                   self.model_config)
       
        self.loss_fn = loss_fn
        if loss_fn is None:
            self.loss_fn = build_loss_fn(self.model_config, 
                                         self.train_config)
        
        self.eval_fn = eval_fn
        if self.eval_fn is None:
            self.eval_fn = build_eval_fn(self.model_config)


    def make_model_dir(self):
        """
        """
        if not os.path.exists(self.model_dir):
            try:
                os.mkdir(self.model_dir)
                self.logger.info("Create model dir success!")
            except:
                self.model_dir = "./"
                self.logger.info("Change model dir to current dir.")
        else:
            self.logger.info("Model dir already exists!")
        
    
    def save_model(self, model_name=None):
        """
        """
        if model_name is None:
            model_name = "%d.%d.%d.%s" % (self.epoch, 
                                          self.steps,
                                          self.total_steps,
                                          self.model_name)
        
        model_path = real_path(os.path.join(self.model_dir, model_name))
        
        self.logger.info("Save model to %s" % model_path)
        
        torch.save(self.model.state_dict(), 
                   model_path, 
                   _use_new_zipfile_serialization=False)
        
        train_state_dict = {
                    "optimizer": self.optimizer.state_dict(),
                    "epoch":self.epoch,
                    "steps":self.steps,
                    "total_steps": self.total_steps
                }
            
        torch.save(train_state_dict, 
                   model_path + ".optimizer", 
                   _use_new_zipfile_serialization=False)
            
        self.logger.info("Save model complete")

    
    def shuffle_data(self, fast_shuffle=False):
        """
        """
        self.logger.info("Shuffle train data...")
        shuffle_data(self.data_dir, 
                     self.train_dir,
                     fast_shuffle,
                     self.num_shards)
        self.logger.info("Shuffle train data completed!")
    

    def pre_shuffle_data(self):
        """
        """
        if self.pre_shuffle == True and self.epoch == 1 and self.steps == 1:
            self.shuffle_data()


    def print_model_info(self):
        """
        """
        self.logger.info("%s" % self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info("Total Model Params:%s" % total_params)
        
    
    def reload_model_weights(self):
        """
        """
        self.logger.info("Reload model weights.")
        state_dict = torch.load(self.reload_model,
                                map_location=lambda storage, loc: storage)
        param_dict = {}
        for k,v in self.model.named_parameters():
            if k in state_dict:
                param_dict[k] = state_dict[k]
            else:
                self.logger.warn("weight %s not found in model file" % k)
        self.model.load_state_dict(param_dict, False)
        self.logger.info("Reload model success!")
    

    def reload_optimizer_weights(self):
        """
        """
        self.logger.info("Reload optimizer weights.")
        try:
            train_state_dict = torch.load(
                    self.reload_model + ".optimizer",
                    map_location=lambda storage, loc: storage)
                
            self.optimizer.load_state_dict(train_state_dict["optimizer"])
            
            if self.recover_checkpoint == True:
                self.epoch = train_state_dict["epoch"]
                self.steps = train_state_dict["steps"]
                self.total_steps = train_state_dict["total_steps"]

            self.logger.info("Reload optimizer success!")
        except:
            self.logger.info("Reload optimizer failed, ignore")
            

    def reload_model_and_optimizer(self):
        """
        """
        if self.reload == True:
            self.reload_model_weights()
            self.reload_optimizer_weights()
    

    def get_accmulate_weights(self, accum_xy):
        """
        """
        weight_by_non_pad = False
        if self.model_config["task"] in ["enc_dec", "lm", "bi-lm"]:
            weight_by_non_pad = True
        elif self.model_config["task"] == "sequence_labeling":
            if "crf" not in self.model_config["model"]:
                weight_by_non_pad = True
        
        if weight_by_non_pad == True:
            weights = []
            
            total = 0
            for _inputs,_targets in accum_xy:
                nonpad = torch.sum(_targets[0].ne(self.model.PAD))
                weights.append(nonpad.item())
                total += nonpad.item()
            
            weights = [nonpad / total for nonpad in weights]
        else:
            weights = [1 / len(accum_xy)] * len(accum_xy)

        return weights


    def train(self):
        """
        """
        self.print_model_info()
    
        self.reload_model_and_optimizer()
                    
        self.pre_shuffle_data()
        
        self.logger.info("Train Start!")

        history_loss = []
        accum_xy = []
        
        while self.epoch < self.max_epoch: 
            self.model.train()
            
            for inputs,targets in self.train_dataset(self.steps):

                inputs = nested_to_cuda(inputs, self.device)
                targets = nested_to_cuda(targets, self.device)
                
                accum_xy.append([inputs, targets])
                
                if len(accum_xy) == self.accumulate_steps:
                    
                    weights = self.get_accmulate_weights(accum_xy)
                    
                    for (inputs,targets), weight in zip(accum_xy, weights):
                        outputs = self.model(inputs)
                
                        loss = self.loss_fn(outputs, targets)
                
                        history_loss = history_loss[-999:] + [loss.item()]
                        ma_loss = sum(history_loss) / len(history_loss)
                
                        self.logger.info(
                                "%d epoch %d step total %d steps loss: %.3f" % 
                                (self.epoch, 
                                 self.steps, 
                                 self.total_steps,
                                 ma_loss)
                                )
                        
                        loss = loss * weight
                        loss.backward()
                    
                        self.lr_scheduler.step()

                        self.total_steps += 1
                        self.steps += 1
                
                        if self.total_steps % self.save_steps == 0:
                            self.save_model()
                
                        if self.total_steps % self.tmp_save_steps == 0:
                            self.save_model("tmp." + self.model_name)
                        
                        for callback_fn in self.step_callback_fn_list:
                            callback_fn(trainer=self)
                        
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                self.grad_clip
                                )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    accum_xy = []
            
            self.shuffle_data(fast_shuffle=True)
            
            self.epoch += 1
            self.steps = 0
            self.save_model()
            
            if self.eval_model == True and self.eval_fn is not None:
                if self.val_dir is not None:
                    self.logger.info("Eval val now...")
                    self.eval_fn(trainer=self)
                if self.test_dir is not None:
                    self.logger.info("Eval test now...")
                    self.eval_fn(trainer=self)
                            
        self.logger.info("Train Completed!")


def run_train():
    usage = "usage: train.py --conf <file>"
    options = parse_args(usage)
    config_file = options.config
    train_config = load_config(real_path(config_file))
    
    trainer = Trainer(train_config)
    trainer.train()


if __name__ == "__main__":
    run_train()