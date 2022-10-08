# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:16:38 2019

@author: Xiaoyuan Yao
"""
import os
from datetime import datetime
import logging
import torch
import torch.distributed as dist
from utils import shuffle_data, real_path, parse_train_args, load_config
from process_data import build_data_processor
from dataset import build_train_dataset, build_val_dataset, build_test_dataset
from models import build_model
from scheduler import build_lr_scheduler
from loss import build_loss_fn
from evaluator import build_eval_fn
from optimizer import build_optimizer
from utils import nested_to_cuda, preprocess_data

LOG_DIR = "../logger"

if not os.path.exists(real_path(LOG_DIR)):
    os.mkdir(real_path(LOG_DIR))

local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
rank = int(os.environ['RANK'])
       
format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
filename = datetime.today().strftime('../logger/%Y-%m-%d-%H-%M-%S.log')
logging.basicConfig(filename=real_path(filename),
                    level=logging.INFO,
                    format=format_str)

dist.init_process_group('nccl',world_size=world_size, rank=rank)

class Trainer():
    """
    """
    def __init__(self, 
                 train_config,
                 model_config,
                 model=None,
                 optimizer=None,
                 lr_scheduler=None,
                 train_dataset=None,
                 val_dataset=None,
                 test_dataset=None,
                 loss_fn=None,
                 eval_fn=None,
                 custom_parse_fn=None,
                 step_callback_fn_list=[]):
        """
        """
        self.train_config = train_config
        self.model_dir = real_path(train_config["model_dir"])
        self.model_name = train_config["model_name"]
        
        self.device = torch.device('cuda:%s' % local_rank)
        
        self.num_shards = train_config["num_shards"]
        
        self.logger = self.build_logger()
        
        self.accumulate_steps = train_config.get("accumulate_steps", 1)
        
        self.epoch = 1
        self.steps = 1
        self.total_steps = 1
        
        self.max_epoch = train_config["max_epoch"] + 1
        
        self.save_steps = train_config.get("save_steps", 100000)  
        self.tmp_save_steps = train_config.get("tmp_save_steps", 10000) 
        self.reload = train_config.get("reload", False)
        self.reload_model = real_path(train_config.get("reload_model", ""))
        self.recover_checkpoint = train_config.get("recover_checkpoint", False)
        self.need_preprocess= train_config.get("need_preprocess", True)
        self.pre_shuffle = train_config.get("pre_shuffle", True)
        self.sort_data = train_config.get("sort_data", False)
        
        self.eval_model = train_config.get("eval_model", False)
        
        self.model_config = model_config
        
        self.raw_train_dir = real_path(train_config.get("raw_train_dir"))
        self.raw_val_dir = real_path(train_config.get("raw_val_dir"))
        self.raw_test_dir = real_path(train_config.get("raw_test_dir"))
        self.train_dir = os.path.join(real_path(train_config.get("tmp_dir")),
                                      "train")
        self.val_dir = None
        if self.raw_val_dir:
            self.val_dir = os.path.join(real_path(train_config.get("tmp_dir")),
                                      "val")
        self.test_dir = None
        if self.test_dir:
            self.test_dir = os.path.join(real_path(train_config.get("tmp_dir")),
                                      "test")       
        
        self.train_config["train_dir"] = self.train_dir
        self.train_config["val_dir"] = self.val_dir
        self.train_config["test_dir"] = self.test_dir
        
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
        
        self.custom_parse_fn = custom_parse_fn
        self.step_callback_fn_list = step_callback_fn_list
        
        self.make_all_dir()
        

    def build_logger(self):
        """
        """
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        
        formatter = logging.Formatter(format_str, "%Y-%m-%d %H:%M:%S")
        console.setFormatter(formatter)
        logging.getLogger(__name__).addHandler(console)
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
        
        self.train_dataset = train_dataset
        if train_dataset is None:
            self.train_dataset = build_train_dataset(self.train_config,
                                                     self.model_config,
                                                     rank,
                                                     world_size)
        
        self.val_dataset = val_dataset
        if val_dataset is None and self.val_dir is not None:
            self.val_dataset = build_val_dataset(self.train_config,
                                                 self.model_config,
                                                 rank,
                                                 world_size)

        self.test_dataset = test_dataset
        if test_dataset is None and self.test_dir is not None:
            self.test_dataset = build_test_dataset(self.train_config,
                                                   self.model_config,
                                                   rank,
                                                   world_size)
       
        self.model.loss_fn = loss_fn
        if loss_fn is None:
            self.model.loss_fn = build_loss_fn(self.model_config, 
                                         self.train_config)
        
        self.eval_fn = eval_fn
        if self.eval_fn is None:
            self.eval_fn = build_eval_fn(self.model_config)

        if self.reload == True:
            self.reload_model_weights()
        
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        self.optimizer = optimizer 
        
        if optimizer is None:
            self.optimizer = build_optimizer(self.model,
                                             self.train_config["optimizer"],
                                             self.train_config["lr"])

        if self.reload == True:
            self.reload_optimizer_weights()

        self.lr_scheduler = lr_scheduler
        if lr_scheduler is None:
            self.lr_scheduler = build_lr_scheduler(self.train_config,
                                                   self.optimizer)

    def make_all_dir(self):
        """
        """
        for _name, _dir in [["Model Dir", self.model_dir],
                            ["Train Dir", self.train_dir],
                            ["Val Dir", self.val_dir],
                            ["Test Dir", self.test_dir]]:
            if _dir is not None and not os.path.exists(_dir):
                try:
                    os.mkdir(_dir)
                    self.logger.info("Create %s success!" % _name)
                except:
                    self.model_dir = "./"
                    self.logger.info("Change %s to current dir." % _name)
            else:
                self.logger.info("%s already exists!" % _name)
        
    
    def save_model(self, model_name=None):
        """
        """
        if rank == 0:
            if model_name is None:
                model_name = "%d.%d.%d.%s" % (self.epoch, 
                                              self.steps,
                                              self.total_steps,
                                              self.model_name)
        
            model_path = real_path(os.path.join(self.model_dir, model_name))
        
            self.logger.info("Save model to %s" % model_path)
        
            torch.save(self.model.module.state_dict(), 
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

    
    def get_sort_key_fn(self):
        """
        """
        sort_key_fn = None
        if self.model_config["task"] == "enc_dec":
            sort_key_fn = lambda x:len(x["trg"])
        elif self.model_config["task"] == "lm":
            sort_key_fn = lambda x:len(x["trg"])
        elif self.model_config["task"] == "classify":
            sort_key_fn = lambda x:len(x["src"])
        elif self.model_config["task"] == "bi_lm":
            sort_key_fn = lambda x:len(x["trg"])
        elif self.model_config["task"] == "sequence_labeling":
            sort_key_fn = lambda x:len(x["src"])
        
        return sort_key_fn
    
    
    def shuffle_data(self, fast_shuffle=False):
        """
        """
        if rank == 0:
            self.logger.info("Shuffle train data...")
            sort_key_fn = None
            if self.sort_data == True:
                sort_key_fn = self.get_sort_key_fn()
            shuffle_data(self.raw_train_dir, 
                         self.train_dir,
                         fast_shuffle=fast_shuffle,
                         num_shards=self.num_shards,
                        sort_key_fn=sort_key_fn)
            self.logger.info("Shuffle train data completed!")
        dist.barrier()


    def pre_shuffle_data(self):
        """
        """
        if rank == 0:
            if self.pre_shuffle == True and self.epoch == 1 and self.steps == 1:
                self.logger.info("Pre Shuffle train data...")
                data_preprocessor = None
                if self.need_preprocess == True:
                    data_preprocessor = build_data_processor(self.train_config, self.model_config)
                    data_preprocessor.custom_parse_fn = self.custom_parse_fn
                sort_key_fn = None
                if self.sort_data == True:
                    sort_key_fn = self.get_sort_key_fn()
            
                shuffle_data(self.raw_train_dir,
                             self.train_dir, 
                             fast_shuffle=False,
                             num_shards=self.num_shards,
                             data_preprocessor=data_preprocessor,
                             sort_key_fn=sort_key_fn)

            if self.eval_model == True and self.eval_fn is not None:
                if self.val_dir is not None:
                    preprocess_data(self.raw_val_dir, self.val_dir, data_preprocessor)
                if self.test_dir is not None:
                    preprocess_data(self.raw_test_dir, self.test_dir, data_preprocessor)
            
                self.logger.info("Pre Shuffle train data completed!")
        dist.barrier()


    def print_model_info(self):
        """
        """
        if rank == 0:
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
                if state_dict[k].shape == v.shape:
                    param_dict[k] = state_dict[k]
                else:
                    self.logger.warn("weight %s shape not same" % k)
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


    def train(self):
        """
        """
        self.print_model_info()
    
        self.pre_shuffle_data()
       
        if rank == 0:
            self.logger.info("Train Start!")

        history_loss = []
        
        while self.epoch < self.max_epoch: 
            self.model.train()
            
            for inputs,targets in self.train_dataset(self.steps):

                inputs = nested_to_cuda(inputs, self.device)
                targets = nested_to_cuda(targets, self.device)
                 
                outputs = self.model(inputs, targets=targets, compute_loss=True)
        
                loss = outputs[0]
        
                history_loss = history_loss[-999:] + [loss.item()]
                ma_loss = sum(history_loss) / len(history_loss)
       
                if rank == 1:
                    self.logger.info(
                            "%d epoch %d step total %d steps loss: %.3f" % 
                            (self.epoch, 
                            self.steps, 
                            self.total_steps,
                            ma_loss)
                            )
                
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
            
            self.shuffle_data(fast_shuffle=True)
            
            self.epoch += 1
            self.steps = 0
            self.save_model()
            
            if rank == 0:
                if self.eval_model == True and self.eval_fn is not None:
                    if self.val_dir is not None:
                        self.logger.info("Eval val now...")
                        self.eval_fn(trainer=self)
                    if self.test_dir is not None:
                        self.logger.info("Eval test now...")
                        self.eval_fn(trainer=self)
        if rank == 0:    
            self.logger.info("Train Completed!")


def run_train():
    """
    """
    usage = "usage: run_train.py --model_conf <file> --train_conf <file>"

    options = parse_train_args(usage)

    train_config = load_config(real_path(options.train_config))
    model_config = load_config(real_path(options.model_config), add_symbol=True)

    trainer = Trainer(train_config, model_config)
    trainer.train()


if __name__ == "__main__":
    run_train()
