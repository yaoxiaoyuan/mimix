# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:16:38 2019

@author: Xiaoyuan Yao
"""

import os
from datetime import datetime
import logging
import torch
from mimix.utils import real_path

LOG_DIR = "logger"

local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
rank = int(os.environ['RANK'])

if rank == 0 and local_rank == 0:
    if not os.path.exists(real_path(LOG_DIR)):
        os.mkdir(real_path(LOG_DIR))
 

def build_logger():
    """
    """
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    filename = datetime.today().strftime('logger/%Y-%m-%d-%H-%M-%S-rank%d.log' % rank)
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

if local_rank == 0:
    logger = build_logger()
 
def save_model(model, optimizer, model_path):
    """
    """
    if rank == 0 and local_rank == 0:
        logger.info("Save model to %s" % model_path)
        torch.save(model.state_dict(),
                   model_path,
                   _use_new_zipfile_serialization=False)

        logger.info("Save model complete")


def print_model_info(model):
    """
    """
    if local_rank == 0:
        logger.info("%s" % model)
    total_params = sum(p.numel() for p in model.parameters())
    if local_rank == 0:
        logger.info("Total Model Params:%s" % total_params)
    total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad==True)
    if local_rank == 0:
        logger.info("Trainable Model Params:%s" % total_train_params)


def train(model,
          optimizer,
          train_config,
          train_generator,
          lr_scheduler=None): 
    """
    """
    if os.path.exists(real_path(train_config["model_dir"])) == False:
        os.mkdir(real_path(train_config["model_dir"]))
        
    print_model_info(model)
    
    if local_rank == 0:
        logger.info("Train Start!")
    
    print_every_n_steps = train_config.get("print_every_n_steps", 100)
    model_path = real_path(os.path.join(real_path(train_config["model_dir"]), "%s." + train_config["model_name"]))
    save_steps = train_config.get("save_steps", 100000)
    tmp_save_steps = train_config.get("tmp_save_steps", 10000)
    history_loss = []
    epoch,steps,total_steps = 0, 0, 0
    while epoch < train_config["max_epoch"]: 
        for inputs,targets in train_generator():
            inputs = [inputs.to(model.device)]
            targets = [targets.to(model.device)] 
            outputs = model(inputs, targets=targets, compute_loss=True)
            loss = outputs[0]        
            history_loss = history_loss[-999:] + [loss.item()]
                
            if total_steps % print_every_n_steps == 0:
                ma_loss = sum(history_loss) / len(history_loss)
                if local_rank == 0:
                    logger.info(
                        "%d epoch %d step total %d steps loss: %.3f" % 
                        (epoch, 
                         steps, 
                         total_steps,
                         ma_loss)
                    )

            model.backward(loss)
            model.step()
            total_steps += 1
            steps += 1
            
            if lr_scheduler is not None:
                lr_scheduler.step()

            if total_steps % save_steps == 0:
                save_model(model, optimizer, model_path % ("%d.%d.%d" % (epoch, steps, total_steps)))
    
            if total_steps % tmp_save_steps == 0:
                save_model(model, optimizer, model_path % "tmp")
            
        epoch += 1
        steps = 0
        
    save_model(model, optimizer, model_path % ("%d.%d.%d" % (epoch, steps, total_steps)))
    if local_rank == 0:
        logger.info("Train Completed!")
