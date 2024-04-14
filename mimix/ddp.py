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

if local_rank == 0:
    if not os.path.exists(real_path(LOG_DIR)):
        os.mkdir(real_path(LOG_DIR))
 

def build_logger():
    """
    """
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    filename = datetime.today().strftime('logger/%Y-%m-%d-%H-%M-%S.log')
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
    if local_rank == 0:
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
          val_generator=None, 
          test_generator=None, 
          eval_fn_list=None,
          lr_scheduler=None):
    """
    """
    if os.path.exists(real_path(train_config["model_dir"])) == False:
        os.mkdir(real_path(train_config["model_dir"]))
        
    use_amp = train_config.get("use_amp", False)
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
            
    print_model_info(model)
    
    if local_rank == 0:
        logger.info("Train Start!")
    
    accumulate_steps = train_config.get("accumulate_steps", 1)
    print_every_n_steps = train_config.get("print_every_n_steps", 100)
    model_path = real_path(os.path.join(real_path(train_config["model_dir"]), "%s." + train_config["model_name"]))
    save_steps = train_config.get("save_steps", 100000)
    tmp_save_steps = train_config.get("tmp_save_steps", 10000)
    grad_clip = train_config.get("grad_clip", None)
    
    history_loss = []
    
    epoch,steps,total_steps = 0, 0, 0
    while epoch < train_config["max_epoch"]: 
        model.train()
        
        for inputs,targets in train_generator():
            if use_amp == True:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs, targets=targets, compute_loss=True)
                    loss = outputs["loss"]        
                    history_loss = history_loss[-999:] + [loss.item()]
                    loss = loss / accumulate_steps
            else:
                outputs = model(inputs, targets=targets, compute_loss=True)                    
                loss = outputs["loss"]    
                history_loss = history_loss[-999:] + [loss.item()]
                loss = loss / accumulate_steps
                
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

            if use_amp == True:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if lr_scheduler is not None:
                lr_scheduler.step()
 
            total_steps += 1
            steps += 1
    
            if total_steps % save_steps == 0:
                save_model(model, optimizer, model_path % ("%d.%d.%d" % (epoch, steps, total_steps)))
    
            if total_steps % tmp_save_steps == 0:
                save_model(model, optimizer, model_path % "tmp")
            
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        grad_clip
                        )
            
            if total_steps % accumulate_steps == 0:
                if use_amp == True:
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()
                else:
                    optimizer.step()
                    optimizer.zero_grad()
        
        epoch += 1
        steps = 0
        
        if len(eval_fn_list) > 0:
            if val_generator is not None:
                if local_rank == 0:
                    logger.info("Eval val now...")
                for eval_fn in eval_fn_list:
                    eval_res = eval_fn(model, val_generator)
                    if local_rank == 0:
                        logger.info("Result: %s" % eval_res)
            if test_generator is not None:
                if local_rank == 0:
                    logger.info("Eval test now...")
                for eval_fn in eval_fn_list:
                    eval_res = eval_fn(model, test_generator)
                    if local_rank == 0:
                        logger.info("Result: %s" % eval_res)
    save_model(model, optimizer, model_path % ("%d.%d.%d" % (epoch, steps, total_steps)))
    if local_rank == 0:
        logger.info("Train Completed!")
