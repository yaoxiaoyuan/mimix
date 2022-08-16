# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 22:24:26 2022

@author: Xiaoyuan Yao
"""
import os
import sys
import json
import subprocess
import collections
import socket
import signal
import logging
import trainer_single_gpu
from utils import real_path, parse_train_args, load_config


def run_train_multi_gpu(nproc_per_node):
    """
    """
    cmd_launch = []
    cmd_launch.extend([sys.executable,'-m', 'torch.distributed.launch'])
    torch_distributed_args = [
                '--nproc_per_node',
                nproc_per_node,
            ]
    cmd_launch.extend(torch_distributed_args)
    cmd_launch.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   "trainer_multi_gpu.py"))
    cmd_launch.extend(sys.argv[1:])

    run_cmd = ' '.join(cmd_launch)
    p = subprocess.Popen(run_cmd, shell=True, preexec_fn=os.setsid)
    def signal_handler(signal, frame):
        os.killpg(os.getpgid(p.pid), 9)
    signal.signal(signal.SIGINT, signal_handler)
    p.wait()
    print ('finish')



def run_train():
    usage = "usage: run_train.py --model_conf <file> --train_conf <file>"
    options = parse_train_args(usage)

    train_config = load_config(real_path(options.train_config))
    model_config = load_config(real_path(options.model_config), add_symbol=True)
     
    if train_config["use_cuda"] == False or len(train_config.get("device_id", "0").split(",")) < 2:
        trainer = trainer_single_gpu.Trainer(train_config, model_config)
        trainer.train()
    else: 
        nproc_per_node = str(len(train_config["device_id"].split(",")))
        run_train_multi_gpu(nproc_per_node)        


if __name__ == "__main__":
    run_train()


