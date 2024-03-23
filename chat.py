# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:41:32 2024

@author: Xiaoyuan Yao
"""
import sys
from argparse import ArgumentParser
from mimix.utils import real_path, load_model_config
from mimix.interact import chat

def main():
    
    parser = ArgumentParser()
    parser.add_argument("--model_conf", type=str, default="conf/MimixLM-0.7b-sft-0.2_conf") 
    args = parser.parse_args(sys.argv[1:])
    conf_file = args.model_conf

    config = load_model_config(real_path(conf_file))
    chat(config)

if __name__ == "__main__":
    main()
