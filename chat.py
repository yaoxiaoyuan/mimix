# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:41:32 2024

@author: Xiaoyuan Yao
"""
import sys
from argparse import ArgumentParser
import re
import platform
import os
from mimix.predictor import LMGenerator
from mimix.utils import real_path, load_model_config

def main():
    
    print("loading model...")

    parser = ArgumentParser()
    parser.add_argument("--model_conf", type=str, default="conf/MimixLM-0.7b-sft-0.2_conf") 
    args = parser.parse_args(sys.argv[1:])
    conf_file = args.model_conf

    model_config = load_model_config(real_path(conf_file))
    max_history_len = 2000
    max_history_turn = 20
    
    assert model_config["is_mimix_chat"] == True
    assert max_history_len < model_config["trg_max_len"]
   
    lm_gen = LMGenerator(model_config) 
    history = []
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    while True:
        print("User:")
        user_input = input()
        if user_input == ":restart":
            if platform.system() == "Windows":
                os.system("cls")
            else:
                os.system("clear")
            history = []
            continue
        elif user_input == ":quit":
            break
        history.append(user_input)
        context = " _mimix_"
        for i,text in enumerate(history[::-1]):
            if i > max_history_turn:
                break
            if len(context) > max_history_len:
                break
            if i % 2 == 0:          
                context = " _mimixuser_ " + text + context
            else:
                context = " _mimix_ " + text + context
        context = context.strip()
        
        search_res = lm_gen.predict_stream(prefix_list=[context])
        resp = ""
        print("Mimix:")
        while True:
            try:
                _resp = next(search_res)[0][1][0][0].split("_mimix_")[-1].strip()
                print(_resp[len(resp):], end="", flush=True)
                resp = _resp
            except:
                break
        print()
        
        history.append(resp)

if __name__ == "__main__":
    main()
