# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 20:54:40 2024

@author: 1
"""
import re
from argparse import ArgumentParser
import sys
import time
import numpy as np
from mimix.predictor import TextEncoder
from mimix.utils import real_path, load_model_config

def sudoku_demo():
    """
    """
    conf_file = "conf/sudoku_bert_base_conf"
    config = load_model_config(real_path(conf_file))
    lm_gen = TextEncoder(config)
    
    print("INPUT PUZZLE:")
    
    for line in sys.stdin:
        line = line.strip()
        
        if len(line) != 81 or re.search("[^0-9]", line):
            print("invalid puzzle!")
            continue

        arr = np.zeros([9, 9], dtype=np.int64)
        for i,w in enumerate(line):
            arr[i//9][i%9] = int(w)
            
        print("puzzle:")
        print(arr)
        
        res = lm_gen.predict_mlm([" ".join(line)])
        for i in range(81):
            if arr[i//9][i%9] == 0:
                arr[i//9][i%9] = int(res[0][1][i+1][0][0])
        print("predict:")
        print(arr)
        
        flag = True
        for i in range(9):
            flag = flag and all(j in arr[i,:] for j in range(1, 10))
        for i in range(9):
            flag = flag and all(j in arr[:,i] for j in range(1, 10))                
        for i in range(3):
            for j in range(3):
                flag = flag and all(k in arr[3*i:3*i+3,3*j:3*j+3] for k in range(1, 10))
        
        if flag == True:
            print("solve success!")
        else:
            print("solve failed!")
            
        
if __name__ == "__main__":
    sudoku_demo()
