# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:10:26 2021

@author: lyn
"""
import torch
from tokenizer import WordPieceTokenizer
from utils import load_vocab
from layers import CRF
from decoding import crf_decoding

def test_tokenize():
    """
    """
    tokenizer = WordPieceTokenizer(vocab="../data/vocab/zh_vocab.txt",
                                   pre_tokenized=False,
                                   pre_vectorized=False)
    
    print(tokenizer.tokenize_to_str("1234567是top100哦"))
    

if __name__ == "__main__":
    
    crf = CRF(10)    
    #crf.start_trans.data.fill_(0)
    #crf.end_trans.data.fill_(0)
    emission = torch.rand([3,9,10])
    
    emission[0,:] = emission[1,:]
    target = torch.randint(1, 10, [3,9])
    target[0,:] = target[1,:]
    target[0,3:] = 0
    mask = target.ne(0).float()
    target[0,3:] = target[0,0]
    print(crf(emission, target, mask))
    
    
    #print(crf_decoding(crf, emission, mask))
    

    
    
    
    
    
    
    