# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:10:26 2021

@author: lyn
"""
import tokenization

def test_tokenize():
    """
    """
    mimix_tokenizer = tokenization.MimixTokenizer(
            vocab_file="../model/vocab/zh_vocab.txt",
            pre_tokenized=False,
            pre_vectorized=False)
    
    print(mimix_tokenizer.tokenize("1234567号选手是top10哦,_mask_hello你好666啊,春 秋 忽 代 谢windows7le"))
    
    #tokenizer = tokenization.BertTokenizer(
    #        vocab_file="../../pretrain/bert-base-chinese/vocab.txt",
    #        pre_tokenized=False,
    #        pre_vectorized=False)
    
    #print(tokenizer.tokenize("1234567号选手是top10哦,hello你好666啊"))  
    #print(tokenizer.tokenize("春 秋 忽 代 谢windows7le"))
    
if __name__ == "__main__":
    
    test_tokenize()
    
    #mimix_tokenizer = tokenization.MimixTokenizer(
    #        vocab_file="../model/vocab/zh_vocab.txt",
    #        pre_tokenized=False,
    #        pre_vectorized=False)
    
    #import torch
    #state_dict = torch.load("../model/summ/tmp.summ.base.model.4",
    #                        map_location=lambda storage, loc: storage)
    
    
    
    