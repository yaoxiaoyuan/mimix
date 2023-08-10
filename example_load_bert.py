# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 22:27:06 2023

@author: 1
"""
import os
import torch
from mimix.bert import load_bert_model
from mimix.tokenization import build_tokenizer

bert_model_path = "model/pretrain/bert"
model = load_bert_model(bert_model_path)
model.eval()
tokenizer = build_tokenizer(**{"tokenizer":"bert", "vocab_file":os.path.join(bert_model_path, "vocab.txt")})

#Test for Chinese BERT MLM Task: [MASK]国的首都是曼谷
#output: ['泰'] 0.9495071768760681
x = tokenizer.tokenize_to_ids("国的首都是曼谷")
y = model([torch.tensor([[101,103]+x+[102]], dtype=torch.long), torch.zeros([1,3+len(x)], dtype=torch.long)])[0]
prob = torch.softmax(y, -1)
word_id = y[0][1].argmax().item()
prob = prob[0][1][word_id].item()
print(tokenizer.convert_ids_to_tokens([word_id]), prob)

#Test for Chinese BERT MLM Task: [MASK]国的首都是首尔
#output: ['韩'] 0.7965040802955627
x = tokenizer.tokenize_to_ids("国的首都是首尔")
y = model([torch.tensor([[101,103]+x+[102]], dtype=torch.long), torch.zeros([1,3+len(x)], dtype=torch.long)])[0]
prob = torch.softmax(y, -1)
word_id = y[0][1].argmax().item()
prob = prob[0][1][word_id].item()
print(tokenizer.convert_ids_to_tokens([word_id]), prob)
