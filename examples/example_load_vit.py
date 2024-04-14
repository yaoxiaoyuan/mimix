# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 23:36:34 2023

@author: Xiaoyuan Yao
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
import re
import numpy as np
import torch
from mimix.models import TransformerEncoder

a = np.load("model/pretrain/vit/imagenet21k+imagenet2012_ViT-B_32.npz")
b = {}
for k in a.files:
    k2 = k.replace('Transformer/', 'encoder.')
    k2 = re.sub('encoderblock_([0-9]+)/', 'layers.\\1.', k2)
    k2 = k2.replace('MultiHeadDotProductAttention_1/', 'self_attention.')
    k2 = k2.replace('query/kernel', 'W_q')
    k2 = k2.replace('query/bias', 'b_q')
    k2 = k2.replace('key/kernel', 'W_k')
    k2 = k2.replace('key/bias', 'b_k')
    k2 = k2.replace('value/kernel', 'W_v')
    k2 = k2.replace('value/bias', 'b_v')
    k2 = k2.replace('out/kernel', 'W_o')
    k2 = k2.replace('out/bias', 'b_o')
    k2 = k2.replace('LayerNorm_0/scale', "norm_1.alpha")
    k2 = k2.replace('LayerNorm_0/bias', "norm_1.bias")
    k2 = k2.replace('LayerNorm_2/scale', "norm_2.alpha")
    k2 = k2.replace('LayerNorm_2/bias', "norm_2.bias")
    k2 = k2.replace('MlpBlock_3/Dense_0/kernel', "ffn.W1")
    k2 = k2.replace('MlpBlock_3/Dense_0/bias', "ffn.b1")
    k2 = k2.replace('MlpBlock_3/Dense_1/kernel', "ffn.W2")
    k2 = k2.replace('MlpBlock_3/Dense_1/bias', "ffn.b2")

    k2 = k2.replace('encoder_norm/bias', 'norm.bias')
    k2 = k2.replace('encoder_norm/scale', 'norm.alpha')

    w = torch.from_numpy(a[k])    
    if k == "cls":
        w = w.flatten()
        k2 = 'encoder.cls'
    if k == 'head/kernel':
        k2 = 'W_cls'
    if k == 'head/bias':
        k2 = 'b_cls'  
    
    if k == 'embedding/kernel':
        k2 = 'encoder.patch_embedding.weight'
        w = torch.einsum('abcd->dcab', w)
    elif 'kernel' in k and "head" not in k:
        if "attention" in k2:
            w = w.view(a["embedding/bias"].shape[0], a["embedding/bias"].shape[0])
        w = w.transpose(0, 1)
        
    if "Attention" in k and "bias" in k:
        w = w.flatten(0)
        
    if k == 'embedding/bias':
        k2 = 'encoder.patch_embedding.bias'
    
    if 'pos_embedding' in k:
        k2 = 'encoder.pos_embedding.W'
        w = w.squeeze(0)
    
    b[k2] = w


vit = TransformerEncoder(use_vit_encoder=True,
                         d_model=768,
                         n_heads=12,
                         img_w=384,
                         img_h=384,
                         patch_w=32,
                         patch_h=32,
                         n_class=1000,
                         n_enc_layers=12,
                         n_channels=3,
                         activation="gelu",
                         use_pre_norm=True,
                         norm_before_pred=True,
                         ln_eps=1e-6)
vit.load_state_dict(b)

from PIL import Image
#from imagenet 
image = Image.open("n01440764_10026.jpg")
from torchvision import transforms
transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5, 0.5),
            ])
x = transform(image).unsqueeze(0)

vit.eval()
with torch.no_grad():
    outputs = vit([x])

#predict label: 0    
print(outputs["cls_logits"].argmax(-1))    
    