# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 18:15:21 2024

@author: 1
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
import re
import torch
from mimix.models import build_model
from mimix.utils import load_model_config

model = build_model(load_model_config("conf/mae_base_conf"))
a = torch.load("model/pretrain/vit/mae_visualize_vit_base.pth")["model"]
b = {}
for k in a:
    if k == "cls_token":
        b["encoder.cls"] = a[k][0,0,:]
    elif k == "mask_token":
        b["mask"] = a[k][0,0,:]
    elif k == "decoder_pred.weight":
        b["out_proj.W"] = a[k]
    elif k == "decoder_pred.bias":
        b["out_proj.b"] = a[k]
    elif k == "decoder_embed.weight":
        b["dec_embedding.W"] = a[k]
    elif k == "decoder_embed.bias":
        b["dec_embedding.b"] = a[k]
    elif k == "patch_embed.proj.weight":
        b["encoder.patch_embedding.weight"] = a[k]
    elif k == "patch_embed.proj.bias":
        b["encoder.patch_embedding.bias"] = a[k]
    elif k == "norm.weight":
        b["encoder.norm.alpha"] = a[k]
    elif k == "norm.bias":
        b["encoder.norm.bias"] = a[k]
    elif k == "decoder_norm.weight":
        b["decoder.norm.alpha"] = a[k]
    elif k == "decoder_norm.bias":
        b["decoder.norm.bias"] = a[k]
    elif re.search("blocks.[0-9]+.norm[0-9]+.weight", k):
        idx = re.findall("[0-9]+", k)[0]
        idx2 = re.findall("[0-9]+", k)[1]
        k2 = "encoder.layers.%s.norm_%s.alpha" % (idx, idx2)
        if "decoder" in k:
            k2 = k2.replace("encoder", "decoder")
        b[k2] = a[k]
    elif re.search("blocks.[0-9]+.norm[0-9]+.bias", k):
        idx = re.findall("[0-9]+", k)[0]
        idx2 = re.findall("[0-9]+", k)[1]
        k2 = "encoder.layers.%s.norm_%s.bias" % (idx, idx2)
        if "decoder" in k:
            k2 = k2.replace("encoder", "decoder")
        b[k2] = a[k]
    elif re.search("blocks.[0-9]+.attn.qkv.weight", k):
        idx = re.findall("[0-9]+", k)[0]
        d = a[k].shape[0] // 3
        k2 = "encoder.layers.%s.self_attention.W_q"  % idx
        if "decoder" in k:
            k2 = k2.replace("encoder", "decoder")
        b[k2] = a[k][:d]
        k2 = "encoder.layers.%s.self_attention.W_k" % idx
        if "decoder" in k:
            k2 = k2.replace("encoder", "decoder")
        b[k2] = a[k][d:2*d]
        k2 = "encoder.layers.%s.self_attention.W_v" % idx
        if "decoder" in k:
            k2 = k2.replace("encoder", "decoder")
        b[k2] = a[k][2*d:]
        
    elif re.search("blocks.[0-9]+.attn.qkv.bias", k):
        idx = re.findall("[0-9]+", k)[0]
        d = a[k].shape[0] // 3
        k2 = "encoder.layers.%s.self_attention.b_q"  % idx
        if "decoder" in k:
            k2 = k2.replace("encoder", "decoder")
        b[k2] = a[k][:d]
        k2 = "encoder.layers.%s.self_attention.b_k" % idx
        if "decoder" in k:
            k2 = k2.replace("encoder", "decoder")
        b[k2] = a[k][d:2*d]
        k2 = "encoder.layers.%s.self_attention.b_v" % idx
        if "decoder" in k:
            k2 = k2.replace("encoder", "decoder")
        b[k2] = a[k][2*d:]
        
    elif re.search("blocks.[0-9]+.attn.proj.weight", k):
        idx = re.findall("[0-9]+", k)[0]
        k2 = "encoder.layers.%s.self_attention.W_o"  % idx
        if "decoder" in k:
            k2 = k2.replace("encoder", "decoder")
        b[k2] = a[k]
    elif re.search("blocks.[0-9]+.attn.proj.bias", k):
        idx = re.findall("[0-9]+", k)[0]
        k2 = "encoder.layers.%s.self_attention.b_o"  % idx
        if "decoder" in k:
            k2 = k2.replace("encoder", "decoder")
        b[k2] = a[k]
    elif re.search("blocks.[0-9]+.mlp.fc[0-9]+.weight", k):
        idx = re.findall("[0-9]+", k)[0]
        idx2 = re.findall("[0-9]+", k)[1]
        k2 = "encoder.layers.%s.ffn.W%s" % (idx, idx2)
        if "decoder" in k:
            k2 = k2.replace("encoder", "decoder")
        b[k2] = a[k]
    elif re.search("blocks.[0-9]+.mlp.fc[0-9]+.bias", k):
        idx = re.findall("[0-9]+", k)[0]
        idx2 = re.findall("[0-9]+", k)[1]
        k2 = "encoder.layers.%s.ffn.b%s" % (idx, idx2)
        if "decoder" in k:
            k2 = k2.replace("encoder", "decoder")
        b[k2] = a[k]    
    elif re.search("pos_embed", k):
        k2 = "encoder.pos_embedding.W"
        if "decoder" in k:
            k2 = k2.replace("encoder", "decoder")
        b[k2] = a[k][0]

model.load_state_dict(b)

import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def run_one_image(img, model):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    model.eval()
    with torch.no_grad():

        outputs = model([x.float(), 0.75])
        dec_output = outputs["output"]
        reconstruct = outputs["reconstruct"]
        mask = outputs["mask"]
        patchify_x = outputs["patchify_x"]
        
        y = torch.einsum('nchw->nhwc', reconstruct).cpu()
            
    print(mask)
    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.pw*model.ph*model.n_channels)  # (N, H*W, p*p*3)
    print(mask)
    h = model.img_h // model.ph
    w = model.img_w // model.pw     
    mask = mask.reshape(shape=(mask.shape[0], h, w, model.ph, model.pw, model.n_channels))
    print(mask)
    mask = torch.einsum('nhwpqc->nchpwq', mask)
    mask = mask.reshape(shape=(mask.shape[0], model.n_channels, model.img_h, model.img_w))
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    print(mask)

    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.show()


torch.save(model.state_dict(), "model/mae/mae.base.model")
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
# load an image
img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg' # fox, from ILSVRC2012_val_00046145
# img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg' # cucumber, from ILSVRC2012_val_00047851
img = Image.open(requests.get(img_url, stream=True).raw)
img = img.resize((224, 224))
img = np.array(img) / 255.

assert img.shape == (224, 224, 3)

# normalize by ImageNet mean and std
img = img - imagenet_mean
img = img / imagenet_std

plt.rcParams['figure.figsize'] = [5, 5]
show_image(torch.tensor(img))

torch.manual_seed(2)
print('MAE with pixel reconstruction:')
run_one_image(img, model)