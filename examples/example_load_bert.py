# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 22:27:06 2023

@author: Xiaoyuan Yao
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
import json
import torch
from mimix.tokenization import build_tokenizer

def load_bert_weights(bert, weight_path):
    """
    """
    bert.eval()
    state_dict = torch.load(weight_path)
    
    map_key_dict = {"encoder.word_embedding.W": "bert.embeddings.word_embeddings.weight",
                    "encoder.pos_embedding.W": "bert.embeddings.position_embeddings.weight",
                    "encoder.type_embedding.W":"bert.embeddings.token_type_embeddings.weight",
                    "encoder.norm_emb.alpha": "bert.embeddings.LayerNorm.gamma",
                    "encoder.norm_emb.bias": "bert.embeddings.LayerNorm.beta",
                    "W_pool": "bert.pooler.dense.weight",
                    "b_pool": "bert.pooler.dense.bias",
                    "W_cls": "cls.seq_relationship.weight",
                    "b_cls": "cls.seq_relationship.bias",        
                    "W_mlm": "cls.predictions.transform.dense.weight",
                    "b_mlm": "cls.predictions.transform.dense.bias",
                    "norm_mlm.alpha": "cls.predictions.transform.LayerNorm.gamma",
                    "norm_mlm.bias": "cls.predictions.transform.LayerNorm.beta",
                    "b_out_mlm": "cls.predictions.bias"}
    
    for i in range(bert.n_layers):
        map_key_dict["encoder.layers.%d.self_attention.W_q" % i] = "bert.encoder.layer.%d.attention.self.query.weight" % i
        map_key_dict["encoder.layers.%d.self_attention.b_q" % i] = "bert.encoder.layer.%d.attention.self.query.bias" % i
        map_key_dict["encoder.layers.%d.self_attention.W_k" % i] = "bert.encoder.layer.%d.attention.self.key.weight" % i
        map_key_dict["encoder.layers.%d.self_attention.b_k" % i] = "bert.encoder.layer.%d.attention.self.key.bias" % i
        map_key_dict["encoder.layers.%d.self_attention.W_v" % i] = "bert.encoder.layer.%d.attention.self.value.weight" % i
        map_key_dict["encoder.layers.%d.self_attention.b_v" % i] = "bert.encoder.layer.%d.attention.self.value.bias" % i
        map_key_dict["encoder.layers.%d.self_attention.W_o" % i] = "bert.encoder.layer.%d.attention.output.dense.weight" % i
        map_key_dict["encoder.layers.%d.self_attention.b_o" % i] = "bert.encoder.layer.%d.attention.output.dense.bias" % i
        map_key_dict["encoder.layers.%d.norm_1.alpha" % i] = "bert.encoder.layer.%d.attention.output.LayerNorm.gamma" % i
        map_key_dict["encoder.layers.%d.norm_1.bias" % i] = "bert.encoder.layer.%d.attention.output.LayerNorm.beta" % i
        map_key_dict["encoder.layers.%d.ffn.W1" % i] = "bert.encoder.layer.%d.intermediate.dense.weight" % i
        map_key_dict["encoder.layers.%d.ffn.b1" % i] = "bert.encoder.layer.%d.intermediate.dense.bias" % i
        map_key_dict["encoder.layers.%d.ffn.W2" % i] = "bert.encoder.layer.%d.output.dense.weight" % i
        map_key_dict["encoder.layers.%d.ffn.b2" % i] = "bert.encoder.layer.%d.output.dense.bias" % i
        map_key_dict["encoder.layers.%d.norm_2.alpha" % i] = "bert.encoder.layer.%d.output.LayerNorm.gamma" % i
        map_key_dict["encoder.layers.%d.norm_2.bias" % i] = "bert.encoder.layer.%d.output.LayerNorm.beta" % i
    
    if bert.share_emb_out_proj == False:
        map_key_dict["W_out_mlm"] = "cls.predictions.decoder.weight"
    
    model_state_dict = {}
    for key,param in bert.named_parameters():
        model_state_dict[key] = state_dict[map_key_dict[key]]
        #model_state_dict[key] = state_dict[map_key_dict[key].replace("gamma", "weight").replace("beta", "bias")]
        if key == "W_out_mlm":
            model_state_dict[key] = state_dict[map_key_dict[key]].T
    
    bert.load_state_dict(model_state_dict, False)
    
    return bert


def load_bert_model(model_path, use_cuda=False):
    """
    """
    config = json.load(open(os.path.join(model_path, "config.json")))
    mimix_config = {}
    mimix_config["attn_dropout"] = config["attention_probs_dropout_prob"]
    mimix_config["activation"] = config["hidden_act"]
    mimix_config["dropout"] = config["hidden_dropout_prob"]
    mimix_config["d_model"] = config["hidden_size"]
    mimix_config["d_ff"] = config["intermediate_size"]
    mimix_config["ln_eps"] = config["layer_norm_eps"]
    mimix_config["src_max_len"] = config["max_position_embeddings"]
    mimix_config["n_heads"] = config["num_attention_heads"]
    mimix_config["n_enc_layers"] = config["num_hidden_layers"]
    mimix_config["n_types"] = config["type_vocab_size"]
    mimix_config["src_vocab_size"] = config["vocab_size"]
    mimix_config["use_pre_norm"] = False
    mimix_config["norm_after_embedding"] = True
    mimix_config["with_mlm"] = True
    from mimix.models import TransformerEncoder    
    mimix_config["symbols"] = {"_pad_": "[PAD]",
                               "_bos_": "[unused1]",
                               "_eos_": "[unused2]",
                               "_unk_": "[UNK]",
                               "_cls_": "[CLS]",
                               "_sep_": "[SEP]",
                               "_mask_": "[MASK]"
                               }
    vocab = {line.strip():i for i,line in enumerate(open(os.path.join(model_path, "vocab.txt"), "r", encoding="utf-8"))}
    mimix_config["symbol2id"] = {k:vocab[mimix_config["symbols"][k]] for k in mimix_config["symbols"]}
    bert = TransformerEncoder(**mimix_config)
    bert = load_bert_weights(bert, os.path.join(model_path, "pytorch_model.bin"))
    if use_cuda == True:
        bert = bert.cuda()
    return bert


bert_model_path = "model/pretrain/bert"
#bert_model_path = "model/pretrain/bert_large"
model = load_bert_model(bert_model_path)
model.eval()
tokenizer = build_tokenizer(**{"tokenizer":"bert", "vocab_file":os.path.join(bert_model_path, "vocab.txt")})

#Test for Chinese BERT MLM Task: [MASK]国的首都是曼谷
#output: ['泰'] 0.9495071768760681
x = [101,103] + tokenizer.tokenize_to_ids("国的首都是曼谷") + [102]
y = model([torch.tensor([x], dtype=torch.long), torch.zeros([1,len(x)], dtype=torch.long)])["mlm_logits"]
prob = torch.softmax(y, -1)
word_id = y[0][x.index(103)].argmax().item()
prob = prob[0][x.index(103)][word_id].item()
print(tokenizer.convert_ids_to_tokens([word_id]), prob)

#Test for Chinese BERT MLM Task: 韩国的首都是[MASK]尔
#output: ['首'] 0.9999905824661255
x = [101] + tokenizer.tokenize_to_ids("韩国的首都是") + [103] + tokenizer.tokenize_to_ids("尔") + [102]
y = model([torch.tensor([x], dtype=torch.long), torch.zeros([1,len(x)], dtype=torch.long)])["mlm_logits"]
prob = torch.softmax(y, -1)
word_id = y[0][x.index(103)].argmax().item()
prob = prob[0][x.index(103)][word_id].item()
print(tokenizer.convert_ids_to_tokens([word_id]), prob)
