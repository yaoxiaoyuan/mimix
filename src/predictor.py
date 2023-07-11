# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:34:48 2019

@author: Xiaoyuan Yao
"""
import random
import torch
import torch.nn.functional as F
from tokenization import build_tokenizer
from decoding import search
from decoding import crf_model_decoding
from utils import load_model, load_vocab, real_path, invert_dict
from utils import cut_and_pad_seq_list

class EncDecGenerator():
    """
    """
    def __init__(self, config):
        """
        """
        self.model = load_model(config)

        self.use_cuda = config.get("use_cuda", False)
        self.device = torch.device('cpu')
        if self.use_cuda == True:
            device_id = config.get("device_id", "0")
            self.device = torch.device('cuda:%s' % device_id)
            
        self.src_tokenizer = build_tokenizer(
                tokenizer=config["src_tokenizer"],
                vocab_file=config["src_vocab"], 
                pre_tokenized=config.get("pre_tokenized", False),  
                pre_vectorized=config.get("pre_vectorized", False))
        self.trg_tokenizer = build_tokenizer(
                tokenizer=config["trg_tokenizer"],
                vocab_file=config["trg_vocab"], 
                pre_tokenized=config.get("pre_tokenized", False),  
                pre_vectorized=config.get("pre_vectorized", False))
        
        vocab = load_vocab(real_path(config["trg_vocab"]))
        self.trg_word2id = vocab
        self.trg_id2word = {vocab[word]:word for word in vocab}

        vocab = load_vocab(real_path(config["src_vocab"]))
        self.src_word2id = vocab
        self.src_id2word = {vocab[word]:word for word in vocab}
        
        self.add_cls = config.get("add_cls", False)
        self.eos_tok = config["symbols"]["EOS_TOK"]
        self.pad_tok = config["symbols"]["PAD_TOK"]
        
        self.src_max_len = config["src_max_len"]
        self.trg_max_len = config["trg_max_len"]
        self.trg_vocab_size = config["trg_vocab_size"]
        self.max_dec_steps = config["max_decode_steps"]
        
        self.beam_size = config.get("beam_size", 3)
        self.group_size = config.get("group_size", -1)
        self.gamma = float(config.get("gamma", 1))
        self.temperature = config.get("temperature", 1)
        self.strategy = config.get("search_strategy", "beam_search")
        self.top_k = config.get("top_k", -1)
        self.top_p = config.get("top_p", -1)
        self.repeat_penalty = config.get("repeat_penalty", 0)
        self.normalize = config.get("normalize", "none")
        self.use_mask_unk =  config.get("use_mask_unk", False)
        self.use_cuda = config["use_cuda"] 
    
    
    def encode_inputs(self, 
                      src_list, 
                      trg_list=None, 
                      add_bos=False, 
                      add_eos=False):
        """
        """        
        src_ids = list(map(self.src_tokenizer.tokenize_to_ids, src_list))
        x = cut_and_pad_seq_list(src_ids,
                                 self.src_max_len, 
                                 self.model.PAD,
                                 True)
        if self.add_cls == True:
            x = [[self.model.CLS] + xx[:self.src_max_len-1] for xx in x]
            
        y = None
        if trg_list is not None:
            prefix_ids = list(map(self.trg_tokenizer.tokenize_to_ids, trg_list))
            if add_bos == True:
                prefix_ids = [[self.model.BOS] + seq for seq in prefix_ids]
            if add_eos == True:
                prefix_ids = [seq + [self.model.EOS] for seq in prefix_ids]
                
            y = cut_and_pad_seq_list(prefix_ids,
                                     self.trg_max_len, 
                                     self.model.PAD,
                                     True)
            

        x = torch.tensor(x, dtype=torch.long)
        if y is not None:
            y = torch.tensor(y, dtype=torch.long)

        if self.use_cuda == True:
            x = x.to(self.device)
            if y is not None:
                y = y.to(self.device)
                
        return x,y
    
    
    def predict(self, src_list, prefix_list=None):
        """
        """    
        self.model.eval()
        x,y = self.encode_inputs(src_list, prefix_list)
        
        with torch.no_grad():
            if self.strategy in ["beam_search", "sample"]:
                states, cache = search(self.model, 
                                       self.beam_size, 
                                       inputs=[x,y],
                                       use_cuda=self.use_cuda,
                                       strategy=self.strategy,
                                       top_k=self.top_k,
                                       top_p=self.top_p,
                                       temperature=self.temperature,
                                       eos=self.model.EOS,
                                       group_size=self.group_size, 
                                       repeat_penalty=self.repeat_penalty,
                                       use_mask_unk=self.use_mask_unk,
                                       max_decode_steps=self.max_dec_steps)
                hypothesis,scores = states[4], states[1]
            else:
                raise ValueError("strategy not correct!")
        
        hypothesis = hypothesis.cpu().numpy()
        scores = scores.cpu().numpy()
            
        res = []
        for i,src in enumerate(src_list):
            tmp = []
            for j in range(i*self.beam_size, (i+1)*self.beam_size):
                trg = self.trg_tokenizer.detokenize_ids(hypothesis[j])
                
                trg = trg.replace(self.pad_tok, "").strip()
                if trg.endswith(self.eos_tok):
                    trg = trg.replace(self.eos_tok, "").strip()
                else:
                    trg = trg + " _unfinished_"
                
                tmp.append([trg, float(scores[j])])
            
            res.append([src, tmp])
        
        return res


class LMGenerator():
    """
    """
    def __init__(self, config):
        """
        """
        self.model = load_model(config)

        self.use_cuda = config.get("use_cuda", False)
        self.device = torch.device("cpu")
        if self.use_cuda == True:
            device_id = config.get("device_id", "0")
            self.device = torch.device('cuda:%s' % device_id)
            self.model = self.model.to(self.device)
        
        self.trg_tokenizer = build_tokenizer(
                tokenizer=config["trg_tokenizer"],
                vocab_file=config["trg_vocab"], 
                pre_tokenized=config.get("pre_tokenized", False),  
                pre_vectorized=config.get("pre_vectorized", False))
        self.use_cuda = config.get("use_cuda", False)
        self.gamma = float(config["gamma"])
        self.trg_max_len = config["trg_max_len"]
        self.trg_vocab_size = config["trg_vocab_size"]
        self.max_dec_steps = config["max_decode_steps"]
        self.use_cuda = config["use_cuda"]

        self.eos_tok = config["symbols"]["EOS_TOK"]
        self.pad_tok = config["symbols"]["PAD_TOK"]
        
        self.beam_size = config.get("beam_size", 3)
        self.group_size = config.get("group_size", -1)
        self.gamma = float(config.get("gamma", 1))
        self.temperature = config.get("temperature", 1)
        self.strategy = config.get("search_strategy", "beam_search")
        self.top_k = config.get("top_k", -1)
        self.top_p = config.get("top_p", -1)
        self.repeat_penalty = config.get("repeat_penalty", 0)
        self.normalize = config.get("normalize", "none")
        self.use_mask_unk =  config.get("use_mask_unk", False)
        self.use_cuda = config["use_cuda"] 
        

    def encode_inputs(self, trg_list):
        """
        """
        trg_ids = list(map(self.trg_tokenizer.tokenize_to_ids, trg_list))
        y = cut_and_pad_seq_list(trg_ids,
                                 self.trg_max_len, 
                                 self.model.PAD,
                                 True)
            
        if y is not None:
            y = torch.tensor(y, dtype=torch.long)

        if self.use_cuda == True:
            if y is not None:
                y = y.to(self.device)
        
        return y

    
    def sample(self, prefix_list):
        """
        """
        y = None
        if prefix_list is not None:
            prefix_list = [prefix_list[i] for i in range(len(prefix_list))]
            y = self.encode_inputs(prefix_list)

        self.model.eval()
        with torch.no_grad():
            if self.strategy in ["beam_search", "sample"]:
                states, cache = search(self.model, 
                                       self.beam_size, 
                                       inputs=[y],
                                       use_cuda=self.use_cuda,
                                       strategy=self.strategy,
                                       top_k=self.top_k,
                                       top_p=self.top_p,
                                       temperature=self.temperature,
                                       eos=self.model.EOS,
                                       group_size=self.group_size, 
                                       repeat_penalty=self.repeat_penalty,
                                       use_mask_unk=self.use_mask_unk,
                                       max_decode_steps=self.max_dec_steps)
                hypothesis,scores = states[4], states[1]
            else:
                raise ValueError("strategy not correct!")
                    
        hypothesis = hypothesis.cpu().numpy()
        scores = scores.cpu().numpy()

        detokenize_res = []
        for hyp,score in zip(hypothesis, scores):
            trg = self.trg_tokenizer.detokenize_ids(hyp)

            trg = trg.replace(self.pad_tok, "").strip()
            if trg.endswith(self.eos_tok):
                trg = trg.replace(self.eos_tok, "").strip()
            else:
                trg = trg + " _unfinished_"
                
            detokenize_res.append([trg, score])
        detokenize_res = [detokenize_res]
        
        return detokenize_res


class BiLMGenerator():
    """
    """
    def __init__(self, config):
        """
        """
        self.model = load_model(config)
        
        self.use_cuda = config.get("use_cuda", False)
        self.device = torch.device('cpu')
        if self.use_cuda == True:
            device_id = config.get("device_id", "0")
            self.device = torch.device('cuda:%s' % device_id)
        
        self.src_word2id = load_vocab(real_path(config["src_vocab"]))
        self.src_id2word = invert_dict(self.src_word2id)
        
        self.src_tokenizer = build_tokenizer(
                tokenizer=config["src_tokenizer"],
                vocab_file=config["src_vocab"], 
                pre_tokenized=config.get("pre_tokenized", False),  
                pre_vectorized=config.get("pre_vectorized", False))
        self.add_cls = config.get("add_cls", False)
        
        self.mask_id = config["symbol2id"][config["symbols"]["MASK_TOK"]]
        
        self.src_vocab_size = config["src_vocab_size"]
        self.use_cuda = config["use_cuda"]
        self.src_max_len = config["src_max_len"]
    
    
    def encode_inputs(self, src_list):
        """
        """        
        src_ids = list(map(self.src_tokenizer.tokenize_to_ids, src_list))

        y = cut_and_pad_seq_list(src_ids,
                                 self.src_max_len, 
                                 self.model.PAD,
                                 True)
        if self.add_cls == True:
            y = [[self.model.CLS] + yy[:self.src_max_len-1] for yy in y]

        if y is not None:
            y = torch.tensor(y, dtype=torch.long)

        if self.use_cuda == True:
            if y is not None:
                y = y.to(self.device)
        
        return y


    def predict(self, src_list, top_k=-1, top_p=-1, return_k=10):
        """
        """
        y = self.encode_inputs(src_list)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model([y])
            logits = outputs[0]
            probs = torch.softmax(logits, -1)
            if top_k > 0 or top_p > 0:
                return_k = 1
                indice = top_k_top_p_sampling(logits, top_k, top_p)
            else:
                score,indice = probs.topk(return_k, -1)

        indice = indice.cpu().numpy()
        score = score.cpu().numpy()
        
        res = []
        for i,src in enumerate(y):
            res.append([src_list[i], []])
            for j,ww in enumerate(src):

                if ww == self.mask_id:
                    pred = []
                    for k in range(return_k):
                        pred.append([self.src_id2word[indice[i][j][k]], score[i][j][k]])

                    res[-1][1].append(pred)
        
        return res
        

class TextMatcher():
    """
    """
    def __init__(self, config):
        """
        """
        self.model = load_model(config)
        
        self.use_cuda = config.get("use_cuda", False)
        self.device = torch.device('cpu')
        if self.use_cuda == True:
            device_id = config.get("device_id", "0")
            self.device = torch.device('cuda:%s' % device_id)
        
        self.src_word2id = load_vocab(real_path(config["src_vocab"]))
        self.src_id2word = invert_dict(self.src_word2id)
        
        self.src_tokenizer = build_tokenizer(
                tokenizer=config["src_tokenizer"],
                vocab_file=config["src_vocab"], 
                pre_tokenized=config.get("pre_tokenized", False),  
                pre_vectorized=config.get("pre_vectorized", False))
        
        self.add_cls = config.get("add_cls", False)
        self.cls_id = config["symbol2id"][config["symbols"]["CLS_TOK"]]
        
        self.src_vocab_size = config["src_vocab_size"]
        self.use_cuda = config["use_cuda"]
        self.src_max_len = config["src_max_len"]
    
    
    def encode_inputs(self, src_list):
        """
        """        
        src_ids = list(map(self.src_tokenizer.tokenize_to_ids, src_list))
        y = cut_and_pad_seq_list(src_ids,
                                 self.src_max_len-1, 
                                 self.model.PAD,
                                 True)
        if self.add_cls == True:
            y = [[self.cls_id] + yy[:self.src_max_len-1] for yy in y]

        if y is not None:
            y = torch.tensor(y, dtype=torch.long)

        if self.use_cuda == True:
            if y is not None:
                y = y.to(self.device)
        
        return y


    def encode_texts(self, src_list):
        """
        """
        y = self.encode_inputs(src_list)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model([y])
        
        return outputs[0]


    def predict(self, src_list):
        """
        """
        y = self.encode_inputs(src_list)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model([y])
            norm_vec = F.normalize(outputs[0][:,0,:], p=2, dim=1)
            sim = torch.mm(norm_vec, norm_vec.T)
        sim = sim.cpu().numpy()
        return sim


class TextClassifier():
    """
    """
    def __init__(self, config):
        """
        """
        self.model = load_model(config)

        self.use_cuda = config.get("use_cuda", False)
        self.device = torch.device('cpu')
        if self.use_cuda == True:
            device_id = config.get("device_id", "0")
            self.device = torch.device('cuda:%s' % device_id)

        self.label2id = None
        self.id2label = None
        if "label2id" in config:
            self.label2id = load_vocab(real_path(config["label2id"]))
            self.id2label = invert_dict(self.label2id)
        
        self.src_tokenizer = build_tokenizer(
                tokenizer=config["src_tokenizer"],
                vocab_file=config["src_vocab"], 
                pre_tokenized=config.get("pre_tokenized", False),  
                pre_vectorized=config.get("pre_vectorized", False))

        self.src_max_len = config["src_max_len"]
        self.num_class = config["n_class"]
        
        self.use_cuda = config["use_cuda"]
    

    def encode_inputs(self, src_list):
        """
        """        
        src_ids = list(map(self.src_tokenizer.tokenize_to_ids, src_list))
        x = cut_and_pad_seq_list(src_ids,
                                 self.src_max_len-1, 
                                 self.model.PAD,
                                 True)        
        x = [[self.model.CLS] + xx[:self.src_max_len-1] for xx in x]

        x = torch.tensor(x, dtype=torch.long)
        if self.use_cuda == True:
            x = x.to(self.device)
        
        return x        


    def predict(self, src_list):
        """
        """
        x = self.encode_inputs(src_list)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model([x])
        
        logits = outputs[0]

        y = torch.softmax(logits, 1)
        prob,label = torch.topk(y, self.num_class)
        prob = prob.cpu().numpy()
        label = label.cpu().numpy()
          
        res = []
        for i,src in enumerate(src_list):
            res.append([src_list[i], []])
            for yy,ss in zip(prob[i,:], label[i,:]):
                label_str = str(ss)
                if self.id2label is not None:
                    label_str = self.id2label[ss]
                
                res[i][1].append([label_str, yy])
        
        return res


class SequenceLabeler():
    """
    """
    def __init__(self, config):
        """
        """
        self.model = load_model(config)

        self.use_cuda = config.get("use_cuda", False)
        self.device = torch.device('cpu')
        if self.use_cuda == True:
            device_id = config.get("device_id", "0")
            self.device = torch.device('cuda:%s' % device_id)

        self.label2id = load_vocab(real_path(config["label2id"]))
        self.id2label = invert_dict(self.label2id)
        
        self.src_tokenizer = build_tokenizer(
                tokenizer=config["src_tokenizer"],
                vocab_file=config["src_vocab"], 
                pre_tokenized=config.get("pre_tokenized", False),  
                pre_vectorized=config.get("pre_vectorized", False))

        self.src_max_len = config["src_max_len"]
        self.n_labels = config["n_labels"]
        
        self.use_cuda = config["use_cuda"]
    

    def encode_inputs(self, src_list):
        """
        """        
        src_ids = list(map(self.src_tokenizer.tokenize_to_ids, src_list))
        x = cut_and_pad_seq_list(src_ids,
                                 self.src_max_len, 
                                 self.model.PAD,
                                 True)        
            
        x = torch.tensor(x, dtype=torch.long)

        if self.use_cuda == True:
            x = x.to(self.device)
        
        return x        


    def predict(self, src_list):
        """
        """
        x = self.encode_inputs(src_list)
        
        self.model.eval()
        with torch.no_grad():
            if self.model.use_crf == True:
                labels = crf_model_decoding(self.model, x)
            else:
                labels = self.model.compute_logits([x])
                
        labels = labels.cpu().numpy()

        res = []
        for i,src in enumerate(src_list):
            tokens = self.src_tokenizer.tokenize(src)
            _labels = [self.id2label[label] for label in labels[i]]
            res.append([src_list[i], list(zip(tokens, _labels))])
        
        return res