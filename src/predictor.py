# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:34:48 2019

@author: Xiaoyuan Yao
"""
import random
import torch
import torch.nn.functional as F
from tokenization import build_tokenizer
from decoding import beam_search, sample
from decoding import beam_search_with_constraints, sample_with_constraints
from decoding import lm_sample, lm_sample_with_constraints
from decoding import top_k_top_p_sampling
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
        
        self.beam_size = config["beam_size"]
        self.group_size = config.get("group_size", 1)
        self.diverse_rate = config.get("diverse_rate", 0)
        self.gamma = float(config["gamma"])
        self.strategy = config.get("search_strategy", "beam_search")
        
        self.sample_size = config.get("sample_size", 1)
        self.sample_top_k = config.get("sample_top_k", -1)
        self.sample_top_k0 = config.get("sample_top_k0", -1)
        self.sample_top_p = config.get("sample_top_p", -1)
        self.sample_top_p0 = config.get("sample_top_p0", -1)
        self.sample_temp = config.get("sample_temp", 1)
        self.sample_alpha_0 = config.get("sample_alpha_0", self.sample_temp)
        self.sample_alpha = config.get("sample_alpha", self.sample_temp)
        self.sample_beta = config.get("sample_beta", 0)

        self.top_group = config.get("top_group", -1)
        self.alpha_0 = config.get("alpha_0", 1)
        self.alpha = config.get("alpha", 1)
        self.beta = config.get("beta", 0)
        
        self.combine_search_steps = config.get("combine_search_steps", 1)
        self.combine_search_eos = config.get("combine_search_eos", None)
        
        self.history_penalty = config.get("history_penalty", 0)
        self.history_penalty_beta = config.get("history_penalty_beta", 0)
        self.repeat_penalty = config.get("repeat_penalty", 0)
        self.return_sample_k = config.get("return_sample_k", -1)
        self.normalize = config.get("normalize", "none")
        self.penalty_vocab_start = config.get("penalty_vocab_start", -1)
        self.penalty_vocab_end = config.get("penalty_vocab_end", -1) 
        self.need_mask_unk =  config.get("need_mask_unk", False)
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
            if "combine" in self.strategy:
                beam_size = self.beam_size * self.sample_size
                outputs = sample_with_constraints(
                        self.model, 
                        x, 
                        self.sample_size,
                        self.combine_search_steps,
                        repeat_penalty=self.repeat_penalty,
                        history_penalty=self.history_penalty,
                        history_penalty_beta=self.history_penalty_beta,
                        penalty_vocab_start=self.penalty_vocab_start,
                        penalty_vocab_end=self.penalty_vocab_end,
                        need_mask_unk=False,
                        sample_alpha_0=self.sample_alpha_0,
                        sample_alpha=self.sample_alpha,
                        sample_beta=self.sample_beta,
                        sample_top_k=self.sample_top_k,
                        sample_top_p=self.sample_top_p,
                        sample_top_k0=self.sample_top_k0,
                        sample_top_p0=self.sample_top_p0,
                        normalize="none",
                        prefix_y=y,
                        eos=self.combine_search_eos,
                        return_states=True,
                        early_stop=True)
                
                init_states = outputs[2:-2] + outputs[0:1] + outputs[-2:]
                init_states[4].fill_(0)
                
                hypothesis,scores = \
                    beam_search_with_constraints(
                            self.model, 
                            x, 
                            self.beam_size, 
                            self.strategy,
                            self.max_dec_steps,
                            group_size=self.group_size, 
                            top_k=self.top_group,
                            diverse_rate=self.diverse_rate,
                            history_penalty=self.history_penalty,
                            history_penalty_beta=self.history_penalty_beta,
                            repeat_penalty=self.repeat_penalty,
                            penalty_vocab_start=self.penalty_vocab_start,
                            penalty_vocab_end=self.penalty_vocab_end,
                            alpha_0=self.alpha_0,
                            alpha=self.alpha,
                            beta=self.beta,
                            prefix_y=y,
                            normalize=self.normalize,
                            gamma=self.gamma,
                            init_states=init_states)

            elif "beam_search" in self.strategy:
                beam_size = self.beam_size
                if self.strategy == "beam_search":
                    hypothesis,scores = \
                    beam_search(self.model,
                                x, 
                                self.beam_size, 
                                self.max_dec_steps,
                                normalize=self.normalize,
                                gamma=self.gamma)
                else: 
                    hypothesis,scores = \
                    beam_search_with_constraints(
                            self.model, 
                            x, 
                            self.beam_size, 
                            self.strategy,
                            self.max_dec_steps,
                            group_size=self.group_size, 
                            top_k=self.top_group,
                            diverse_rate=self.diverse_rate,
                            history_penalty=self.history_penalty,
                            history_penalty_beta=self.history_penalty_beta,
                            repeat_penalty=self.repeat_penalty,
                            penalty_vocab_start=self.penalty_vocab_start,
                            penalty_vocab_end=self.penalty_vocab_end,
                            alpha_0=self.alpha_0,
                            alpha=self.alpha,
                            beta=self.beta,
                            prefix_y=y,
                            normalize=self.normalize,
                            need_mask_unk=self.need_mask_unk,
                            gamma=self.gamma)
                   
            elif "sample" in self.strategy:
                beam_size = self.sample_size
                if self.strategy == "sample":
                    hypothesis,scores = \
                    sample(self.model, 
                           x, 
                           self.sample_size,
                           self.max_dec_steps, 
                           sample_temp=self.sample_temp,
                           normalize=self.normalize,
                           gamma=self.gamma)
                else:
                    hypothesis,scores = \
                    sample_with_constraints(
                            self.model, 
                            x, 
                            self.sample_size,
                            self.max_dec_steps,
                            repeat_penalty=self.repeat_penalty,
                            history_penalty=self.history_penalty,
                            history_penalty_beta=self.history_penalty_beta,
                            penalty_vocab_start=self.penalty_vocab_start,
                            penalty_vocab_end=self.penalty_vocab_end,
                            need_mask_unk=False,
                            sample_alpha_0=self.sample_alpha_0,
                            sample_alpha=self.sample_alpha,
                            sample_beta=self.sample_beta,
                            sample_top_k=self.sample_top_k,
                            sample_top_p=self.sample_top_p,
                            sample_top_k0=self.sample_top_k0,
                            sample_top_p0=self.sample_top_p0,
                            normalize=self.normalize,
                            prefix_y=y,
                            eos=None)
            else:
                raise ValueError("strategy not correct!")
        
        hypothesis = hypothesis.cpu().numpy()
        scores = scores.cpu().numpy()
            
        return_sample_k = min(beam_size, self.return_sample_k)
        res = []
        for i,src in enumerate(src_list):
            tmp = []
            for j in range(i*beam_size, (i+1)*beam_size):
                trg = self.trg_tokenizer.detokenize_ids(hypothesis[j])
                
                trg = trg.replace(self.pad_tok, "").strip()
                if trg.endswith(self.eos_tok):
                    trg = trg.replace(self.eos_tok, "").strip()
                else:
                    trg = trg + " _unfinished_"
                
                tmp.append([trg, float(scores[j])])
            
            if return_sample_k > 0:
                tmp = random.sample(y[:return_sample_k], 1)
                
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
        
        self.alpha_0 = config.get("alpha_0", 1)
        self.alpha = config.get("alpha", 1)
        self.beta = config.get("beta", 0)
        
        self.temp = config.get("temp", 1)
        
        self.top_k = config.get("top_k", -1)
        self.top_k0 = config.get("top_k0", -1)
        self.top_p = config.get("top_p", -1)
        self.top_p0 = config.get("top_p0", -1)
        
        self.history_penalty = config.get("history_penalty", 0)
        self.history_penalty_beta = config.get("history_penalty_beta", 0) 

        self.penalty_vocab_start = config.get("penalty_vocab_start", -1)
        self.penalty_vocab_end = config.get("penalty_vocab_end", -1)
                                          
        self.repeat_penalty = config.get("repeat_penalty", 0)
        self.normalize = config.get("normalize", "none")
        
        self.need_mask_unk =  config.get("need_mask_unk", False)
        self.strategy = config.get("search_strategy", "sample")
        self.sample_size = config.get("sample_size", 1)
        

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
        prefix = None
        if "prefix" in self.strategy and prefix_list is not None:
            prefix_list = [prefix_list[i//self.sample_size] for i in range(len(prefix_list) * self.sample_size)]
            prefix = self.encode_inputs(prefix_list)

        self.model.eval()
        with torch.no_grad():
            if self.strategy == "sample":
                hypothesis,scores = \
                    lm_sample(self.model, 
                              self.max_dec_steps, 
                              self.use_cuda,
                              self.device,
                              self.sample_size,
                              normalize=self.normalize,
                              gamma=self.gamma,
                              temp=self.temp)
            else:
                batch_size = self.sample_size
                if prefix is not None and len(prefix_list) > 0:
                    batch_size = len(prefix_list)
                hypothesis,scores = \
                    lm_sample_with_constraints(                            
                            self.model, 
                            self.max_dec_steps, 
                            self.use_cuda,
                            self.device,
                            batch_size=batch_size,
                            alpha_0=self.alpha_0,
                            alpha=self.alpha,
                            beta=self.beta,
                            history_penalty=self.history_penalty,
                            history_penalty_beta=self.history_penalty_beta,
                            repeat_penalty=self.repeat_penalty,
                            penalty_vocab_start=self.penalty_vocab_start,
                            penalty_vocab_end=self.penalty_vocab_end,
                            prefix=prefix,
                            gamma=1,
                            normalize="none",
                            top_k=self.top_k,
                            top_k0=self.top_k0,
                            top_p=self.top_p,
                            top_p0=self.top_p0,
                            eos=None,
                            need_mask_unk=self.need_mask_unk,
                            return_states=False)
                    
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
        
        self.trg_word2id = load_vocab(real_path(config["trg_vocab"]))
        self.trg_id2word = invert_dict(self.trg_word2id)
        
        self.trg_tokenizer = build_tokenizer(
                tokenizer=config["trg_tokenizer"],
                vocab_file=config["trg_vocab"], 
                pre_tokenized=config.get("pre_tokenized", False),  
                pre_vectorized=config.get("pre_vectorized", False))
        self.add_cls = config.get("add_cls", False)
        
        self.mask_id = config["symbol2id"][config["symbols"]["MASK_TOK"]]
        
        self.trg_vocab_size = config["trg_vocab_size"]
        self.use_cuda = config["use_cuda"]
        self.trg_max_len = config["trg_max_len"]
    
    
    def encode_inputs(self, trg_list):
        """
        """        
        trg_ids = list(map(self.trg_tokenizer.tokenize_to_ids, trg_list))

        y = cut_and_pad_seq_list(trg_ids,
                                 self.trg_max_len, 
                                 self.model.PAD,
                                 True)
        if self.add_cls == True:
            y = [[self.model.CLS] + yy[:self.trg_max_len-1] for yy in y]

        if y is not None:
            y = torch.tensor(y, dtype=torch.long)

        if self.use_cuda == True:
            if y is not None:
                y = y.to(self.device)
        
        return y


    def predict(self, trg_list, top_k=-1, top_p=-1, return_k=10):
        """
        """
        y = self.encode_inputs(trg_list)

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
        for i,trg in enumerate(y):
            res.append([trg_list[i], []])
            for j,ww in enumerate(trg):

                if ww == self.mask_id:
                    pred = []
                    for k in range(return_k):
                        pred.append([self.trg_id2word[indice[i][j][k]], score[i][j][k]])

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
        
        return outputs[1]


    def predict(self, src_list):
        """
        """
        y = self.encode_inputs(src_list)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model([y])
            sim = outputs[0]
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