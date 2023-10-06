# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:34:48 2019

@author: Xiaoyuan Yao
"""
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from mimix.tokenization import build_tokenizer
from mimix.decoding import search
from mimix.decoding import crf_model_decoding
from mimix.utils import load_vocab, real_path, invert_dict, cut_and_pad_seq_list
from mimix.models import build_model


class EncDecGenerator():
    """
    """
    def __init__(self, config):
        """
        """
        self.model = build_model(config, real_path(config["load_model"]))
        
        self.use_cuda = config.get("use_cuda", False)
        self.device = torch.device('cpu')
        if self.use_cuda == True:
            device_id = config.get("device_id", "0")
            self.device = torch.device('cuda:%s' % device_id)
            self.model = self.model.to(self.device)
        
        self.use_vit_encoder = config.get("use_vit_encoder", False)
        if self.use_vit_encoder == False:
            self.src_tokenizer = build_tokenizer(
                    tokenizer=config["src_tokenizer"],
                    vocab_file=config["src_vocab"])
        else:
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize((config["img_w"], config["img_h"])),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5, 0.5),
            ])
        self.trg_tokenizer = build_tokenizer(
                tokenizer=config["trg_tokenizer"],
                vocab_file=config["trg_vocab"])
        
        vocab = load_vocab(real_path(config["trg_vocab"]))
        self.trg_word2id = vocab
        self.trg_id2word = {vocab[word]:word for word in vocab}
        
        if self.use_vit_encoder == False:
            vocab = load_vocab(real_path(config["src_vocab"]))
            self.src_word2id = vocab
            self.src_id2word = {vocab[word]:word for word in vocab}
        
        self.add_tokens = config.get("add_tokens", "")
        self.bos_tok = config["symbols"]["BOS_TOK"]
        self.eos_tok = config["symbols"]["EOS_TOK"]
        self.pad_tok = config["symbols"]["PAD_TOK"]
        
        if self.use_vit_encoder == False:     
            self.src_max_len = config["src_max_len"]
        self.trg_max_len = config["trg_max_len"]
        self.trg_vocab_size = config["trg_vocab_size"]
        self.max_decode_steps = config["max_decode_steps"]
        
        self.beam_size = config.get("beam_size", 3)
        self.group_size = config.get("group_size", 0)
        self.gamma = float(config.get("gamma", 1))
        self.temperature = config.get("temperature", 1.)
        self.strategy = config.get("search_strategy", "beam_search")
        self.top_k = config.get("top_k", 0)
        self.top_p = config.get("top_p", 0.)
        self.repetition_penalty = config.get("repetition_penalty", 0.)
        self.normalize = config.get("normalize", "none")
        self.use_mask_unk =  config.get("use_mask_unk", False)
        self.use_cuda = config["use_cuda"] 
    
    
    def encode_inputs(self, 
                      src_list, 
                      trg_list=None, 
                      add_bos=False, 
                      add_eos=False,
                      pad_trg_left=False):
        """
        """   
        if self.use_vit_encoder == False: 
            src_list = [self.add_tokens + src for src in src_list]
            src_ids = list(map(self.src_tokenizer.tokenize_to_ids, src_list))
            x = cut_and_pad_seq_list(src_ids,
                                     self.src_max_len, 
                                     self.model.PAD,
                                     True)
        else:
            x = [self.transform(img).unsqueeze(0) for img in src_list]
            
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
                                     True,
                                     pad_trg_left)
            
        if self.use_vit_encoder == False:  
            x = torch.tensor(x, dtype=torch.long)
        else:
            x = torch.cat(x, 0).float()
            
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
        x,y = self.encode_inputs(src_list, prefix_list, add_bos=True, pad_trg_left=True)
        
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
                                       repetition_penalty=self.repetition_penalty,
                                       use_mask_unk=self.use_mask_unk,
                                       max_decode_steps=self.max_decode_steps)
                hypothesis,scores = states[4], states[1]
            else:
                raise ValueError("strategy not correct!")
        
        h_len = torch.sum(hypothesis.ne(self.model.PAD), -1).cpu().numpy()
        hypothesis = hypothesis.cpu().numpy()
        scores = scores.cpu().numpy()
        res = []
        for i,src in enumerate(src_list):
            tmp = []
            for j in range(i*self.beam_size, (i+1)*self.beam_size):
                
                trg = self.trg_tokenizer.detokenize_ids(hypothesis[j])
                
                trg = trg.replace(self.bos_tok, "").replace(self.pad_tok, "").strip()
                trg = re.sub(self.eos_tok + ".*", "", trg)
                
                score = float(scores[j])
                
                if self.normalize == "linear":
                    score = score/(h_len[j] - 1)
                tmp.append([trg, score])
            tmp.sort(key=lambda x:x[-1], reverse=True)
            res.append([src, tmp])
        
        return res


    def scoring(self, pairs_list):
        """
        """
        src_list, trg_list, map_ids = [], [], []
        for i,(src,trgs) in enumerate(pairs_list):
            src_list.append(src)
            trg_list.extend(trgs)
            for trg in trgs:
                map_ids.append(i)
        

        x, y = self.encode_inputs(src_list,
                                  trg_list,
                                  add_bos=True,
                                  add_eos=True)
        
        self.model.eval()
        with torch.no_grad():
            
            y_target = y[:, 1:]
            y = y[:, :-1]
            y_len = torch.sum(y.ne(self.model.PAD), -1)
            
            enc_self_attn_mask = self.model.get_attn_mask(x, x)
            dec_self_attn_mask = self.model.get_subsequent_mask(y)
        
            dec_self_attn_mask = dec_self_attn_mask | self.model.get_attn_mask(y, y)
            dec_enc_attn_mask = self.model.get_attn_mask(y, x)
        
            enc_pos_ids = x.ne(self.model.PAD).cumsum(-1) - 1
            dec_pos_ids = y.ne(self.model.PAD).cumsum(-1) - 1
        
            enc_outputs = self.model.encoder(x, 
                                             enc_self_attn_mask,
                                             self_pos_ids=enc_pos_ids)
        
            enc_output = enc_outputs[0]
            
            enc_output = enc_output[map_ids]
            dec_self_attn_mask = self.model.get_subsequent_mask(y)
            dec_self_attn_mask = dec_self_attn_mask | self.model.get_attn_mask(y, y)
            dec_enc_attn_mask = self.model.get_attn_mask(y, x[map_ids])
            
            trg_embedding = None
            if self.model.share_src_trg_emb == True:
                trg_embedding = self.model.encoder.word_embedding
    
            dec_outputs = self.model.decoder(y, 
                                             self_attn_mask=dec_self_attn_mask, 
                                             enc_kv_list=enc_output,
                                             self_enc_attn_mask=dec_enc_attn_mask,
                                             self_pos_ids=dec_pos_ids,
                                             enc_pos_ids=enc_pos_ids,
                                             past_pos_ids=dec_pos_ids,
                                             embedding=trg_embedding)
    
            logits = dec_outputs[0]
            
            log_probs = -F.nll_loss(F.log_softmax(logits.view(-1, self.trg_vocab_size), -1),
                                    y_target.contiguous().view(-1),
                                    ignore_index=self.model.PAD,
                                    reduction='none')
    
            log_probs = torch.sum(log_probs.view(logits.size(0), -1), -1)
            
        log_probs = log_probs.cpu().numpy()
        y_len = y_len.cpu().numpy()
        res = []
        i = 0
        for src,trgs in pairs_list:
            tmp = []
            for trg in trgs:
                score = float(log_probs[i])
                
                if self.normalize == "linear":
                    score /= y_len[i]
                tmp.append([trg, score])
                i += 1
            res.append([src, tmp])
            
        return res


    def get_topk_pred(self, src_list, trg_list, topk=10):
        """
        """
        x, y = self.encode_inputs(src_list, trg_list, add_bos=True, add_eos=True)
    
        self.model.eval()
        with torch.no_grad():
            outputs = self.model([x,y])
                
            logits = outputs[0]
            logits = logits.view(-1, self.trg_vocab_size)
            
            probs = F.softmax(logits, -1)
            probs = probs.view(y.size(0), y.size(1), -1)
            
            top_probs, top_ids = probs.topk(topk, -1)
            
            y_len = torch.sum(y.ne(self.model.PAD), -1)
            
            entropy = Categorical(probs=probs).entropy()
            
            history = torch.gather(probs[:,:-1,:], -1, y[:,1:].unsqueeze(-1))
            
        y = y.cpu().numpy()
        top_ids = top_ids.cpu().numpy()
        top_probs = np.round(top_probs.cpu().numpy(), 3)
        y_len = y_len.cpu().numpy()
        history = np.round(history.squeeze(-1).cpu().numpy(), 3)
        entropy = np.round(entropy.cpu().numpy(), 3)
    
        res = []
        for i,(src,trg) in enumerate(zip(src_list, trg_list)):
            words = [self.trg_id2word.get(w, "_unk_") for w in  y[i][1:]]
            topk_pairs = []
            for j in range(y_len[i].item() - 1):
    
                top_words = [self.trg_id2word.get(w, "_unk_") for w in top_ids[i][j]]
                pairs = tuple(zip(top_words, top_probs[i][j]))
                topk_pairs.append(pairs)
            
            history_probs = list(history[i][:y_len[i]])
            sum_log_probs = sum(np.log(history[i][:y_len[i]]))
            
            res.append([words, topk_pairs, history_probs, sum_log_probs, list(entropy[i])])
            
        return res


class LMGenerator():
    """
    """
    def __init__(self, config):
        """
        """
        self.model = build_model(config, real_path(config["load_model"]))

        self.use_cuda = config.get("use_cuda", False)
        self.device = torch.device("cpu")
        if self.use_cuda == True:
            device_id = config.get("device_id", "0")
            self.device = torch.device('cuda:%s' % device_id)
            self.model = self.model.to(self.device)
        
        self.trg_tokenizer = build_tokenizer(
                tokenizer=config["trg_tokenizer"],
                vocab_file=config["trg_vocab"])
        self.use_cuda = config.get("use_cuda", False)
        self.gamma = float(config.get("gamma", 1))
        self.trg_max_len = config["trg_max_len"]
        self.trg_vocab_size = config["trg_vocab_size"]
        self.max_decode_steps = config["max_decode_steps"]
        self.use_cuda = config["use_cuda"]
        
        self.bos_tok = config["symbols"]["BOS_TOK"]
        self.eos_tok = config["symbols"]["EOS_TOK"]
        self.pad_tok = config["symbols"]["PAD_TOK"]
        
        self.beam_size = config.get("beam_size", 3)
        self.group_size = config.get("group_size", 0)
        self.temperature = config.get("temperature", 1.)
        self.strategy = config.get("search_strategy", "beam_search")
        self.top_k = config.get("top_k", 0)
        self.top_p = config.get("top_p", 0.)
        self.repetition_penalty = config.get("repetition_penalty", 0.)
        self.normalize = config.get("normalize", "none")
        self.use_mask_unk =  config.get("use_mask_unk", False)
        self.use_cuda = config["use_cuda"] 
        

    def encode_inputs(self, trg_list, add_bos=False, add_eos=False, pad_trg_left=False):
        """
        """
        trg_ids = list(map(self.trg_tokenizer.tokenize_to_ids, trg_list))
        y = cut_and_pad_seq_list(trg_ids,
                                 self.trg_max_len, 
                                 self.model.PAD,
                                 True,
                                 pad_trg_left)
        if add_bos == True:
            y = [[self.model.BOS] + seq for seq in y]
        if add_eos == True:
            y = [seq + [self.model.EOS] for seq in y]
                
        if y is not None:
            y = torch.tensor(y, dtype=torch.long)

        if self.use_cuda == True:
            if y is not None:
                y = y.to(self.device)
        
        return y

    
    def predict(self, prefix_list, pad_trg_left=True):
        """
        """
        batch_size = 1
        y = None
        if prefix_list is not None:
            prefix_list = [prefix_list[i] for i in range(len(prefix_list))]
            y = self.encode_inputs(prefix_list, add_bos=True, pad_trg_left=True)
            batch_size = len(prefix_list)
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
                                       repetition_penalty=self.repetition_penalty,
                                       use_mask_unk=self.use_mask_unk,
                                       max_decode_steps=self.max_decode_steps)
                hypothesis,scores = states[4], states[1]
            else:
                raise ValueError("strategy not correct!")
        
        h_len = torch.sum(hypothesis.ne(self.model.PAD), -1).cpu().numpy()
        hypothesis = hypothesis.cpu().numpy()
        scores = scores.cpu().numpy()

        res = []
        for i in range(batch_size):
            tmp = []
            for j in range(i*self.beam_size, (i+1)*self.beam_size):
                
                trg = self.trg_tokenizer.detokenize_ids(hypothesis[j])
                
                trg = trg.replace(self.bos_tok, "").replace(self.pad_tok, "").strip()
                trg = re.sub(self.eos_tok + ".*", "", trg)
                score = float(scores[j])
                
                if self.normalize == "linear":
                    score = score/(h_len[j] - 1)
                tmp.append([trg, score])
            tmp.sort(key=lambda x:x[-1], reverse=True)
            res.append(["" if prefix_list is None else prefix_list[i], tmp])
        
        return res


    def scoring(self, trg_list):
        """
        """
        y = self.encode_inputs(trg_list,
                               add_bos=True,
                               add_eos=True)
        self.model.eval()
        with torch.no_grad():
            y_target = y[:, 1:]
            y = y[:, :-1]
            y_len = torch.sum(y.ne(self.model.PAD), -1)
            
            dec_self_attn_mask = self.model.get_subsequent_mask(y)
            dec_self_attn_mask = dec_self_attn_mask | self.model.get_attn_mask(y, y)
    
            self_pos_ids = y.ne(self.model.PAD).cumsum(-1) - 1
            dec_outputs = self.model.decoder(y, 
                                             self_attn_mask=dec_self_attn_mask,
                                             self_pos_ids=self_pos_ids,
                                             past_pos_ids=self_pos_ids)
    
            logits = dec_outputs[0]
            log_probs = -F.nll_loss(F.log_softmax(logits.view(-1, self.trg_vocab_size), -1),
                                    y_target.contiguous().view(-1),
                                    ignore_index=self.model.PAD,
                                    reduction='none')
    
            log_probs = torch.sum(log_probs.view(logits.size(0), -1), -1)
            
        log_probs = log_probs.cpu().numpy()
        y_len = y_len.cpu().numpy()
        res = []
        i = 0
        for trg in trg_list:
            score = float(log_probs[i])
            
            if self.normalize == "linear":
                score /= y_len[i]
            res.append([trg, score])
            i += 1
            
        return res        


class TextEncoder():
    """
    """
    def __init__(self, config):
        """
        """
        self.model = build_model(config, real_path(config["load_model"]))
        
        self.use_cuda = config.get("use_cuda", False)
        self.device = torch.device('cpu')
        if self.use_cuda == True:
            device_id = config.get("device_id", "0")
            self.device = torch.device('cuda:%s' % device_id)
            self.model = self.model.to(self.device)
        
        self.src_word2id = load_vocab(real_path(config["src_vocab"]))
        self.src_id2word = invert_dict(self.src_word2id)
        
        self.src_tokenizer = build_tokenizer(
                tokenizer=config["src_tokenizer"],
                vocab_file=config["src_vocab"])
        self.add_tokens = config.get("add_tokens", "")
        
        self.mask_id = config["symbol2id"][config["symbols"]["MASK_TOK"]]
        
        self.src_vocab_size = config["src_vocab_size"]
        self.use_cuda = config["use_cuda"]
        self.src_max_len = config["src_max_len"]

        self.label2id = None
        self.id2label = None
        if "label2id" in config:
            self.label2id = load_vocab(real_path(config["label2id"]))
            self.id2label = invert_dict(self.label2id)
        self.num_class = config.get("n_class", None)
    
    
    def encode_inputs(self, src_list):
        """
        """  
        src_list = [self.add_tokens + src for src in src_list]
        src_ids = list(map(self.src_tokenizer.tokenize_to_ids, src_list))

        y = cut_and_pad_seq_list(src_ids,
                                 self.src_max_len, 
                                 self.model.PAD,
                                 True)

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


    def predict_mlm(self, src_list, top_k=-1, top_p=-1, return_k=10):
        """
        """
        y = self.encode_inputs(src_list)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model([y])
            logits = outputs[0]
            probs = torch.softmax(logits, -1)
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
                        pair = [self.src_id2word[indice[i][j][k]], score[i][j][k]]
                        pred.append(pair)

                    res[-1][1].append(pred)
        
        return res


    def predict_sim(self, src_list):
        """
        """
        vec = self.encode_texts(src_list)
        norm_vec = F.normalize(vec, p=2, dim=1)
        sim = torch.mm(norm_vec, norm_vec.T)
        sim = sim.cpu().numpy()
        return sim


    def predict_cls(self, src_list):
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


    def predict_seq(self, src_list):
        """
        """
        x = self.encode_inputs(src_list)
        
        self.model.eval()
        with torch.no_grad():
            if self.model.crf is not None:
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


    def dump_encode_text(self, fi_path, fo_path):
        """
        """
        fo = open(fo_path, "w", encoding="utf-8")
        
        def process_batch(cache):
            """
            """
            texts = [d["text"] for d in cache]
            vecs = self.encode_texts(texts).cpu().numpy().tolist()
            for data,vec in zip(cache, vecs):
                data["vec"] = vec
                fo.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        cache = []
        for line in open(fi_path, "r", encoding="utf-8"):
            data = json.loads(line)
            cache.append(data)
            if len(cache) >= config["test_batch_size"]:
                process_batch(cache)
                cache = []
        if len(cache) > 0:
            process_batch(cache)
    
        fo.close()


class ImageEncoder():
    """
    """
    def __init__(self, config):
        """
        """
        self.model = build_model(config, real_path(config["load_model"]))

        self.use_cuda = config.get("use_cuda", False)
        self.device = torch.device('cpu')
        if self.use_cuda == True:
            device_id = config.get("device_id", "0")
            self.device = torch.device('cuda:%s' % device_id)
            self.model = self.model.to(self.device)

        self.label2id = None
        self.id2label = None
        if "label2id" in config:
            self.label2id = load_vocab(real_path(config["label2id"]))
            self.id2label = invert_dict(self.label2id)
        self.num_class = config.get("n_class", None)

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((config["img_w"], config["img_h"])),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5, 0.5),
            ])


    def transform_image(self, images):
        """
        """
        x = []
        for image in images:
            x.append(self.transform(image).unsqueeze(0))
        x = torch.cat(x, 0).to(self.device)
        
        return x


    def predict_cls(self, images, return_topk=5):
        """
        """
        x = self.transform_image(images)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model([x])

        logits = outputs[0]

        y = torch.softmax(logits, 1)
        prob,label = torch.topk(y, self.num_class)
        prob = prob.cpu().numpy()
        label = label.cpu().numpy()

        res = []
        for i,image in enumerate(images):
            res.append([image, []])
            for yy,ss in zip(prob[i,:], label[i,:]):
                label_str = str(ss)
                if self.id2label is not None:
                    label_str = self.id2label[ss]

                res[i][1].append([label_str, yy])
                if len(res[i][1]) >= return_topk:
                    break
        return res


class ClipMatcher():
    """
    """
    def __init__(self, config):
        """
        """
        self.model = build_model(config, real_path(config["load_model"]))
        self.sim_alpha = config.get("sim_alpha", 20)
        self.use_cuda = config.get("use_cuda", False)
        self.device = torch.device('cpu')
        if self.use_cuda == True:
            device_id = config.get("device_id", "0")
            self.device = torch.device('cuda:%s' % device_id)
            self.model = self.model.to(self.device)

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((config["image_img_w"], config["image_img_h"])),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5, 0.5),
            ])


        self.src_tokenizer = build_tokenizer(
                tokenizer=config["src_tokenizer"],
                vocab_file=config["src_vocab"])
        self.add_tokens = config.get("add_tokens", "")
        
        self.use_cuda = config["use_cuda"]
        self.src_max_len = config["text_src_max_len"]

    
    def encode_inputs(self, src_list):
        """
        """      
        src_list = [self.add_tokens + src for src in src_list]
        
        src_ids = list(map(self.src_tokenizer.tokenize_to_ids, src_list))

        y = cut_and_pad_seq_list(src_ids,
                                 self.src_max_len, 
                                 self.model.PAD,
                                 True)

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
            outputs = self.model.text_encoder([y])
        
        return outputs[0]


    def transform_image(self, images):
        """
        """
        x = []
        for image in images:
            x.append(self.transform(image).unsqueeze(0))
        x = torch.cat(x, 0).to(self.device)
        
        return x


    def encode_images(self, images):
        """
        """
        x = self.transform_image(images)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.img_encoder([x])
        
        return outputs[0]
    

    def predict_sim(self, images, texts):
        """
        """
        img_vec = self.encode_images(images)
        text_vec = self.encode_texts(texts)
 
        with torch.no_grad():
            img_vec = F.normalize(img_vec, p=2, dim=1)
            text_vec = F.normalize(text_vec, p=2, dim=1)
            sim = torch.mm(img_vec, text_vec.T)
            prob_text = torch.softmax(self.sim_alpha*sim, 1)
            prob_image = torch.softmax(self.sim_alpha*sim, 0)
        sim = sim.cpu().numpy()
        prob_text = prob_text.cpu().numpy()
        prob_image = prob_image.cpu().numpy()
        
        return sim,prob_text,prob_image


    def predict_images_sim(self, images):
        """
        """
        img_vec = self.encode_images(images)
 
        with torch.no_grad():
            img_vec = F.normalize(img_vec, p=2, dim=1)
            sim = torch.mm(img_vec, img_vec.T)
            prob = torch.softmax(self.sim_alpha*sim, 1)
        sim = sim.cpu().numpy()
        prob = prob.cpu().numpy()
        
        return sim,prob


    def predict_texts_sim(self, texts):
        """
        """
        text_vec = self.encode_texts(texts)
 
        with torch.no_grad():
            text_vec = F.normalize(text_vec, p=2, dim=1)
            sim = torch.mm(text_vec, text_vec.T)
            prob = torch.softmax(self.sim_alpha*sim, 1)
        sim = sim.cpu().numpy()
        prob = prob.cpu().numpy()
        
        return sim,prob