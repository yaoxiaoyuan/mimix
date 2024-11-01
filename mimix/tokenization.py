# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 17:40:07 2020

@author: Xiaoyuan Yao
"""
import re
from abc import abstractmethod
from mimix.utils import load_vocab

def is_alphabet(ch):
    """
    """
    code = ord(ch)
    return 0x3041 <= code <= 0x3093 or \
        0x30a1 <= code <= 0x30f3 or \
        0xac00 <= code <= 0xd7af or \
        0x1100 <= code <= 0x11ff or \
        0x3130 <= code <= 0x318f or \
        0xa960 <= code <= 0xa97f or \
        0xd7b0 <= code <= 0xd7ff or \
        0x61 <= code <= 0x7a or \
        0x41 <= code <= 0x5a or \
        0x430 <= code <= 0x44f or \
        0x410 <= code <= 0x42f or \
        0x3b1 <= code <= 0x3c9 or \
        0x391 <= code <= 0x3a9 or \
        0xc0 <= code <= 0xff


def is_cjk(ch):
    """
    """
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or \
        0x3400 <= code <= 0x4DBF or \
        0x20000 <= code <= 0x2A6DF or \
        0x2A700 <= code <= 0x2B73F or \
        0x2B740 <= code <= 0x2B81F or \
        0x2B820 <= code <= 0x2CEAF or \
        0xF900 <= code <= 0xFAFF or \
        0x2F800 <= code <= 0x2FA1F
    
    
def is_num(ch):
    """
    """
    code = ord(ch)
    return 0x30 <= code <= 0x39


def is_special(ch):
    """
    """
    return ch == "_"


def is_useless(ch):
    """
    """
    code = ord(ch)
    return code == 0xfffd or \
        0x80 <= code <= 0x9F or \
        0x00 <= code <= 0x1F
        
        
def is_space(ch):
    """
    """
    return ch == " " or ch == "　"


def is_newline(ch):
    """
    """
    return ch == "\n"


class Tokenizer():
    """
    """
    def __init__(self, vocab_file):
        """
        """
        self.vocab = None
        self.id2word = None
        
        self.vocab = load_vocab(vocab_file)
        self.id2word = {self.vocab[w]:w for w in self.vocab}
    
    
    @abstractmethod
    def tokenize(self, text):
        """
        """
        pass
    
    
    @abstractmethod
    def detokenize(self, tokens, convert_special_token=True):
        """
        """
        pass


    def convert_tokens_to_ids(self, tokenized):
        """
        """
        unk = self.vocab["_unk_"]
        word_ids = [self.vocab.get(word, unk) for word in tokenized]
        return word_ids


    def convert_ids_to_tokens(self, word_ids):
        """
        """
        tokens = [self.id2word.get(word, "_unk_") for word in word_ids]
        return tokens


    def tokenize_to_ids(self, text):
        """
        """
        tokens = self.tokenize(text)
        word_ids = self.convert_tokens_to_ids(tokens)
        
        return word_ids
    
    
    def detokenize_ids(self, word_ids, convert_special_token=True):
        """
        """
        tokens = self.convert_ids_to_tokens(word_ids)
        text = self.detokenize(tokens, convert_special_token=convert_special_token)
                            
        return text


class SpaceTokenizer(Tokenizer):
    """
    """
    def __init__(self, vocab_file):
        """
        """
        super(SpaceTokenizer,self).__init__(vocab_file)
    
    
    def tokenize(self, text):
        """
        """
        tokens = text.split()
        
        return tokens
    
    
    def detokenize(self, tokens, convert_special_token=True):
        """
        """
        text = " ".join(tokens)
        
        return text
    

class MimixTokenizer(Tokenizer):
    """
    """
    def __init__(self, vocab_file, uncased=True, match_special_symbols=True):
        """
        """
        super(MimixTokenizer,self).__init__(vocab_file)
        
        zh_words = [ww for ww in self.vocab if all([is_cjk(ch) for ch in ww])]
        self.tri_tree = self.build_tri_tree(zh_words)
        
        self.space_token = "_mimixsp_"
        self.newline_token = "_mimixnl_"
        self.pad_token = "_pad_"
        self.bos_token = "_bos_"
        self.eos_token = "_eos_"
        self.unk_token = "_unk_"
        self.sep_token = "_sep_"
        self.mask_token = "_mask_"
        self.cls_token = "_cls_"
        
        self.special_symbols = [self.pad_token, 
                                self.bos_token, 
                                self.eos_token, 
                                self.unk_token]
        
        self.match_special_symbols = match_special_symbols
        self.symbols = set()
        for word in self.vocab:
            if re.search("^_unused[0-9]+_$", word):
                continue
            if word in self.special_symbols:
                continue
            if match_special_symbols == False:
                continue
            if re.search("^_[0-9a-z]+_$", word):
                self.symbols.add(word)

        self.symbols_tri_tree = self.build_tri_tree(self.symbols)
        
        self.uncased = uncased


    def build_tri_tree(self, keywords):
        """   
        """    
        tri_tree = {}
        for key in keywords:
            root = tri_tree
            for ch in key:
                if ch not in root:
                    root[ch] = {}
                root = root[ch]                                
                
            root.setdefault(u"##", {})
        
        return tri_tree
    

    def prefix_match(self, s):
        """
        """
        start = 0 
        size = len(s)

        root = self.symbols_tri_tree
        end = start
        matched = ""
        matched_end = start
        while end < size and s[end] in root:
            if u"##" in root:
                matched = s[start:end]
                matched_end = end

            root = root[s[end]]
            end += 1

        if u"##" in root:
            matched = s[start:end]
            matched_end = end

        if matched_end == start:
            return ""
        
        return matched
    

    def maximum_match(self, s):
        """
        """
        
        tokenized = []
        
        start = 0
        size = len(s)
        
        while start < size:
            root = self.tri_tree
            end = start
            matched = ""
            matched_end = start
            while end < size and s[end] in root: 
                if u"##" in root: 
                    matched = s[start:end]
                    matched_end = end
    
                root = root[s[end]]
                end += 1
                
            if u"##" in root: 
                matched = s[start:end]
                matched_end = end
            
            if matched_end == start:
                matched = s[start:start + 1]
                matched_end = start + 1
            
            tokenized.append(matched)
            
            start = matched_end
        
        return tokenized

    
    def tokenize(self, text):
        """
        """
        if self.uncased == True:
            text = text.lower()
        
        i = 0
        tokenized = ""
        is_last_cjk = False
        is_last_num_or_alphabet = False
        while i < len(text):
            ch = text[i]
            if is_special(ch):
                if self.match_special_symbols == False:
                    tokenized += (" " + ch + " ")
                    i += 1
                else:
                    matched = self.prefix_match(text[i:]) 
                    if len(matched) > 0:
                        tokenized += (" " + matched + " ")
                        i += len(matched)
                    else:
                        tokenized += (" " + ch + " ")
                        i += 1
                is_last_cjk = False
                is_last_num_or_alphabet = False
            elif is_cjk(ch):
                if is_last_cjk == True:
                    tokenized += (ch) 
                else:
                    tokenized += (" " + ch)
                is_last_cjk = True
                is_last_num_or_alphabet = False
                i += 1
            elif is_num(ch) or is_alphabet(ch):                
                if is_last_num_or_alphabet == True:
                    tokenized += ch
                else:
                    tokenized += (" " + ch)
                is_last_cjk = False
                is_last_num_or_alphabet = True
                i += 1
            elif is_space(ch):
                
                if i == 0 or i == len(text) - 1:
                    tokenized += (" " + self.space_token + " ")
                elif is_alphabet(text[i-1]) and is_alphabet(text[i+1]): 
                    tokenized += " "
                else:
                    ignore = False
                    if self.match_special_symbols == True:
                        if re.search("_[0-9a-z]+_$", text[:i]): 
                            ignore = True
                        if re.search(" _[0-9a-z]+_", text[i:]):
                            ignore = True
                    if ignore == True:
                        tokenized += " "
                    else:
                        tokenized += (" " + self.space_token + " ")
                            
                is_last_cjk = False
                is_last_num_or_alphabet = False   
                i += 1
            elif is_newline(ch):
                tokenized += (" " + self.newline_token + " ")
                is_last_cjk = False
                is_last_num_or_alphabet = False 
                i += 1
            elif is_useless(ch):
                is_last_cjk = False
                is_last_num_or_alphabet = False  
                i += 1
            else:
                tokenized += (" " + ch + " ") 
                is_last_cjk = False
                is_last_num_or_alphabet = False
                i += 1
                
        tokens = []
        for token in tokenized.split():
            if len(token) == 0:
                continue
            elif re.search("^_[0-9a-z]+_$", token):
                tokens.append(token)
            elif all(is_cjk(ch) for ch in token):
                tokens.extend(self.maximum_match(token))    
            else:
                tokens.extend(self.wordpiece(token))
        
        return tokens
    

    def wordpiece(self, word):
        """
        """
        if word in self.vocab:
            return [word]

        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self.vocab:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub)
            start = stop

        return tokens


    def detokenize(self, tokens, convert_special_token=True):
        """
        """
        text = ""
        
        is_last_alphabet = False
        for token in tokens:
            if all(is_cjk(ch) for ch in token):
                text += token
                is_last_alphabet = False
            elif token.startswith("##"):
                text += token[2:]
                is_last_alphabet = True
            elif re.search("^_[0-9a-z]+_$", token):
                if convert_special_token == False:
                    text = text + (" " + token + " ")
                else:
                    if token == self.space_token:
                        text += " "
                    elif token == self.newline_token:
                        text += "\n"
                    else:
                        if text.endswith(" ") == False:
                            text += " "
                        text += (token + " ")
                    
                is_last_alphabet = False
            else:
                is_cur_alphabet = False
                if all(is_alphabet(ch) or is_num(ch) for ch in token):
                    is_cur_alphabet = True
                if is_last_alphabet == True and is_cur_alphabet == True:
                    text += (" " + token)
                else:
                    text += token
                is_last_alphabet = is_cur_alphabet
        
        return text


class BertTokenizer(Tokenizer):
    """
    """
    def __init__(self, vocab_file, uncased=True):
        """
        """
        super(BertTokenizer,self).__init__(vocab_file)
        from mimix.bert_tokenizer import FullTokenizer
        self.tokenizer = FullTokenizer(vocab_file, do_lower_case=uncased)
        
        
    def tokenize(self, text):
        """
        """
        return self.tokenizer.tokenize(text)
    
    
    def detokenize(self, tokens, convert_special_token=True):
        """
        """
        return " ".join(tokens)
    

    def convert_tokens_to_ids(self, tokenized):
        """
        """
        return self.tokenizer.convert_tokens_to_ids(tokenized)


    def convert_ids_to_tokens(self, word_ids):
        """
        """
        return self.tokenizer.convert_ids_to_tokens(word_ids)


def build_tokenizer(**args):
    """
    """
    if args["tokenizer"] == "default":
        tokenizer = SpaceTokenizer(vocab_file=args["vocab_file"])
    elif args["tokenizer"] == "mimix":
        tokenizer = MimixTokenizer(vocab_file=args["vocab_file"])
    elif args["tokenizer"] == "mimix-cased":
        tokenizer = MimixTokenizer(vocab_file=args["vocab_file"], uncased=False)
    elif args["tokenizer"] == "bert":
        tokenizer = BertTokenizer(vocab_file=args["vocab_file"], uncased=True)
    elif args["tokenizer"] == "bert-cased":
        tokenizer = BertTokenizer(vocab_file=args["vocab_file"], uncased=False)
    else:
        raise ValueError("tokenizer not correct!")
        
    return tokenizer

