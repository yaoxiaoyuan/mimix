# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 17:40:07 2020

@author: lyn
"""
import re
from abc import abstractmethod
from utils import load_vocab

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
    return ch == " "


class Tokenizer():
    """
    """
    def __init__(self, vocab, pre_tokenized, pre_vectorized):
        """
        """
        self.vocab = None
        self.id2word = None
        
        if pre_vectorized == False:
            self.vocab = load_vocab(vocab)
            self.id2word = {self.vocab[w]:w for w in self.vocab}
            
        self.pre_tokenized = pre_tokenized
        self.pre_vectorized = pre_vectorized
    
    
    @abstractmethod
    def tokenize_str2list(self, text):
        """
        """
        pass
    
    
    @abstractmethod
    def detokenize_tokens(self, tokens):
        """
        """
        pass


    def convert_word2id(self, tokenized):
        """
        """
        unk = self.vocab["_unk_"]
        word_ids = [self.vocab.get(word, unk) for word in tokenized]
        return word_ids


    def convert_id2word(self, word_ids):
        """
        """
        tokens = [self.id2word.get(word, "_unk_") for word in word_ids]
        return tokens


    def tokenize(self, text):
        """
        """
        if self.pre_vectorized == False:
            if self.pre_tokenized == False:
                tokens = self.tokenize_str2list(text)
            else:
                tokens = text.split()
            word_ids = self.convert_word2id(tokens)
        else:
            tokens = text.split()
            word_ids = [int(word) for word in tokens]
        
        return word_ids
    
    
    def detokenize(self, word_ids):
        """
        """
        if self.pre_vectorized == False:
            tokens = self.convert_id2word(word_ids)
            if self.pre_tokenized == False:
                text = self.detokenize_tokens(tokens)
            else:
                text = " ".join(tokens)
        else:
            text = " ".join([str(word) for word in word_ids])
        
        return text


class SpaceTokenizer(Tokenizer):
    """
    """
    def __init__(self, vocab, pre_tokenized, pre_vectorized):
        """
        """
        super(SpaceTokenizer,self).__init__(
                vocab, pre_tokenized, pre_vectorized)
    
    
    def tokenize_str2list(self, text):
        """
        """
        tokens = text.split()
        
        return tokens
    
    
    def detokenize_tokens(self, tokens):
        """
        """
        text = " ".join(tokens)
        
        return text
    

class WordPieceTokenizer(Tokenizer):
    """
    """
    def __init__(self, vocab, pre_tokenized, pre_vectorized):
        """
        """
        super(WordPieceTokenizer,self).__init__(
                vocab, pre_tokenized, pre_vectorized)
        
        if pre_vectorized == False:
            self.tri_tree = self.build_tri_tree(self.vocab)
        

    def build_tri_tree(self, keywords):
        """   
        """    
        tri_tree = {}
        for key in keywords:
            if any(not is_cjk(ch) for ch in key):
                continue
            root = tri_tree
            for ch in key:
                if ch not in root:
                    root[ch] = {}
                root = root[ch]                                
                
            root.setdefault(u"##", {})
        
        return tri_tree
    

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

    
    def tokenize_str2list(self, text):
        """
        """
        text = text.strip().lower()
        
        text = re.sub("[ ]+", " ", text)
        
        tokenized = ""
        is_last_special = False
        is_last_cjk = False
        is_last_num_or_alphabet = False
        for i,ch in enumerate(text):
            if is_last_special == True:
                tokenized += ch
                if is_special(ch) == True:
                    tokenized += " "
                    is_last_special = False
            elif is_cjk(ch):
                if is_last_cjk == True:
                    tokenized += (ch) 
                else:
                    tokenized += (" " + ch)
                is_last_cjk = True
                is_last_num_or_alphabet = False
            elif is_num(ch) or is_alphabet(ch):                
                if is_last_num_or_alphabet == True:
                    tokenized += ch
                else:
                    tokenized += (" " + ch)
                is_last_cjk = False
                is_last_num_or_alphabet = True
            elif is_special(ch):
                tokenized += (" " + ch)
                is_last_special = True
                is_last_cjk = False
                is_last_num_or_alphabet = True
            elif is_useless(ch):
                is_last_cjk = False
                is_last_num_or_alphabet = False                  
            elif is_space(ch):
                if is_alphabet(text[i-1]) and is_alphabet(text[i+1]): 
                    tokenized += " "
                elif is_special(text[i-1]) or is_special(text[i+1]): 
                    tokenized += " "
                else:
                    tokenized += " _s_ "
                is_last_cjk = False
                is_last_num_or_alphabet = False                      
            else:
                tokenized += (" " + ch + " ") 
                is_last_cjk = False
                is_last_num_or_alphabet = False
        
        tokenized = re.sub("[ ]+", " ", tokenized).strip()

        tokens = []
        for token in tokenized.split():
            if is_special(token[0]) and is_special(token[-1]):
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


    def detokenize_tokens(self, tokens):
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
            elif is_special(token[0]) and is_special(token[-1]):
                text += (" " + token + " ")
                is_last_alphabet = False
            else:
                is_cur_alphabet = False
                if all(is_alphabet(ch) for ch in token):
                    is_cur_alphabet = True
                    
                if is_last_alphabet == True and is_cur_alphabet == True:
                    text += (" " + token)
                else:
                    text += token
                is_last_alphabet = is_cur_alphabet
        
        text = text.replace("_s_", " ")
        text = re.sub("[ ]+", " ", text)
        text = text.strip()

        return text


def build_tokenizer(**args):
    """
    """
    if args["tokenizer"] == "default":
        tokenizer = SpaceTokenizer(vocab=args["vocab"], 
                                   pre_tokenized=args["pre_tokenized"], 
                                   pre_vectorized=args["pre_vectorized"])
    elif args["tokenizer"] == "wordpiece":
        tokenizer = WordPieceTokenizer(vocab=args["vocab"], 
                                       pre_tokenized=args["pre_tokenized"], 
                                       pre_vectorized=args["pre_vectorized"])
    else:
        raise ValueError("tokenizer not correct!")
        
    return tokenizer

