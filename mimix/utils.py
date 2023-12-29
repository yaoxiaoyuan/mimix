# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:39:56 2019

@author: Xiaoyuan Yao
"""
import sys
import os
import configparser
import json
import random
import numpy as np
from abc import ABC, abstractmethod

home_dir = os.path.abspath(os.getcwd())


SYMBOLS = {"PAD_TOK" : "_pad_",
           "BOS_TOK" : "_bos_",
           "EOS_TOK" : "_eos_",
           "UNK_TOK" : "_unk_",
           "SEP_TOK" : "_sep_",
           "CLS_TOK" : "_cls_",
           "MASK_TOK" : "_mask_"}
           
SYMBOL2ID = {"_pad_":0,
             "_bos_":1,
             "_eos_":2,
             "_unk_":3,
             "_sep_":4,
             "_cls_":5,
             "_mask_":6}


def real_path(path, base_dir=None):
    """
    get real path
    """
    if path is None:
        return None
    if os.path.isabs(path) == True:
        return path
    if base_dir is None:
        base_dir = home_dir
    return os.path.join(base_dir, path)


def load_config(config_file):
    """
    load config
    """
    config = configparser.RawConfigParser()
    config.optionxform = str 

    config_file = real_path(config_file)
    if not os.path.exists(config_file):
        print("config file %s not exist!" % config_file)
        sys.exit(0)
        
    config.read(config_file, encoding='utf-8')
    
    loaded_config = {}
    
    for dtype in config.sections():
        if dtype not in ["str", "int", "float", "bool"]:
            continue
        for k,v in config.items(dtype):
            if dtype == "str":
                loaded_config[k] = str(v)
            elif dtype == "int":
                loaded_config[k] = int(v)
            elif dtype == "float":
                loaded_config[k] = float(v)                 
            elif dtype == "bool":
                if v.lower() == "false":
                    loaded_config[k] = False
                elif v.lower() == "true":
                    loaded_config[k] = True
    return loaded_config


def load_model_config(config_file):
    """
    load config
    """
    loaded_config = load_config(config_file)
    
    loaded_config["symbols"] = SYMBOLS
    loaded_config["symbol2id"] = SYMBOL2ID
    
    for symbol in SYMBOLS:
        if symbol + "2tok" in loaded_config:
            loaded_config["symbols"][symbol] = loaded_config[symbol + "2tok"]
    
    for symbol in SYMBOL2ID:
        if symbol + "2id" in loaded_config:
            loaded_config["symbol2id"][symbol] = loaded_config[symbol + "2id"]   

    return loaded_config


def load_vocab(vocab_path):
    """
    """
    vocab = {}
    for i,line in enumerate(open(real_path(vocab_path), "rb")):
        line = line.decode("utf-8").strip()
        if "\t" in line:
            word, word_id = line.split("\t")
        else:
            word, word_id = line, i
        vocab[word] = int(word_id)
    
    return vocab


def invert_dict(dic):
    """
    """
    return {dic[k]:k for k in dic}


def cut_and_pad_seq(seq, max_len, pad, left=False):
    """
    """
    if left == True:
        return [pad] * (max_len - len(seq)) + seq[:max_len]
    return seq[:max_len] + [pad] * (max_len - len(seq))


def cut_and_pad_seq_list(seq_list, max_len, pad, auto=False, pad_left=False):
    """
    """
    if auto == True:
        max_len = min(max(len(seq) for seq in seq_list), max_len)
        
    x = []
    for seq in seq_list:
        x.append(cut_and_pad_seq(seq, max_len, pad, pad_left))

    return x


def derange(xs):
    for a in range(1, len(xs)):
        b = random.randint(0, a-1)
        xs[a], xs[b] = xs[b], xs[a]
    return xs


def nested_to_device(nested_tensor, device):
    """
    """
    res = nested_tensor
    if isinstance(nested_tensor, list) == True or isinstance(nested_tensor, tuple) == True:
        res = []
        for elem in nested_tensor:
            res.append(nested_to_device(elem, device))
    else:
        res = nested_tensor.to(device)
    return res


def word_dropout(word_list, rate, replace_token):
    """
    """
    if rate > 0:
        tmp = []
        
        for word in word_list:
            if random.random() < rate:
                tmp.append(replace_token)
            else:
                tmp.append(word)
        
        word_list = tmp
        
    return word_list


class SimpleDataset(ABC):
    """
    """
    def __init__(self, device="cpu", rank=0, world_size=1):
        """
        """
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.sort_key_fn = None
    

    @abstractmethod
    def vectorize(self, batch_data):
        """
        """
        pass

    
    def local_shuffle(self):
        """
        """
        for f in os.listdir(self.data_dir):
            lines = [line for line in open(os.path.join(self.data_dir, f), "r", encoding="utf-8")]
            random.shuffle(lines)
            if self.sort_key_fn is not None:
                lines = [[line, self.sort_key_fn(json.loads(line))] for line in lines]
                lines.sort(key=lambda x:x[1])
                lines = [x[0] for x in lines]
            fo = open(os.path.join(self.data_dir, f), "w", encoding="utf-8")
            for line in lines:
                fo.write(line)
            fo.close()
    
    
    def __call__(self, start_steps=0):
        """
        """
        data = []
        files = os.listdir(self.data_dir)
        files.sort()
        
        steps = 1
        for fi in files:
            fi = os.path.join(self.data_dir, fi)
            for line in open(fi, "r", encoding="utf-8", errors="ignore"):
                steps += 1
                if steps < start_steps * self.batch_size:
                    continue
                if steps % self.world_size != self.rank:
                    continue
                data.append(json.loads(line))
                if len(data) % (20 * self.batch_size) == 0:
                    batch_data = data[:self.batch_size]
                    data = data[self.batch_size:]
                    yield nested_to_device(self.vectorize(batch_data), self.device)
                
        while len(data) > 0:
            batch_data = data[:self.batch_size]
            yield nested_to_device(self.vectorize(batch_data), self.device)           
            data = data[self.batch_size:]


def convert_tokens_to_midi(tokens, output_path):
    """
    """
    import mido

    new_midi_file = mido.MidiFile(ticks_per_beat=384)
    new_track = mido.MidiTrack()
    new_midi_file.tracks.append(new_track)
    new_midi_file.tracks[0].append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    new_midi_file.tracks[0].append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    
    def add_event(track, kv):
        """
        """
        kv = {k:int(kv[k]) if k != "type" else kv[k] for k in kv}
        if len(kv) == 0:
            pass
        elif kv["type"] == "note_on":
            track.append(mido.Message('note_on', note=kv["note"], velocity=kv["velocity"], time=kv["time"]))
        elif kv["type"] == "control_change":
            track.append(mido.Message('control_change', control=kv["control"], value=kv["value"], time=kv["time"]))
        elif kv["type"] == "program_change":
            track.append(mido.Message('program_change', program=int(kv["program"])))
    
    event = {}
    for token in tokens:
        print(token, event)
        if token.startswith("note_on_note_"):
            add_event(new_midi_file.tracks[0], event)
            event = {}
            event["type"] = "note_on"
            k = "note"
            event[k] = token.split("_")[-1]
        elif token.startswith("note_on_velocity_"):
            k = "velocity"
            event[k] = token.split("_")[-1]
        elif token.startswith("note_on_time_"):
            k = "time"
            event[k] = token.split("_")[-1]
        elif token.startswith("control_change_control_"):
            add_event(new_midi_file.tracks[0], event)
            event = {}
            event["type"] = "control_change"
            k = "control"
            event[k] = token.split("_")[-1]
        elif token.startswith("control_change_time_"):
            k = "time"
            event[k] = token.split("_")[-1]
        elif token.startswith("control_change_value_"):
            k = "value"
            event[k] = token.split("_")[-1]
        elif token.startswith("program_change_program_"):
            add_event(new_midi_file.tracks[0], event)
            event["type"] = "program_change"
            event = {}
            k = "program"
            event[k] = token.split("_")[-1]
        elif token.startswith("#"):
            event[k] = event[k] + token[1:]
    new_midi_file.tracks[0].append(mido.MetaMessage('end_of_track', time=0))
    new_midi_file.save(output_path)


if __name__ == "__main__":
    pass
    


