# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata
import six

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  #with tf.gfile.GFile(vocab_file, "r") as reader:
  with open(vocab_file, "r", encoding="utf-8") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab


def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    output.append(vocab[item])
  return output


def convert_tokens_to_ids(vocab, tokens):
  return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
  return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens


class FullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=True):
    self.vocab = load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

  def tokenize(self, text):
    split_tokens = []
    for token in self.basic_tokenizer.tokenize(text):
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
  """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

  def __init__(self, do_lower_case=True):
    """Constructs a BasicTokenizer.
    Args:
      do_lower_case: Whether to lower case the input.
    """
    self.do_lower_case = do_lower_case

  def tokenize(self, text):
    """Tokenizes a piece of text."""
    text = convert_to_unicode(text)
    text = self._clean_text(text)

    # This was added on November 1st, 2018 for the multilingual and Chinese
    # models. This is also applied to the English models now, but it doesn't
    # matter since the English models were not trained on any Chinese data
    # and generally don't have any Chinese data in them (there are Chinese
    # characters in the vocabulary because Wikipedia does have some Chinese
    # words in the English Wikipedia.).
    text = self._tokenize_chinese_chars(text)

    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
      if self.do_lower_case:
        token = token.lower()
        token = self._run_strip_accents(token)
      split_tokens.extend(self._run_split_on_punc(token))

    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens

  def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]

  def _tokenize_chinese_chars(self, text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

  def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)


class WordpieceTokenizer(object):
  """Runs WordPiece tokenziation."""

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text):
    """Tokenizes a piece of text into its word pieces.
    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.
    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]
    Args:
      text: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.
    Returns:
      A list of wordpiece tokens.
    """

    text = convert_to_unicode(text)

    output_tokens = []
    for token in whitespace_tokenize(text):
      chars = list(token)
      if len(chars) > self.max_input_chars_per_word:
        output_tokens.append(self.unk_token)
        continue

      is_bad = False
      start = 0
      sub_tokens = []
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr is None:
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        start = end

      if is_bad:
        output_tokens.append(self.unk_token)
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens


def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically contorl characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  """Checks whether `chars` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat in ("Cc", "Cf"):
    return True
  return False


def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False


# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 21:43:09 2023

@author: 1
"""
import os
import json
import torch

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


def load_gpt2_weights(gpt2, weight_path):
    """
    """
    gpt2.eval()
    state_dict = torch.load(weight_path,
                            map_location=lambda storage, loc: storage)
    
    model_state_dict = {}
    model_state_dict["decoder.trg_embedding.W"] = state_dict["transformer.wte.weight"]
    model_state_dict["decoder.pos_embedding.W"] = state_dict["transformer.wpe.weight"]
    
    for i in range(gpt2.n_dec_layers):
        n = state_dict["transformer.h.%d.attn.c_attn.weight" % i].shape[1]
        model_state_dict["decoder.layers.%d.self_attention.W_q" % i] = state_dict["transformer.h.%d.attn.c_attn.weight" % i][:,:n//3].T
        model_state_dict["decoder.layers.%d.self_attention.b_q" % i] = state_dict["transformer.h.%d.attn.c_attn.bias" % i][:n//3]
        model_state_dict["decoder.layers.%d.self_attention.W_k" % i] = state_dict["transformer.h.%d.attn.c_attn.weight" % i][:,n//3:2*n//3].T
        model_state_dict["decoder.layers.%d.self_attention.b_k" % i] = state_dict["transformer.h.%d.attn.c_attn.bias" % i][n//3:2*n//3]
        model_state_dict["decoder.layers.%d.self_attention.W_v" % i] = state_dict["transformer.h.%d.attn.c_attn.weight" % i][:,-n//3:].T
        model_state_dict["decoder.layers.%d.self_attention.b_v" % i] = state_dict["transformer.h.%d.attn.c_attn.bias" % i][-n//3:]
        model_state_dict["decoder.layers.%d.self_attention.W_o" % i] = state_dict["transformer.h.%d.attn.c_proj.weight" % i].T
        model_state_dict["decoder.layers.%d.self_attention.b_o" % i] = state_dict["transformer.h.%d.attn.c_proj.bias" % i]
        model_state_dict["decoder.layers.%d.norm_1.alpha" % i] = state_dict["transformer.h.%d.ln_1.weight" % i]
        model_state_dict["decoder.layers.%d.norm_1.bias" % i] = state_dict["transformer.h.%d.ln_1.bias" % i]
        model_state_dict["decoder.layers.%d.ffn.W1" % i] = state_dict["transformer.h.%d.mlp.c_fc.weight" % i].T
        model_state_dict["decoder.layers.%d.ffn.b1" % i] = state_dict["transformer.h.%d.mlp.c_fc.bias" % i]
        model_state_dict["decoder.layers.%d.ffn.W2" % i] = state_dict["transformer.h.%d.mlp.c_proj.weight" % i].T
        model_state_dict["decoder.layers.%d.ffn.b2" % i] = state_dict["transformer.h.%d.mlp.c_proj.bias" % i]
        model_state_dict["decoder.layers.%d.norm_3.alpha" % i] = state_dict["transformer.h.%d.ln_2.weight" % i]
        model_state_dict["decoder.layers.%d.norm_3.bias" % i] = state_dict["transformer.h.%d.ln_2.bias" % i]
    
    if gpt2.share_emb_out_proj == False:
        model_state_dict["decoder.W"] = state_dict["lm_head.weight"]
    if "lm_head.bias" in state_dict:
        model_state_dict["decoder.b"] = state_dict["lm_head.bias"]
    
    model_state_dict["decoder.norm.alpha"] = state_dict["transformer.ln_f.weight"]
    model_state_dict["decoder.norm.bias"] = state_dict["transformer.ln_f.bias"]

    
    gpt2.load_state_dict(model_state_dict, True)
    
    return gpt2
