# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 21:50:35 2022

@author: Xiaoyuan Yao

This is a modified version of tensor2tensor bleu and rouge
See https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/bleu_hook.py
See https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/rouge.py
"""
import re
import six
import unicodedata
import collections
import math
import sys
import numpy as np

def _count_ngrams(segment, max_order):
    """Extracts all n-grams up to a given maximum order from an input segment.
    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.
    Returns:
      The Counter containing all n-grams up to max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
      for i in range(0, len(segment) - order + 1):
        ngram = tuple(segment[i:i + order])
        ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus,
                 translation_corpus,
                 max_order=4,
                 use_bp=True):
    """Computes BLEU score of translated segments against one or more references.
    Args:
      reference_corpus: list of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      use_bp: boolean, whether to apply brevity penalty.
    Returns:
      BLEU score.
    """
    reference_length = 0
    translation_length = 0
    bp = 1.0
    geo_mean = 0
    
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    precisions = []
    
    for (references, translations) in zip(reference_corpus, translation_corpus):
      reference_length += len(references)
      translation_length += len(translations)
      ref_ngram_counts = _count_ngrams(references, max_order)
      translation_ngram_counts = _count_ngrams(translations, max_order)
    
      overlap = dict((ngram,
                      min(count, translation_ngram_counts[ngram]))
                     for ngram, count in ref_ngram_counts.items())
    
      for ngram in overlap:
        matches_by_order[len(ngram) - 1] += overlap[ngram]
      for ngram in translation_ngram_counts:
        possible_matches_by_order[len(ngram)-1] += translation_ngram_counts[ngram]
    precisions = [0] * max_order
    smooth = 1.0
    for i in range(0, max_order):
      if possible_matches_by_order[i] > 0:
        precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
        if matches_by_order[i] > 0:
          precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
        else:
          smooth *= 2
          precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
      else:
        precisions[i] = 0.0
    
    if max(precisions) > 0:
      p_log_sum = sum(math.log(p) for p in precisions if p)
      geo_mean = math.exp(p_log_sum/max_order)
    
    if use_bp:
      if not reference_length:
        bp = 1.0
      else:
        ratio = translation_length / reference_length
        if ratio <= 0.0:
          bp = 0.0
        elif ratio >= 1.0:
          bp = 1.0
        else:
          bp = math.exp(1 - 1. / ratio)
    bleu = geo_mean * bp
    return np.float32(bleu)


class UnicodeRegex(object):
  """Ad-hoc hack to recognize all punctuation and symbols."""

  def __init__(self):
    punctuation = self.property_chars("P")
    self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
    self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
    self.symbol_re = re.compile("([" + self.property_chars("S") + "])")

  def property_chars(self, prefix):
    return "".join(six.unichr(x) for x in range(sys.maxunicode)
                   if unicodedata.category(six.unichr(x)).startswith(prefix))

uregex = UnicodeRegex()


def bleu_tokenize(string):
    r"""Tokenize a string following the official BLEU implementation.
    See https://github.com/moses-smt/mosesdecoder/"
             "blob/master/scripts/generic/mteval-v14.pl#L954-L983
    In our case, the input string is expected to be just one line
    and no HTML entities de-escaping is needed.
    So we just tokenize on punctuation and symbols,
    except when a punctuation is preceded and followed by a digit
    (e.g. a comma/dot as a thousand/decimal separator).
    Note that a number (e.g. a year) followed by a dot at the end of sentence
    is NOT tokenized,
    i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
    does not match this case (unless we add a space after each sentence).
    However, this error is already in the original mteval-v14.pl
    and we want to be consistent with it.
    Args:
      string: the input string
    Returns:
      a list of tokens
    """
    string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
    string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
    string = uregex.symbol_re.sub(r" \1 ", string)
    return string.split()


def _len_lcs(x, y):
    """Returns the length of the Longest Common Subsequence between two seqs.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Args:
      x: sequence of words
      y: sequence of words
    Returns
      integer: Length of LCS between x and y
    """
    table = _lcs(x, y)
    n, m = len(x), len(y)
    return table[n, m]


def _lcs(x, y):
    """Computes the length of the LCS between two seqs.
    The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Args:
      x: collection of words
      y: collection of words
    Returns:
      Table of dictionary of coord and len lcs
    """
    n, m = len(x), len(y)
    table = {}
    for i in range(n + 1):
      for j in range(m + 1):
        if i == 0 or j == 0:
          table[i, j] = 0
        elif x[i - 1] == y[j - 1]:
          table[i, j] = table[i - 1, j - 1] + 1
        else:
          table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def _recon_lcs(x, y):
    """
    Returns the Longest Subsequence between x and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: sequence of words
      y: sequence of words

    Returns:
      sequence: LCS of x and y
    """
    i, j = len(x), len(y)
    table = _lcs(x, y)

    def _recon(i, j):
        """private recon calculation"""
        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [(x[i - 1], i)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)
    
    recon_list = list(map(lambda x: x[0], _recon(i, j)))
    
    return recon_list


def _f_lcs(llcs, m, n):
    """Computes the LCS-based F-measure score.
    Source: https://www.microsoft.com/en-us/research/publication/
    rouge-a-package-for-automatic-evaluation-of-summaries/
    Args:
      llcs: Length of LCS
      m: number of words in reference summary
      n: number of words in candidate summary
    Returns:
      Float. LCS-based F-measure score
    """
    r_lcs = llcs / m
    p_lcs = llcs / n
    f_lcs = 2.0 * ((p_lcs * r_lcs) / (p_lcs + r_lcs + 1e-8))
    return f_lcs

    beta = p_lcs / (r_lcs + 1e-12)
    num = (1 + (beta**2)) * r_lcs * p_lcs
    denom = r_lcs + ((beta**2) * p_lcs)
    f_lcs = num / (denom + 1e-12)
    return f_lcs


def rouge_l_sentence_level(eval_sentences, ref_sentences):
    """Computes ROUGE-L (sentence level) of two collections of sentences.
    Source: https://www.microsoft.com/en-us/research/publication/
    rouge-a-package-for-automatic-evaluation-of-summaries/
    Calculated according to:
    R_lcs = LCS(X,Y)/m
    P_lcs = LCS(X,Y)/n
    F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
    where:
    X = reference summary
    Y = Candidate summary
    m = length of reference summary
    n = length of candidate summary
    Args:
      eval_sentences: The sentences that have been picked by the summarizer
      ref_sentences: The sentences from the reference set
    Returns:
      A float: F_lcs
    """

    f1_scores = []
    for eval_sentence, ref_sentence in zip(eval_sentences, ref_sentences):
      #m = len(ref_sentence)
      #n = len(eval_sentence)
      #lcs = _len_lcs(eval_sentence, ref_sentence)
      
      m = len(_get_ngrams(1, ref_sentence))
      n = len(_get_ngrams(1, eval_sentence))
      lcs = _recon_lcs(eval_sentence, ref_sentence)
      lcs = len(_get_ngrams(1, lcs))
     
      f1_scores.append(_f_lcs(lcs, m, n))
    return np.mean(f1_scores, dtype=np.float32)


def _get_ngrams(n, text):
    """Calculates n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
      ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def rouge_n(eval_sentences, ref_sentences, n=2):
    """Computes ROUGE-N f1 score of two text collections of sentences.
    Source: https://www.microsoft.com/en-us/research/publication/
    rouge-a-package-for-automatic-evaluation-of-summaries/
    Args:
      eval_sentences: The sentences that have been picked by the summarizer
      ref_sentences: The sentences from the reference set
      n: Size of ngram.  Defaults to 2.
    Returns:
      f1 score for ROUGE-N
    """
    
    f1_scores = []
    for eval_sentence, ref_sentence in zip(eval_sentences, ref_sentences):
      eval_ngrams = _get_ngrams(n, eval_sentence)
      ref_ngrams = _get_ngrams(n, ref_sentence)
      ref_count = len(ref_ngrams)
      eval_count = len(eval_ngrams)
    
      # Gets the overlapping ngrams between evaluated and reference
      overlapping_ngrams = eval_ngrams.intersection(ref_ngrams)
      overlapping_count = len(overlapping_ngrams)
    
      # Handle edge case. This isn't mathematically correct, but it's good enough
      if eval_count == 0:
        precision = 0.0
      else:
        precision = overlapping_count / eval_count
    
      if ref_count == 0:
        recall = 0.0
      else:
        recall = overlapping_count / ref_count
      
      f1_scores.append(2.0 * ((precision * recall) / (precision + recall + 1e-8)))
    
    # return overlapping_count / reference_count
    return np.mean(f1_scores, dtype=np.float32)


def run_generation_evaluation(ref_file, hyp_file):
    """
    """
    res = {}
    
    ref_tokens = []
    hyp_tokens = []
    for line in open(ref_file, "r", encoding="utf-8"):
        src,trg = line.strip().split("\t")[0:2]
        ref_tokens.append([src.split()])
    for line in open(hyp_file, "r", encoding="utf-8"):
        src,trg = line.strip().split("\t")[0:2]
        hyp_tokens.append(trg.split())
    
    assert len(ref_tokens) == len(hyp_tokens)

    res["bleu_4"] = compute_bleu(ref_tokens, hyp_tokens, 4)
    res["rouge-1"] = rouge_n(hyp_tokens, ref_tokens,  n=1)
    res["rouge-2"] = rouge_n(hyp_tokens, ref_tokens, n=2)
    res["rouge-L"] = rouge_l_sentence_level(hyp_tokens, ref_tokens, n=2)
    return res

if __name__ == "__main__":
    ref_file, hyp_file = sys.argv[1], sys.argv[2]
    print(run_generation_evaluation(ref_file, hyp_file))
    
