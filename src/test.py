# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:10:26 2021

@author: Xiaoyuan Yao
"""
import tokenization
import generation_metric

def test_tokenization():
    """
    """
    test_str = "1234567号选手是top10哦,hello你好666啊,春 秋 忽 代 谢windows7le, _苍天啊_苍天_"
    mimix_tokenizer = tokenization.MimixTokenizer(
            vocab_file="../model/vocab/zh_vocab.txt",
            pre_tokenized=False,
            pre_vectorized=False)

    print(mimix_tokenizer.tokenize(test_str))

    mimix_tokenizer = tokenization.MimixTokenizer(
            vocab_file="../model/vocab/zh_words_vocab.txt",
            pre_tokenized=False,
            pre_vectorized=False)
    
    print(mimix_tokenizer.tokenize(test_str))
    
    bert_tokenizer = tokenization.BertTokenizer(
            vocab_file="../model/pretrain/bert-base-chinese/vocab.txt",
            pre_tokenized=False,
            pre_vectorized=False)
    
    print(bert_tokenizer.tokenize(test_str))  

    test_str = "1234567号选手是top10哦,_mask_hello你好666啊,春 秋 忽 代 谢windows7le, _苍天啊_苍天_"
    mimix_tokenizer = tokenization.MimixTokenizer(
            vocab_file="../model/vocab/zh_vocab.txt",
            pre_tokenized=False,
            pre_vectorized=False)

    print(mimix_tokenizer.tokenize(test_str))


def test_generation_metric():
    """
    """
    reference_corpus = ["今 天 天 气 真 好".split()]
    translation_corpus = ["今 天 天 气 真 不 错".split()]
    print(generation_metric.compute_bleu(reference_corpus, translation_corpus))
    
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    print(sentence_bleu(
                    references=reference_corpus,
                    hypothesis=translation_corpus[0],
                    smoothing_function=SmoothingFunction().method1
                )
        )

if __name__ == "__main__":
    
    test_tokenization()
    
    test_generation_metric()
    
    
    