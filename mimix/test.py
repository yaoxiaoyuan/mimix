# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:10:26 2021

@author: Xiaoyuan Yao
"""
import mimix.tokenization as tokenization
import mimix.evaluate as evaluate

def test_tokenization():
    """
    """
    test_str = "1234567号选手是top10哦,hello你好666啊,春 秋 忽 代 谢windows7le, _苍天啊_苍天_오늘 날씨가 참 좋네요."
    mimix_tokenizer = tokenization.MimixTokenizer(vocab_file="model/vocab/zh_vocab.txt")

    print(mimix_tokenizer.tokenize(test_str))

    mimix_tokenizer = tokenization.MimixTokenizer(vocab_file="model/vocab/zh_words_vocab.txt")
    
    print(mimix_tokenizer.tokenize(test_str))
    
    #bert_tokenizer = tokenization.BertTokenizer(
    #        vocab_file="model/pretrain/bert-base-chinese/vocab.txt")
    
    #print(bert_tokenizer.tokenize(test_str))  

    test_str = "1234567号选手是top10哦,_mask_hello你好666啊,春 秋 忽 代 谢windows7le, _苍天啊_苍天_오늘 날씨가 참 좋네요."
    mimix_tokenizer = tokenization.MimixTokenizer(vocab_file="model/vocab/zh_vocab.txt")

    print(mimix_tokenizer.tokenize(test_str))


def test_evaluate():
    """
    """
    ref_corpus = ["今 天 天 气 真 不 错".split(),
                  "我 好 无 聊 啊".split(),]
    eval_corpus = ["今 天 天 气 真 的 不 错".split(),
                   "我 真 的 很 无 聊".split()]
    print(ref_corpus, eval_corpus)
    print(evaluate.compute_bleu(ref_corpus, eval_corpus, 4))
    print(evaluate.rouge_n(eval_corpus, ref_corpus, 1))
    print(evaluate.rouge_n(eval_corpus, ref_corpus, 2))
    print(evaluate.rouge_l_sentence_level(eval_corpus, ref_corpus))
    
    print("------")
    
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
    print("bleu-4", corpus_bleu(
                    list_of_references=[[s] for s in ref_corpus],
                    hypotheses=eval_corpus,
                    smoothing_function=SmoothingFunction().method1
                )
        )
    

    from rouge import Rouge
    rouge = Rouge()
    res = rouge.get_scores(hyps=[" ".join(s) for s in eval_corpus], 
                           refs=[" ".join(s) for s in ref_corpus],
                           avg=True)
    for k in res:
        print(k, res[k]["f"])

if __name__ == "__main__":
    
    test_tokenization()
    
    test_evaluate()
    
    
    