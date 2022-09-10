# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 19:52:32 2022

@author: Xiaoyuan Yao
"""

import json
from collections import Counter
from annoy import AnnoyIndex
from predictor import TextMatcher
from utils import real_path, parse_test_args, load_config

def encode_texts(config, fi_path, fo_path):
    """
    """
    encoder = TextMatcher(config)
    cache = []
    fo = open(fo_path, "w", encoding="utf-8")

    def process_batch(cache):
        """
        """
        texts = [d["text"] for d in cache]
        vecs = encoder.encode_texts(texts).cpu().numpy().tolist()
        for data,vec in zip(cache, vecs):
            data["vec"] = vec
            fo.write(json.dumps(data, ensure_ascii=False) + "\n")

    for line in open(fi_path, "r", encoding="utf-8"):
        data = json.loads(line)
        cache.append(data)
        if len(cache) >= config["test_batch_size"]:
            process_batch(cache)
            cache = []
    if len(cache) > 0:
        process_batch(cache)
    
    fo.close()


def ann_clustering(fi_path, fo_path, dim, n, threshold, min_size):
    """
    """
    data = []       
    t = AnnoyIndex(dim, 'angular')
    for i,line in enumerate(open(fi_path, "r", encoding="utf-8")):
        if i % 10000 == 0:
            print("load data", i)
        d = json.loads(line)
        t.add_item(i, d["vec"])
        #del d["vec"]
        data.append(d)
    print("load over")
    t.build(100)

    id2clu = {}
    clusters = []
    for i,d in enumerate(data):
        if i % 10000 == 0:
            print("processed %d" % i)
        if i in id2clu:
            continue
        indexs,scores = t.get_nns_by_item(i, n, include_distances=True)
        count = Counter()
        near_but_no_label = []
        for idx,score in zip(indexs, scores):
            if i == idx:
                continue
            if 1 - score**2/2 > threshold:
                if idx in id2clu:
                    count[id2clu[idx]] += 1
                else:
                    count[-1] += 1
                    near_but_no_label.append(idx)
        if len(count) > 0:
            max_id = count.most_common(1)[0][0]
            if max_id > -1:
                id2clu[i] = max_id
                clusters[max_id].append(d)
            else:
                id2clu[i] = len(clusters)
                for idx in near_but_no_label:
                    id2clu[idx] = len(clusters)
                clu = [d]
                for idx in near_but_no_label:
                    clu.append(data[idx])
                clusters.append(clu)
        else:
            id2clu[i] = len(clusters)
            clu = [d]
            clusters.append(clu)

    clusters.sort(key=lambda x:len(x), reverse=True)
    fo = open(fo_path, "w", encoding="utf-8")
    for i,clu in enumerate(clusters):
        if len(clu) < min_size:
            break
        if i < 20:
            print("clu %d size: %d\n--------\n" % (i, len(clu)))
            for d in clu[:5]:
                print(" "*8, d["text"])
        fo.write(json.dumps(clu, ensure_ascii=False) + "\n")
    fo.close()
            

def text_clustering(config):
    """
    """
    fi_path = real_path(config["test_in"])

    fo_path = real_path(config["test_out"])
    
    print("encode text to vector...")
    encode_texts(config, fi_path, fo_path + ".vec")
    
    print("text clustering...")
    ann_clustering(fo_path + ".vec", 
                   fo_path + ".clu", 
                   config["d_model"],
                   config.get("clu_ann_n", 50), 
                   config.get("clu_ann_threshold", 0.8),
                   config.get("clu_min_size", 3)
                   )
 

def run_clustering():
    """
    """
    usage = "usage: run_clustering.py --model_conf <file>"
    options = parse_test_args(usage)
    conf_file = options.model_config
    config = load_config(real_path(conf_file), add_symbol=True)
    
    text_clustering(config)


if __name__ == "__main__":
    run_clustering()
