# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 19:52:32 2022

@author: Xiaoyuan Yao
"""
from argparse import ArgumentParser
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold,datasets
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import json
from collections import Counter
from annoy import AnnoyIndex
from mimix.predictor import TextEncoder
from mimix.utils import real_path, load_model_config


def ann_clustering(fi_path, fo_path, dim, search_n, threshold, min_size):
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
        indexs,scores = t.get_nns_by_item(i, search_n, include_distances=True)
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
            

def text_clustering(model_config, fi_path, fo_path, search_n, threshold, min_size):
    """
    """    
    model = TextEncoder(model_config)
    print("encode text to vector...")
    model.dump_encode_text(fi_path, fo_path + ".vec")
    
    print("text clustering...")
    ann_clustering(fo_path + ".vec", 
                   fo_path + ".clu", 
                   config["d_model"],
                   search_n, 
                   threshold,
                   min_size
                   )

    print("vis clustering...")
    vis_clusters(fo_path + ".clu", fo_path + ".png")


def vis_clusters(fi_path, fig_path):
    """
    """
    X = []
    y = []
    labels = []
    n = 0
    for i,line in enumerate(open(fi_path, "r", encoding="utf-8")):
        if i >= 10:
            break
        data = json.loads(line)
        for d in data:
            X.append(d["vec"])
            y.append(i)
            labels.append(d["text"])
        n += 1

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8))

    X_trans = [[[], [], []] for i in range(n)]
    for i,j in enumerate(y):
        X_trans[j][0].append(X_norm[i, 0])
        X_trans[j][1].append(X_norm[i, 1])
        X_trans[j][2].append(labels[i])

    for i, (x, y, label) in enumerate(X_trans):
        plt.scatter(x, y, s=50, color=plt.cm.Set3(i), marker="x", label=label[0])
    plt.legend(loc='best')

    plt.savefig(fig_path, dpi=300, bbox_inches = 'tight')
    #plt.show()
    plt.close()


def run_clustering():
    """
    """
    parser = ArgumentParser()

    parser.add_argument("--model_conf", type=str)
    parser.add_argument("--fi", type=str)
    parser.add_argument("--fo", type=str)
    parser.add_argument("--search_n", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--min_size", type=int)
    
    args = parser.parse_args(sys.argv[1:])

    model_config = load_model_config(real_path(args.model_conf))
    
    text_clustering(model_config, 
                    real_path(args.fi), 
                    real_path(args.fo),
                    n,
                    threshold,
                    min_size)


if __name__ == "__main__":
    run_clustering()
