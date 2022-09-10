#encoding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold,datasets
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

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


if __name__ == "__main__":

    vis_clusters("../test_data/data/clustering/news_title_20220831_out.clu", "test.png")
