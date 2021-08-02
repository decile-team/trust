import numpy as np
import time
from matplotlib import pyplot as plt

def tsne_smi(lake_set, lake_targets, query_set, query_targets, selected_idx, device="cpu"):
    colors = ['blue', 'purple', 'turquoise', 'green', 'red', 'lime', 'cyan', 'orange', 'gray', 'pink']
    if(device=="cpu"): from sklearn.manifold import TSNE
    if(device=="cuda"): from tsnecuda import TSNE
    lake_tsne = TSNE(n_components=2).fit_transform(lake_set)
    query_tsne = TSNE(n_components=2).fit_transform(query_set)
    lake_idx = list(set(list(range(len(lake_tsne)))) - set(selected_idx))
    for i in lake_idx:
        plt.scatter(lake_tsne[i][0],lake_tsne[i][1], c=colors[lake_targets[i]])
    for i in range(len(query_tsne)):
        plt.scatter(query_tsne[i][0],query_tsne[i][1], c=colors[query_targets[i]],edgecolors='black')
    for i in selected_idx:
        plt.scatter(lake_tsne[i][0],lake_tsne[i][1], c=colors[lake_targets[i]], edgecolors='yellow')
    return plt