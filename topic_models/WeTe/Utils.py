import numpy as np
import os
from sklearn.cluster import KMeans
import torch


def vision_phi(Phi, outpath='phi_output.txt', voc=None, top_n=50, topic_diversity=True):
    def get_diversity(topics):
        word = []
        for line in topics:
            word += line
        word_unique = np.unique(word)
        return len(word_unique) / len(word)
    if voc is not None:
        phi = 1
        for num, phi_layer in enumerate(Phi):
            phi = np.dot(phi, phi_layer)
            phi_k = phi.shape[1]
            f = open(outpath, 'w')
            topic_word = []
            for each in range(phi_k):
                top_n_words = get_top_n(phi[:, each], top_n, voc)
                topic_word.append(top_n_words.split()[:25])
                f.write(top_n_words)
                f.write('\n')
            f.close()
            if topic_diversity:
                td_value = get_diversity(topic_word)
            print('topic diversity at layer {}: {}'.format(num, td_value))
    else:
        print('voc need !!')

def to_list(data, device='cuda:0'):
    data_list = []
    for i in range(len(data)):
        idx = torch.where(data[i]>0)[0]
        data_list.append(torch.tensor([j for j in idx for _ in range(data[i,j])], device=device))
    return data_list



def get_top_n(phi, top_n, voc):
    top_n_words = ''
    idx = np.argsort(-phi)
    for i in range(top_n):
        index = idx[i]
        top_n_words += voc[index]
        top_n_words += ' '
    return top_n_words


def normalization(data):
    _range = np.max(data, axis=1, keepdims=True) - np.min(data, axis=1, keepdims=True)
    return (data - np.min(data, axis=1, keepdims=True)) / _range


def standardization(data):
    mu = np.mean(data, axis=1, keepdims=True)
    sigma = np.std(data, axis=1, keepdims=True)
    return (data - mu) / sigma

def cluster_kmeans(x, n=50):
    # x_norm = standardization(x)
    kmeans = KMeans(n_clusters=n, random_state=0).fit(x)
    cluster_center = kmeans.cluster_centers_    ### n, d
    return cluster_center

def pac_vis(path):
    pass