import pickle
import joblib
import numpy as np
from os.path import join
from sklearn import preprocessing
from fly.utils import read_vocab
from fly.vectorizer import vectorize


def generate_cluster_labels(lang=None, spf=None, labels=None, logprob_power=None, top_words=None, verbose=True):
    print('--- Generating cluster labels for Birch model ---')
    spm_vocab = f"./spm/{lang}/{lang}wiki.vocab"
    _, reverse_vocab, _ = read_vocab(spm_vocab)
    cl2idx = {}
    for i in range(len(labels)):
        cl = labels[i]
        if cl in cl2idx:
            cl2idx[cl].append(i)
        else:
            cl2idx[cl] = [i]

    m, titles, _ = vectorize(lang, spf, logprob_power, top_words)
    print(m.shape,len(labels))
    cluster_titles = {}
    for cl,idx in cl2idx.items():
        clm = m[idx]
        pn_use = np.squeeze(np.asarray(clm.sum(axis=0)))
        pn_use = pn_use / sum(pn_use)
        pn_sorted_ids = np.argsort(pn_use)[:-pn_use.shape[0]-1:-1] #Give sorted list from most to least used PNs
        pn_sorted_ids = np.squeeze(np.asarray(pn_sorted_ids))
        vocab_sorted = [reverse_vocab[pn] for pn in pn_sorted_ids if reverse_vocab[pn].replace('▁','').isalpha()]
        if verbose:
            print(cl,len(idx),vocab_sorted[:10])
        cluster_titles[cl] = ' '.join(vocab_sorted[:10])
    file_path = f'./datasets/data/{lang}/{lang}wiki.cluster.labels.pkl'
    with open(file_path,'wb') as f:
        pickle.dump(cluster_titles,f)
    return cluster_titles

def generate_cluster_centroids(train_path=None):
    idx2cl = pickle.load(open(train_path.replace('sp','idx2cl.pkl'),'rb'))
    umap_m = joblib.load(train_path.replace('sp','umap.m'))

    cl2idx = {}
    for idx in range(len(idx2cl)):
        cl = idx2cl[idx]
        if cl in cl2idx:
            cl2idx[cl].append(idx)
        else:
            cl2idx[cl] = [idx]

    cluster_centroid_mat = np.zeros((len(cl2idx),umap_m.shape[1]))
    for i in range(len(cl2idx)):
        centroid = np.sum(umap_m[cl2idx[i]], axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        cluster_centroid_mat[i] = centroid
    #scaler = preprocessing.Normalizer(norm='l2').fit(cluster_centroid_mat)
    #cluster_centroid_mat = scaler.transform(cluster_centroid_mat)
    print(cluster_centroid_mat)

    file_path = train_path.replace('sp','umap.centroids')
    joblib.dump(cluster_centroid_mat,file_path)


#generate_cluster_labels('en', '../../PeARS-fruit-fly/web_map/umap/processed/enwiki-latest-pages-articles1.xml-p1p41242.sp')
#generate_cluster_centroids('../datasets/data/simple/simplewiki-latest-pages-articles.train.sp')
