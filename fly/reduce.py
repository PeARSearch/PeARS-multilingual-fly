from os.path import join, exists
import umap
import joblib
from pathlib import Path
import pickle
from glob import glob

import nltk
import numpy as np
from docopt import docopt
from joblib import Parallel, delayed
import multiprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.cluster import Birch
from sklearn.metrics import pairwise_distances
from collections import Counter

from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from fly.utils import read_vocab, hash_dataset_, read_n_encode_dataset, encode_docs


def init_vectorizer(lang): 
    spm_vocab = f"./spm/{lang}/{lang}wiki.vocab"
    vocab, reverse_vocab, logprobs = read_vocab(spm_vocab)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern='[^ ]+')
    return vectorizer, logprobs


def apply_clustering(lang, brc=None, spf=None, save=True):
    print('--- Cluster matrix using pretrained Birch ---')
    #Cluster points in matrix m, in batches of 20k
    vectorizer, logprobs = init_vectorizer(lang)
    logprob_power=7 #From experiments on wiki dataset
    data_set, data_titles, data_labels = read_n_encode_dataset(spf, vectorizer, logprobs, logprob_power)
    m = joblib.load(spf.replace('.sp','.umap.m'))
    idx2clusters = list(brc.predict(m[:20000,:]))
    clusters2idx = {}
    m = m.todense()

    for i in range(20000,m.shape[0],20000):
        print("Clustering",i,"to",i+20000)
        idx2clusters.extend(list(brc.predict(m[i:i+20000,:])))

    print('--- Save Birch output in cl2cats and cl2idx pickled files (./processed folder) ---')
    #Count items in each cluster, using labels for whole data
    cluster_counts = Counter(idx2clusters)
    print(len(idx2clusters),cluster_counts)

    #Make dictionary clusters to idx
    for cl in cluster_counts:
        clusters2idx[cl] = []
    for idx,cl in enumerate(idx2clusters):
        clusters2idx[cl].append(idx)
    
    #Make a dictionary clusters to list of categories
    clusters2titles = {}
    for cl,idx in clusters2idx.items():
        cats = [data_titles[i] for i in idx]
        clusters2titles[cl] = cats
   
    if save:
        pklf = spf.replace('sp','cl2titles.pkl')
        with open(pklf, 'wb') as f:
            pickle.dump(clusters2titles,f)
        
        pklf = spf.replace('sp','cl2idx.pkl')
        with open(pklf, 'wb') as f:
            pickle.dump(clusters2idx,f)




def apply_umap(lang, umap_model, dataset, save=True):
    print('\n---Applying UMAP to ',dataset)
    vectorizer, logprobs = init_vectorizer(lang)
    logprob_power=7 #From experiments on wiki dataset
    data_set, data_titles, data_labels = read_n_encode_dataset(dataset, vectorizer, logprobs, logprob_power)
    scaler = preprocessing.MinMaxScaler().fit(data_set.todense())
    data_set = scaler.transform(data_set.todense())
    m = csr_matrix(umap_model.transform(data_set[:20000,:]))

    for i in range(20000,data_set.shape[0],20000):
        print("Reducing",i,"to",i+20000)
        m2=csr_matrix(umap_model.transform(data_set[i:i+20000,:]))
        #print(m.shape,m2.shape)
        m = vstack((m,m2))
        #print("New m shape:",m.shape)
    data_set = np.nan_to_num(m)
    
    if save:
        dfile = dataset.replace('.sp','.umap.m')
        joblib.dump(data_set, dfile)
    return data_set, data_titles, data_labels



def reduce_data(lang, birch_model):
    print('\n-- Reduce data and apply clustering --')
    umap_dir = join(Path(__file__).parent.resolve(),join("models/umap",lang))
    birch_dir = join(Path(__file__).parent.resolve(),join("models/birch",lang))
    umap_model_path = glob(join(umap_dir,"*umap"))[0]
    umap_model = joblib.load(umap_model_path)
    
    sp_files = glob(join(f'./datasets/data/{lang}','*.sp'))
    for spf in sp_files:
        print("Processing",spf,"...")
        if not exists(spf.replace('.sp','.umap.m')):
            m, _, _ = apply_umap(lang, umap_model,spf,True)
            print("Output matrix shape:", m.shape)
            apply_clustering(lang, birch_model,spf,True)

