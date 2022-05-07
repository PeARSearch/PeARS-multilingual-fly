import joblib
import pickle
from glob import glob
from os.path import join, exists
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.cluster import Birch
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from collections import Counter
import numpy as np
import umap
from fly.utils import read_vocab, hash_dataset_, read_n_encode_dataset, encode_docs
from fly.fly import Fly
from fly.label_clusters import generate_cluster_labels 


def init_vectorizer(lang): 
    spm_vocab = f"./spm/{lang}/{lang}wiki.vocab"
    vocab, reverse_vocab, logprobs = read_vocab(spm_vocab)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern='[^ ]+')
    return vectorizer, logprobs


def apply_umap(umap_model, dataset, spf, save=True):
    print('\n---Applying UMAP---')
    scaler = preprocessing.MinMaxScaler().fit(dataset.todense())
    dataset = scaler.transform(dataset.todense())
    m = csr_matrix(umap_model.transform(dataset[:20000,:]))

    for i in range(20000,dataset.shape[0],20000):
        print("Reducing",i,"to",i+20000)
        m2=csr_matrix(umap_model.transform(dataset[i:i+20000,:]))
        m = vstack((m,m2))
    dataset = np.nan_to_num(m)

    if save:
        dfile = spf.replace('.sp','.umap.m')
        joblib.dump(dataset, dfile)

def apply_birch(brc, dataset, data_titles, spf, save=True):
    print('--- Cluster matrix using pretrained Birch ---')
    #Cluster points in matrix m, in batches of 20k
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



def apply_fly(fly, spf, data_titles, cluster_labels, save=True):
    print('--- Apply fly to',spf,'---')
    umap_mat = joblib.load(spf.replace('.sp','.umap.m'))
    cl2idx = pickle.load(open(spf.replace('sp','cl2idx.pkl'),'rb'))
    idx2cl = {}
    for cl,idx in cl2idx.items():
        for i in idx:
            idx2cl[i] = cluster_labels[cl]
    umap_labels = [idx2cl[i] for i in range(len(idx2cl))]
    #Compute precision at k using cluster IDs from Birch model
    score, hashed_data = fly.evaluate(umap_mat,umap_mat,umap_labels,umap_labels)

    #Save hashes 
    title2hash = {}
    for i in range(hashed_data.shape[0]):
        b = hashed_data[i][0].todense()
        #Transform long binary array into an int
        bstr = ''.join(str(i) for i in np.asarray(b)[0])
        print(bstr,data_titles[i],umap_labels[i])
        title2hash[data_titles[i]] = bstr
    if save:
        hfile = spf.replace('.sp','.fh')
        joblib.dump(title2hash, hfile)
    return score


def apply_trained_models(lang):
    umap_model = joblib.load(glob(join(f'./fly/models/umap/{lang}','*umap'))[0])
    birch_model = joblib.load(glob(join(f'./fly/models/birch/{lang}','*birch'))[0])
    fly_model = joblib.load(glob(join(f'./fly/models/flies/{lang}','*fly.m'))[0])

    sp_files = glob(join(f'./datasets/data/{lang}','*.sp'))
    for spf in sp_files:
        if not exists(spf.replace('.sp','.fh')):
            logprob_power=7 #From BO on wikipedia
            vectorizer, logprobs = init_vectorizer(lang)
            dataset, data_titles, data_labels = read_n_encode_dataset(spf, vectorizer, logprobs, logprob_power)
            apply_umap(umap_model, dataset, spf, True)
            apply_birch(birch_model, dataset, data_titles, spf,True)
            cluster_labels = generate_cluster_labels(lang,verbose=False)
            apply_fly(fly_model, spf, data_titles, cluster_labels, True)
            
