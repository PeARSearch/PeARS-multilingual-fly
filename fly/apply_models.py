import joblib
import pickle
from glob import glob
from os.path import join, exists
from sklearn.cluster import Birch
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from scipy.spatial.distance import cdist
from collections import Counter
import numpy as np
import umap
from fly.vectorizer import vectorize_scale
from fly.fly import Fly
from pathlib import Path




def apply_umap(lang, umap_model, spf, logprob_power, top_words, save=True):
    print('\n---Applying UMAP---')
    dataset, titles = vectorize_scale(lang, spf, logprob_power, top_words)
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
    m = m.todense()

    for i in range(20000,m.shape[0],20000):
        print("Clustering",i,"to",i+20000)
        idx2clusters.extend(list(brc.predict(m[i:i+20000,:])))

    print('--- Save Birch output in idx2cl pickled file ---')
    #Count items in each cluster, using labels for whole data
    cluster_counts = Counter(idx2clusters)
    print(len(idx2clusters),cluster_counts)

    if save:
        pklf = spf.replace('sp','idx2cl.pkl')
        with open(pklf, 'wb') as f:
            pickle.dump(idx2clusters,f)



def apply_hacked_umap(lang, ridge, spf, logprob_power, top_words, save=True):
    dataset, titles = vectorize_scale(lang, spf, logprob_power, top_words)
    m = csr_matrix(ridge.predict(dataset[:20000,:]))

    for i in range(20000,dataset.shape[0],20000):
        print("Reducing",i,"to",i+20000)
        m2=csr_matrix(ridge.predict(dataset[i:i+20000,:]))
        m = vstack((m,m2))
    dataset = np.nan_to_num(m)

    if save:
        dfile = spf.replace('.sp','.umap.m')
        joblib.dump(dataset, dfile)
    return dataset, titles


<<<<<<< HEAD
def fly_cluster(trainlang, fly, spf, data_titles, savelang):
    print('--- Apply fly to',spf,'---')
    save_dir = f'./datasets/data/{savelang}/fhs'

    Path(save_dir).mkdir(exist_ok=True, parents=True)
    cluster_centroids = joblib.load('./datasets/data/'+trainlang+'/'+trainlang+'wiki-latest-pages-articles.train.umap.centroids')
    umap_mat = joblib.load(spf.replace('.sp','.umap.m')).todense()
    print("centroids:", cluster_centroids)
    print(umap_mat)
    cosines = 1 - cdist(umap_mat, cluster_centroids, metric="cosine")
    umap_labels = list(np.argpartition(cosines, -1, axis=1)[:, -1:].squeeze())
    score, hashed_data = fly.evaluate(umap_mat,umap_mat,umap_labels,umap_labels)

    #Check length 
    print(score, hashed_data.shape,len(data_titles))

    #Save hashes per class 
    for cl in list(set(umap_labels)):
        print(save_dir, savelang, cl)
        hfile = join(save_dir,savelang+'wiki.'+str(cl)+'.fh')
        idx = [i for i,c in enumerate(umap_labels) if c == cl]
        tls = [t for i,t in enumerate(data_titles) if i in idx]
        if exists(hfile):
            m, titles = joblib.load(hfile)
            m = vstack((m,hashed_data[idx]))
            titles.extend(tls)
            print(titles[-20:])
        else:
            m = hashed_data[idx]
            titles = tls
            print(titles[-20:])
        joblib.dump([m, titles], hfile)
    return score


def fly(trainlang, fly, spf, data_titles, savelang):
    print('--- Apply fly to',spf,'---')
    umap_mat = joblib.load(spf.replace('.sp','.umap.m')).todense()
    fake_umap_labels = list(np.zeros(umap_mat.shape[0]))
    print(umap_mat.shape, len(fake_umap_labels))

    score, hashed_data = fly.evaluate(umap_mat,umap_mat,fake_umap_labels,fake_umap_labels)

    #Check length
    print(hashed_data.shape,len(data_titles))
    hfile = spf.replace('.sp','.fh')
    joblib.dump(hashed_data, hfile)


def apply_dimensionality_reduction(lang, hacked_path, logprob_power, top_words, brm):
    ridge_model = joblib.load(hacked_path)
    sp_files = glob(join(f'./datasets/data/{lang}','*.sp'))
    for spf in sp_files:
        if "titles" in spf:
            continue
        dataset, titles = apply_hacked_umap(lang, ridge_model, spf, logprob_power, top_words, True)
        if 'train' in spf:
             apply_birch(brm, dataset, titles, spf, True)
            
def apply_dimensionality_reduction_titles(lang, hacked_path, logprob_power, top_words):
    ridge_model = joblib.load(hacked_path)
    sp_files = glob(join(f'./datasets/data/{lang}','*train.titles.sp'))
    for spf in sp_files:
        dataset, titles = apply_hacked_umap(lang, ridge_model, spf, logprob_power, top_words, True)

def apply_fly(lang=None, logprob_power=None, top_words=None, cluster=False, lang2=None):
    fly_model = joblib.load(glob(join(f'./fly/models/flies/{lang}','*fly.m'))[0])
    if lang2 != None:
        sp_files = glob(join(f'./datasets/data/{lang2}','*.sp'))
    else:
        sp_files = glob(join(f'./datasets/data/{lang}','*.sp'))
    for spf in sp_files:
        if "titles" in spf:
            continue
        dataset, titles = vectorize_scale(lang, spf, logprob_power, top_words)
        if lang2 == None:
            lang2 = lang
        if cluster:
            fly_cluster(lang, fly_model, spf, titles, lang2)
        else:
            fly(lang, fly_model, spf, titles, lang2)

