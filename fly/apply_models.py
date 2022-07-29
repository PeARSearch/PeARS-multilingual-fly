import joblib
import pickle
from glob import glob
from os.path import join, exists
from sklearn.cluster import Birch
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
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


def fly(lang, fly, spf, data_titles):
    print('--- Apply fly to',spf,'---')
    save_dir = f'./datasets/data/{lang}/fhs'
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    umap_mat = joblib.load(spf.replace('.sp','.umap.m'))
    idx2cl = pickle.load(open(spf.replace('sp','idx2cl.pkl'),'rb'))
    umap_labels = [idx2cl[i] for i in range(len(idx2cl))]
    #Compute precision at k using cluster IDs from Birch model
    score, hashed_data = fly.evaluate(umap_mat,umap_mat,umap_labels,umap_labels)

    #Check length 
    print(hashed_data.shape,len(idx2cl))

    #Save hashes per class 
    for cl in list(set(idx2cl)):
        hfile = join(save_dir,spf.split('/')[-1].replace('sp',str(cl)+'.fh'))
        idx = [i for i,c in enumerate(idx2cl) if c == cl]
        tls = [t for i,t in enumerate(data_titles) if i in idx]
        if exists(hfile):
            shf = joblib.load(hfile)
            m = vstack((shf[1],hashed_data[idx]))
            titles = hf[0].extend(tls)
        else:
            m = hashed_data[idx]
            titles = tls
        joblib.dump([m, titles], hfile)
    return score


def apply_dimensionality_reduction(lang, hacked_path, birch_model, logprob_power, top_words):
    ridge_model = joblib.load(hacked_path)
    sp_files = glob(join(f'./datasets/data/{lang}','*.sp'))
    for spf in sp_files:
        dataset, titles = apply_hacked_umap(lang, ridge_model, spf, logprob_power, top_words, True)
        apply_birch(birch_model, dataset, titles, spf, True)
            

def apply_fly(lang, logprob_power, top_words):
    fly_model = joblib.load(glob(join(f'./fly/models/flies/{lang}','*fly.m'))[0])
    sp_files = glob(join(f'./datasets/data/{lang}','*.sp'))
    for spf in sp_files:
        if 'train.sp' in spf:
            continue
        dataset, titles = vectorize_scale(lang, spf, logprob_power, top_words)
        #cluster_path = f'./datasets/data/{lang}/{lang}wiki.cluster.labels.pkl'
        #cluster_labels = pickle.load(open(cluster_path,'rb'))
        fly(lang, fly_model, spf, titles)
