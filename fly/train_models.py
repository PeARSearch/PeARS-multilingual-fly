from os.path import join, exists
import umap
import joblib
from pathlib import Path
import pickle
from glob import glob

import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.cluster import Birch
from sklearn.metrics import pairwise_distances

from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from fly.utils import read_vocab, hash_dataset_, read_n_encode_dataset, encode_docs
from fly.fly import Fly
from fly.label_clusters import generate_cluster_labels 

def init_vectorizer(lang): 
    spm_vocab = f"./spm/{lang}/{lang}wiki.vocab"
    vocab, reverse_vocab, logprobs = read_vocab(spm_vocab)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern='[^ ]+')
    return vectorizer, logprobs

def train_birch(lang, m):
    print('--- Training Birch ---')

    brc = Birch(threshold=0.3,n_clusters=None)
    brc.fit(m[:50000,:]) #train on first 50k
    
    umap_dir = join(Path(__file__).parent.resolve(),join("models/umap",lang))
    birch_dir = join(Path(__file__).parent.resolve(),join("models/birch",lang))
    Path(birch_dir).mkdir(exist_ok=True, parents=True)
    umap_model_path = glob(join(umap_dir,"*umap"))[0]
    cfile = umap_model_path.replace('umap','birch')
    joblib.dump(brc, cfile)


#The default values here are from the BO on our Wikipedia dataset. Alternative in 2D for plotting.
#def train_umap(logprob_power=7, umap_nns=5, umap_min_dist=0.1, umap_components=2):
def train_umap(lang=None, dataset=None, logprob_power=7, umap_nns=16, umap_min_dist=0.0, umap_components=31):
    print('--- Training UMAP ---')
    vectorizer, logprobs = init_vectorizer(lang)
    train_set, train_titles, train_labels = read_n_encode_dataset(dataset, vectorizer, logprobs, logprob_power)
    train_set = train_set.todense()[:50000]
    train_labels = train_labels[:50000]
    scaler = preprocessing.MinMaxScaler().fit(train_set)
    train_set = scaler.transform(train_set)
    umap_model = umap.UMAP(n_neighbors=umap_nns, min_dist=umap_min_dist, n_components=umap_components, metric='hellinger', random_state=32).fit(train_set)

    dfile = dataset.split('/')[-1].replace('.sp','.umap')
    model_dir = join(Path(__file__).parent.resolve(),join("models/umap",lang))
    Path(model_dir).mkdir(exist_ok=True, parents=True)
    filename = join(model_dir,dfile)
    joblib.dump(umap_model, filename)
    return csr_matrix(umap_model.transform(train_set))


def train_fly(lang=None, dataset=None, logprob_power=7, kc_size=256, wta=50, proj_size=4, k=20, num_trial=50):
    max_thread = int(multiprocessing.cpu_count() * 0.2)
    vectorizer, logprobs = init_vectorizer(lang)

    train_set, train_titles, train_labels = read_n_encode_dataset(dataset, vectorizer, logprobs, logprob_power)
    cluster_labels = generate_cluster_labels(lang,verbose=False)
    umap_mat = joblib.load(dataset.replace('.sp','.umap.m'))
    cl2idx = pickle.load(open(dataset.replace('sp','cl2idx.pkl'),'rb'))
    idx2cl = {}
    for cl,idx in cl2idx.items():
        for i in idx:
            idx2cl[i] = cluster_labels[cl]
    umap_labels = [idx2cl[i] for i in range(len(idx2cl))]
    pn_size = umap_mat.shape[1]
    top_words = pn_size
    init_method = "random"
    eval_method = "similarity"
    proj_store = None
    hyperparameters = {'C':100,'num_iter':200,'num_nns':k}
    #fly = Fly(pn_size, kc_size, wta, proj_size, top_words, init_method, eval_method, proj_store, hyperparameters)
    fly_list = [Fly(pn_size, kc_size, wta, proj_size, top_words, init_method, eval_method, proj_store, hyperparameters) for _ in range(num_trial)]
    
    '''Compute precision at k using cluster IDs from Birch model'''
    print('\n--- Generating cluster labels for fly training ---')
    with Parallel(n_jobs=max_thread, prefer="threads") as parallel:
        delayed_funcs = [delayed(lambda x:x.evaluate(umap_mat,umap_mat,umap_labels,umap_labels))(fly) for fly in fly_list]
        scores = parallel(delayed_funcs)
    score_list = np.array([p[0] for p in scores])
    print(score_list)
    best = np.argmax(score_list)
    model_dir = join(Path(__file__).parent.resolve(),join("models/flies",lang))
    Path(model_dir).mkdir(exist_ok=True, parents=True)
    fly_path = join(model_dir, dataset.split('/')[-1].replace('sp','fly.m'))
    joblib.dump(fly_list[best],fly_path)
    return score_list[best], fly_list[best]


