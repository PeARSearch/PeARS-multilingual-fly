from os.path import join, exists
import umap
import joblib
from pathlib import Path
import pickle
from glob import glob

import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from sklearn.cluster import Birch
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import ParameterGrid

from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from fly.vectorizer import vectorize_scale
from fly.evals import wiki_cat_purity
from fly.fly import Fly


def train_birch(lang, m):
    print('--- Training Birch ---')
    n = int(min(m.shape[0],50000) / 100) # Number of clusters max 500
    print("Looking for ",n,"clusters")
    brm = Birch(threshold=0.3,n_clusters=n)
    brm.fit(m) 
    labels = brm.labels_
    n_clusters = np.unique(labels).size
    print("n_clusters : %d" % n_clusters)
    return brm, labels #NB: because of a highly recursive structure, the Birch model can cause issues pickling
    

#The default values here are from the BO on our Wikipedia dataset. Alternative in 2D for plotting.
#def train_umap(logprob_power=7, umap_nns=5, umap_min_dist=0.1, umap_components=2):
def train_umap(lang=None, spf=None, logprob_power=7, umap_nns=20, umap_min_dist=0.0, umap_components=31):
    print('\n\n--- Training UMAP ---')
    dfile = spf.split('/')[-1].replace('.sp','.umap')
    model_dir = join(Path(__file__).parent.resolve(),join("models/umap",lang))
    Path(model_dir).mkdir(exist_ok=True, parents=True)
    filename = join(model_dir,dfile)
    
    param_grid = {'logprob_power': [4,6,8], 'top_words': [300,400,500], 'umap_nns' : [10,15,20], 'umap_min_dist': [0.2, 0.6, 0.9], 'umap_components':[16, 32]}
    #param_grid = {'logprob_power': [4], 'top_words': [300], 'umap_nns' : [15], 'umap_min_dist': [0.9], 'umap_components':[32]}
    grid = ParameterGrid(param_grid)
    
    scores = []
    for p in grid:
        input_m, _ = vectorize_scale(lang, spf, p['logprob_power'], p['top_words'])
        #umap_model = umap.UMAP(n_neighbors=umap_nns, min_dist=umap_min_dist, n_components=umap_components, metric='hellinger', random_state=32).fit(input_m)
        umap_model = umap.UMAP(n_neighbors=p['umap_nns'], min_dist=p['umap_min_dist'], n_components=p['umap_components'], metric='hellinger', random_state=32).fit(input_m)
        umap_m = umap_model.transform(input_m)
        print("EXAMPLE UMAP VEC:",umap_m[0][:20])
        score = wiki_cat_purity(lang=lang, spf=spf, m=umap_m, logprob_power=p['logprob_power'], top_words=p['top_words'], num_nns=20, metric="cosine", verbose=False)
        scores.append(score)
        print("ORIG UMAP SCORE:",score, p)
    best = np.argmax(scores)
    print("BEST:",scores[best],"PARAMS:",grid[best])
    p = grid[best]
    umap_model = umap.UMAP(n_neighbors=p['umap_nns'], min_dist=p['umap_min_dist'], n_components=p['umap_components'], metric='hellinger', random_state=32).fit(input_m)
    umap_m = umap_model.transform(input_m)
    wiki_cat_purity(lang=lang, spf=spf, m=umap_m, logprob_power=p['logprob_power'], top_words=p['top_words'], num_nns=20, metric="cosine", verbose=True)
    joblib.dump(umap_model, filename)
    umap_m = umap_model.transform(input_m)
    best_logprob_power = p['logprob_power']
    best_top_words = p['top_words']
    return input_m, umap_m, best_logprob_power, best_top_words

def hack_umap_model(lang=None, spf=None, logprob_power=None, top_words=None, input_m=None, umap_m=None):
    print('\n\n--- Learning regression model over UMAP ---')
    
    scores = []
    alphas = [0.1,0.3,0.5,0.7,0.9]
    for a in alphas:
        ridge = Ridge(alpha = a)
        ridge.fit(input_m, umap_m)
        ridge_m = ridge.predict(input_m)
        print("EXAMPLE RIDGE VEC:",ridge_m[0][:20])
        score = wiki_cat_purity(lang=lang, spf=spf, m=ridge_m, logprob_power=logprob_power, top_words=top_words, num_nns=20, metric="cosine", verbose=False)
        scores.append(score)
        print("HACKED UMAP SCORE:",score, a)
    
    best = np.argmax(scores)
    print("BEST:",scores[best], "ALPHA:",alphas[best])
    umap_dir = join(Path(__file__).parent.resolve(),join("models/umap",lang))
    umap_model_path = glob(join(umap_dir,"*.train.umap"))[0]
    cfile = umap_model_path.replace('.train.umap','.train.hacked.umap')
    ridge = Ridge(alpha = alphas[best])
    ridge.fit(input_m, umap_m)
    ridge_m = ridge.predict(input_m)
    joblib.dump(ridge, cfile)
    return ridge_m

def train_fly(lang=None, dataset=None, num_trials=None, logprob_power=7, kc_size=256, wta=50, proj_size=4, k=20):
    print('--- Spawning fruit flies ---')
    max_thread = int(multiprocessing.cpu_count() * 0.3)
    umap_mat = joblib.load(dataset.replace('.sp','.umap.m'))
    idx2cl = pickle.load(open(dataset.replace('sp','idx2cl.pkl'),'rb'))
    umap_labels = [idx2cl[i] for i in range(len(idx2cl))]
    pn_size = umap_mat.shape[1]
    top_words = pn_size
    init_method = "random"
    eval_method = "similarity"
    proj_store = None
    hyperparameters = {'C':100,'num_iter':200,'num_nns':k}

    print("\n\n----- Initialising",num_trials,"flies ----")
    fly_list = [Fly(pn_size, kc_size, wta, proj_size, top_words, init_method, eval_method, proj_store, hyperparameters) for _ in range(num_trials)]
    
    '''Compute precision at k using cluster IDs from Birch model'''
    print("\n\n----- Evaluating",num_trials,"flies ----")
    with Parallel(n_jobs=max_thread, prefer="threads") as parallel:
        delayed_funcs = [delayed(lambda x:x.evaluate(umap_mat,umap_mat,umap_labels,umap_labels))(fly) for fly in fly_list]
        scores = parallel(delayed_funcs)
    score_list = np.array([p[0] for p in scores])
    print("\n\n----- Outputting score list for",num_trials,"flies ----")
    print(score_list)
    best = np.argmax(score_list)
    model_dir = join(Path(__file__).parent.resolve(),join("models/flies",lang))
    Path(model_dir).mkdir(exist_ok=True, parents=True)
    fly_path = join(model_dir, dataset.split('/')[-1].replace('sp','fly.m'))
    joblib.dump(fly_list[best],fly_path)
    return score_list[best], fly_list[best]


