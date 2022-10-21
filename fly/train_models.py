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
from sklearn.decomposition import PCA
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
  

def run_pca(lang=None, spf=None):
    print('\n\n--- Running PCA ---')
    dfile = spf.split('/')[-1].replace('.sp','.pca')
    model_dir = join(Path(__file__).parent.resolve(),join("models/pca",lang))
    Path(model_dir).mkdir(exist_ok=True, parents=True)
    filename = join(model_dir,dfile)

    param_grid = {'logprob_power': [4,7], 'top_words': [100,300,500], 'n_components':[64,128,256]}
    #param_grid = {'logprob_power': [4], 'top_words': [300], 'n_components':[256]}
    grid = ParameterGrid(param_grid)

    scores = []
    for p in grid:
        input_m, _ = vectorize_scale(lang, spf, p['logprob_power'], p['top_words'])
        #input_m = csr_matrix(input_m)
        print("\n>>> Computing PCA model...")
        pca_model = PCA(n_components=p['n_components']).fit(input_m)
        pca_m = pca_model.transform(input_m)
        print(">>> Evaluating PCA model...")
        score = wiki_cat_purity(lang=lang, spf=spf, m=pca_m, logprob_power=p['logprob_power'], top_words=p['top_words'], num_nns=20, metric="cosine", verbose=False)
        scores.append(score)
        print("ORIG PCA SCORE:",score, p)
    best = np.argmax(scores)
    print("BEST:",scores[best],"PARAMS:",grid[best])
    p = grid[best]
    input_m, _ = vectorize_scale(lang, spf, p['logprob_power'], p['top_words'])
    pca_model = PCA(n_components=p['n_components']).fit(input_m)
    pca_m = pca_model.transform(input_m)
    #wiki_cat_purity(lang=lang, spf=spf, m=pca_m, logprob_power=p['logprob_power'], top_words=p['top_words'], num_nns=20, metric="cosine", verbose=True)
    joblib.dump(pca_model, filename)
    best_logprob_power = p['logprob_power']
    best_top_words = p['top_words']
    return filename, input_m, pca_m, best_logprob_power, best_top_words


def hack_pca_model(lang=None, spf=None, logprob_power=None, top_words=None, input_m=None, pca_m=None):
    print('\n\n--- Learning regression model over PCA ---')

    scores = []
    alphas = [0.3,0.5,0.7]
    for a in alphas:
        ridge = Ridge(alpha = a)
        ridge.fit(input_m, pca_m)
        ridge_m = ridge.predict(input_m)
        score = wiki_cat_purity(lang=lang, spf=spf, m=ridge_m, logprob_power=logprob_power, top_words=top_words, num_nns=20, metric="cosine", verbose=False)
        scores.append(score)
        print("HACKED UMAP SCORE:",score, a)

    best = np.argmax(scores)
    print("BEST:",scores[best], "ALPHA:",alphas[best])
    pca_dir = join(Path(__file__).parent.resolve(),join("models/pca",lang))
    pca_model_path = glob(join(pca_dir,"*train.pca"))[0]
    cfile = pca_model_path.replace('train.pca','train.hacked.pca')
    ridge = Ridge(alpha = alphas[best])
    ridge.fit(input_m, pca_m)
    ridge_m = ridge.predict(input_m)
    joblib.dump(ridge, cfile)
    joblib.dump(ridge_m, cfile+'.m')
    return cfile, ridge_m




#The default values here are from the BO on our Wikipedia dataset. Alternative in 2D for plotting.
#def train_umap(logprob_power=7, umap_nns=5, umap_min_dist=0.1, umap_components=2):
def train_umap(lang=None, spf=None, logprob_power=7, umap_nns=20, umap_min_dist=0.0, umap_components=31):
    print('\n\n--- Training UMAP ---')
    dfile = spf.split('/')[-1].replace('.sp','.umap')
    model_dir = join(Path(__file__).parent.resolve(),join("models/umap",lang))
    Path(model_dir).mkdir(exist_ok=True, parents=True)
    filename = join(model_dir,dfile)
    
    param_grid = {'logprob_power': [6,8], 'top_words': [300,400,500], 'umap_nns' : [15,20], 'umap_min_dist': [0.2, 0.6, 0.9], 'umap_components':[16, 32]}
    #param_grid = {'logprob_power': [4], 'top_words': [300], 'umap_nns' : [15], 'umap_min_dist': [0.9], 'umap_components':[32]}
    grid = ParameterGrid(param_grid)
    
    scores = []
    for p in grid:
        input_m, _ = vectorize_scale(lang, spf, p['logprob_power'], p['top_words'])
        #umap_model = umap.UMAP(n_neighbors=umap_nns, min_dist=umap_min_dist, n_components=umap_components, metric='hellinger', random_state=32).fit(input_m)
        umap_model = umap.UMAP(n_neighbors=p['umap_nns'], min_dist=p['umap_min_dist'], n_components=p['umap_components'], metric='hellinger', random_state=32, verbose=False).fit(input_m)
        umap_m = umap_model.transform(input_m)
        score = wiki_cat_purity(lang=lang, spf=spf, m=umap_m, logprob_power=p['logprob_power'], top_words=p['top_words'], num_nns=20, metric="cosine", verbose=False)
        scores.append(score)
        print("ORIG UMAP SCORE:",score, p)
    best = np.argmax(scores)
    print("BEST:",scores[best],"PARAMS:",grid[best])
    p = grid[best]
    input_m, _ = vectorize_scale(lang, spf, p['logprob_power'], p['top_words'])
    umap_model = umap.UMAP(n_neighbors=p['umap_nns'], min_dist=p['umap_min_dist'], n_components=p['umap_components'], metric='hellinger', random_state=32).fit(input_m)
    umap_m = umap_model.transform(input_m)
    wiki_cat_purity(lang=lang, spf=spf, m=umap_m, logprob_power=p['logprob_power'], top_words=p['top_words'], num_nns=20, metric="cosine", verbose=True)
    joblib.dump(umap_model, filename)
    best_logprob_power = p['logprob_power']
    best_top_words = p['top_words']
    return filename, input_m, umap_m, best_logprob_power, best_top_words

def hack_umap_model(lang=None, spf=None, logprob_power=None, top_words=None, input_m=None, umap_m=None):
    print('\n\n--- Learning regression model over UMAP ---')
    
    scores = []
    alphas = [0.3,0.5,0.7]
    for a in alphas:
        ridge = Ridge(alpha = a)
        ridge.fit(input_m, umap_m)
        ridge_m = ridge.predict(input_m)
        score = wiki_cat_purity(lang=lang, spf=spf, m=ridge_m, logprob_power=logprob_power, top_words=top_words, num_nns=20, metric="cosine", verbose=False)
        scores.append(score)
        print("HACKED UMAP SCORE:",score, a)
    
    best = np.argmax(scores)
    print("BEST:",scores[best], "ALPHA:",alphas[best])
    umap_dir = join(Path(__file__).parent.resolve(),join("models/umap",lang))
    umap_model_path = glob(join(umap_dir,"*umap"))[0]
    cfile = umap_model_path.replace('.umap','.hacked.umap')
    ridge = Ridge(alpha = alphas[best])
    ridge.fit(input_m, umap_m)
    ridge_m = ridge.predict(input_m)
    joblib.dump(ridge, cfile)
    joblib.dump(ridge_m, cfile+'.m')
    return cfile, ridge_m

def train_fly(lang=None, dataset=None, num_trials=None, kc_size=None, k=None):
    print('--- Spawning fruit flies ---')
    model_dir = join(Path(__file__).parent.resolve(),join("models/flies",lang))
    Path(model_dir).mkdir(exist_ok=True, parents=True)
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
    fly_path = join(model_dir, dataset.split('/')[-1].replace('sp','fly.m'))

    param_grid = {'wta': [10,30,50,70], 'proj_size' : [4,8,12,16]}
    grid = ParameterGrid(param_grid)
   
    best_overall_score = 0.0
    for p in grid:
        print("\n\n----- Initialising",num_trials,"flies ----")
        fly_list = [Fly(pn_size, kc_size, p['wta'], p['proj_size'], top_words, init_method, eval_method, proj_store, hyperparameters) for _ in range(num_trials)]
        
        '''Compute precision at k using cluster IDs from Birch model'''
        print("\n\n----- Evaluating",num_trials,"flies ----")
        with Parallel(n_jobs=max_thread, prefer="threads") as parallel:
            delayed_funcs = [delayed(lambda x:x.evaluate(umap_mat,umap_mat,umap_labels,umap_labels))(fly) for fly in fly_list]
            score_list = parallel(delayed_funcs)
        scores = np.array([p[0] for p in score_list])
        print("\n\n----- Outputting score list for",num_trials,"flies ----")
        print(scores)
        best = np.argmax(scores)
        print("BEST:",scores[best],"PARAMS:",p)
        if scores[best] > best_overall_score:
            joblib.dump(fly_list[best],fly_path)
    return fly_path, best_overall_score

def train_query_expansion_model(lang=None, spf=None):
    print('\n\n--- Learning regression model for query expansion ---')

    scores = []
    m_titles = joblib.load(spf.replace('.sp','.titles.umap.m')).todense()
    m_docs = joblib.load(spf.replace('.sp','.umap.m')).todense()

    ridge = Ridge(alpha = 0.5)
    ridge.fit(m_titles, m_docs)
    score = ridge.score(m_titles, m_docs)
    scores.append(score)
    print("UMAP EXPANSION SCORE:",score)
    savedir = join(Path(__file__).parent.resolve(),join("models/umap",lang))
    savefile = join(savedir,lang+"wiki.expansion.m")
    joblib.dump(ridge,savefile)

