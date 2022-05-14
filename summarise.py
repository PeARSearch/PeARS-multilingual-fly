"""Inspect output of fruit fly pipeline on Wikipedia

Usage:
  summarise.py clusters --lang=<language code>
  summarise.py umap --lang=<language code>
  summarise.py (-h | --help)
  summarise.py --version

Options:
  --lang=<language code>         The language of the processed Wikipedia.
  -h --help                      Show this screen.
  --version                      Show version.

"""

from docopt import docopt
from glob import glob
import joblib
import  pickle
from os.path import join
from random import shuffle
from fly.vectorizer import vectorize
from scipy.spatial.distance import cdist
import numpy as np


if __name__ == '__main__':
    args = docopt(__doc__, version='Inspect Wikipedia processe dump, ver 0.1')
    lang = args['--lang']

    if args['clusters']:
        print("## Showing cluster labels retrieved by Birch over hacked UMAP representations (training set) ##")
        lang_dir = f"./datasets/data/{lang}/"
        cluster_path = join(lang_dir,f"{lang}wiki.cluster.labels.pkl")
        clusters = pickle.load(open(cluster_path,'rb'))
        for k,v in clusters.items():
            print(k,v)

    if args['umap']:
        print("## Showing random sample of corpus (UMAP representations) ##")
        sp_files = glob(f"./datasets/data/{lang}/{lang}wiki-latest-pages-articles*sp")
        shuffle(sp_files)
        random_sp_file = sp_files[0]
        umap_m = joblib.load(random_sp_file.replace('sp','umap.m')).todense()
        idx2cl = pickle.load(open(random_sp_file.replace('sp','idx2cl.pkl'),'rb'))
        _, titles = vectorize(lang, random_sp_file)

        lang_dir = f"./datasets/data/{lang}/"
        cluster_path = join(lang_dir,f"{lang}wiki.cluster.labels.pkl")
        clusters = pickle.load(open(cluster_path,'rb'))
        random_idx = list(range(umap_m.shape[0]))
        shuffle(random_idx)
        for idx in random_idx[:10]:
            cl = idx2cl[idx]
            print(idx,titles[idx], cl, clusters[cl])

        query = random_idx[:10] 
        out = cdist(umap_m,umap_m,'cosine')
        print(out)
