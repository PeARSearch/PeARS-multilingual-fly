"""Inspect output of fruit fly pipeline on Wikipedia

Usage:
  summarise.py clusters --lang=<language_code>
  summarise.py umap --lang=<language_code>
  summarise.py fly --lang=<language_code>
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
    args = docopt(__doc__, version='Inspect Wikipedia processed dump, ver 0.1')
    lang = args['--lang']

    if args['clusters']:
        print("\n## Showing cluster labels retrieved by Birch over hacked UMAP representations (training set) ##")
        lang_dir = f"./datasets/data/{lang}/"
        cluster_path = join(lang_dir,f"{lang}wiki.cluster.labels.pkl")
        clusters = pickle.load(open(cluster_path,'rb'))
        for k,v in clusters.items():
            print(k,v)

    if args['umap']:
        print("\n## Showing random sample of corpus (UMAP representations) ##")
        sp_files = glob(f"./datasets/data/{lang}/{lang}wiki-latest-pages-articles*sp")
        shuffle(sp_files)
        random_sp_file = sp_files[0]
        umap_m = joblib.load(random_sp_file.replace('sp','umap.m')).todense()
        idx2cl = pickle.load(open(random_sp_file.replace('sp','idx2cl.pkl'),'rb'))
        _, titles,_ = vectorize(lang, random_sp_file, 2)

        cl2titles = {}
        for idx,cl in enumerate(idx2cl):
            if cl not in cl2titles:
                cl2titles[cl] = [titles[idx]]
            else:
                cl2titles[cl].append(titles[idx])

        for cl,titles in cl2titles.items():
            print(cl,titles[:10])


        lang_dir = f"./datasets/data/{lang}/"
        cluster_path = join(lang_dir,f"{lang}wiki.cluster.labels.pkl")
        clusters = pickle.load(open(cluster_path,'rb'))
        random_idx = list(range(umap_m.shape[0]))
        shuffle(random_idx)
        for idx in random_idx[:10]:
            cl = idx2cl[idx]
            print(idx,titles[idx], cl, clusters[cl])

        k = 10
        sample_m = umap_m[:10000]
        cos = 1 - cdist(sample_m,sample_m,'cosine')
        inds = np.argpartition(cos, -k, axis=1)[:, -k:]
        for query in range(inds.shape[0]):
            print(query, titles[query])
            print([titles[i] for i in inds[query]])

    if args['fly']:
        print("\n## Showing sample docs from fruit fly buckets) ##")
        fh_files = glob(f"./datasets/data/{lang}/{lang}wiki-latest-pages-articles*fh")
        shuffle(fh_files)
        random_fh_file = fh_files[0]
        titles2fh = joblib.load(random_fh_file)
        fh2titles = {}
        for title,fh in titles2fh.items():
            if fh not in fh2titles:
                fh2titles[fh] = [title]
            else:
                fh2titles[fh].append(title)

        for h,titles in fh2titles.items():
            print(h,titles[:10])



