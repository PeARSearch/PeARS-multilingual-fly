"""Inspect output of fruit fly pipeline on Wikipedia

Usage:
  analyze.py clusters --lang=<language_code>
  analyze.py umap --lang=<language_code>
  analyze.py fly --lang=<language_code>
  analyze.py hack --lang=<language_code>
  analyze.py filecheck --lang=<language_code>
  analyze.py (-h | --help)
  analyze.py --version

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
from fly.apply_models import apply_hacked_umap
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
        print(len(clusters), " clusters found:")
        for k,v in clusters.items():
            print(k,v)

    if args['umap']:
        print("\n## Showing distribution of articles across clusters (UMAP representations) ##")
        sp_files = glob(f"./datasets/data/{lang}/{lang}wiki-latest-pages-articles.*sp")
        shuffle(sp_files)
        random_sp_file = sp_files[0]
        umap_m = joblib.load(random_sp_file.replace('sp','umap.m')).todense()
        idx2cl = pickle.load(open(random_sp_file.replace('sp','idx2cl.pkl'),'rb'))
        _, all_titles,_ = vectorize(lang, random_sp_file, 6, 500)

        cl2titles = {}
        for idx,cl in enumerate(idx2cl):
            #print(all_titles[idx],cl)
            if cl not in cl2titles:
                cl2titles[cl] = [all_titles[idx]]
            else:
                cl2titles[cl].append(all_titles[idx])

        for cl,titles in cl2titles.items():
            print('\n',cl,titles)


        print("\n\n\n## Showing list of random articles and their cluster labels (UMAP representations) ##")
        lang_dir = f"./datasets/data/{lang}/"
        cluster_path = join(lang_dir,f"{lang}wiki.cluster.labels.pkl")
        clusters = pickle.load(open(cluster_path,'rb'))
        random_idx = list(range(umap_m.shape[0]))
        shuffle(random_idx)
        for idx in random_idx[:20]:
            cl = idx2cl[idx]
            print(idx,all_titles[idx], cl, clusters[cl])

        print("\n\n\n## Showing articles similar to some query (UMAP representations) ##")
        k = 10
        sample_m = umap_m[:10000]
        cos = 1 - cdist(sample_m,sample_m,'cosine')
        inds = np.argpartition(cos, -k, axis=1)[:, -k:]
        for query in range(inds.shape[0]):
            print(query, all_titles[query])
            print([all_titles[i] for i in inds[query]])

    if args['fly']:
        print("\n## Showing sample docs from fruit fly buckets) ##")
        fh_files = glob(f"./datasets/data/{lang}/fhs/*fh")
        shuffle(fh_files)
        random_fh_file = fh_files[0]
        print("RANDOM FILE:",random_fh_file)
        fh_tls = joblib.load(random_fh_file)
        titles = fh_tls[1]
        fh = fh_tls[0].todense()
        print("SAMPLE TITLES:", titles[:20])
        hams = 1 - cdist(fh,fh,'hamming')
        print("AVG HAMMING:",np.mean(hams))


    if args['hack']:
        print("\n## Checking consistency of hacked UMAP ##")
        ridge_model = joblib.load(glob(join(f'./fly/models/umap/{lang}','*hacked.umap'))[0])
        sp_files = glob(f"./datasets/data/{lang}/{lang}wiki-latest-pages-articles.*sp")
        shuffle(sp_files)
        random_sp_file = sp_files[0]
        logprob_power = 6
        top_words = 500
        hacked_m, all_titles = apply_hacked_umap(lang, ridge_model, random_sp_file, logprob_power, top_words, save=False)
        k = 10
        sample_m = hacked_m.todense()[:10000]
        cos = 1 - cdist(sample_m,sample_m,'cosine')
        inds = np.argpartition(cos, -k, axis=1)[:,-k:]
        for query in range(inds.shape[0]):
            print(query, all_titles[query])
            print([(all_titles[i],cos[query][i]) for i in inds[query]])


    if args['filecheck']:
        print("\n## Checking file size for entire dump ##")
        fh_files = glob(f"./datasets/data/{lang}/{lang}wiki-latest-pages-articles*fh")

        total_docs = 0
        for f in fh_files:
            print("FILE",f)
            titles2fh = joblib.load(f)
            print("HASH FILE SIZE:",len(titles2fh))
            idx2cl = pickle.load(open(f.replace('fh','idx2cl.pkl'),'rb'))
            print("IDX2CL FILE SIZE:",len(idx2cl))
            if len(titles2fh) != len(idx2cl):
                print("WARNING -- FILE SIZES ARE DIFFERENT")
            total_docs+=len(titles2fh)
        print("TOTAL NUMBER OF DOCS:",total_docs)
