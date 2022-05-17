import pickle
import numpy as np
from os.path import join
from fly.utils import read_vocab
from fly.vectorizer import vectorize


def generate_cluster_labels(lang=None, spf=None, labels=None, logprob_power=None, top_words=None, verbose=True):
    print('--- Generating cluster labels for Birch model ---')
    spm_vocab = f"./spm/{lang}/{lang}wiki.vocab"
    _, reverse_vocab, _ = read_vocab(spm_vocab)
    cl2idx = {}
    for i in range(len(labels)):
        cl = labels[i]
        if cl in cl2idx:
            cl2idx[cl].append(i)
        else:
            cl2idx[cl] = [i]

    m, titles, _ = vectorize(lang, spf, logprob_power, top_words)
    print(m.shape,len(labels))
    cluster_titles = {}
    for cl,idx in cl2idx.items():
        clm = m[idx]
        pn_use = np.squeeze(np.asarray(clm.sum(axis=0)))
        pn_use = pn_use / sum(pn_use)
        pn_sorted_ids = np.argsort(pn_use)[:-pn_use.shape[0]-1:-1] #Give sorted list from most to least used PNs
        pn_sorted_ids = np.squeeze(np.asarray(pn_sorted_ids))
        vocab_sorted = [reverse_vocab[pn] for pn in pn_sorted_ids if reverse_vocab[pn].replace('▁','').isalpha()]
        if verbose:
            print(cl,len(idx),vocab_sorted[:10])
        cluster_titles[cl] = ' '.join(vocab_sorted[:10])
    file_path = f'./datasets/data/{lang}/{lang}wiki.cluster.labels.pkl'
    with open(file_path,'wb') as f:
        pickle.dump(cluster_titles,f)
    return cluster_titles


#generate_cluster_labels('en', '../../PeARS-fruit-fly/web_map/umap/processed/enwiki-latest-pages-articles1.xml-p1p41242.sp')
