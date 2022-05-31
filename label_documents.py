"""
Label new documents from Common Crawl

Usage:
  label_documents.py --lang=<language> --path_wet=<path>
  label_documents.py (-h | --help)
  label_documents.py --version
Options:
  -h --help                 Show this screen.
  --version                 Show version.
  --lang=<language>         Language of the Common Crawl documents
  --path_wet=<path>         Folder where the .gz files from Common Crawl are located

"""

import numpy as np
import sentencepiece as spm
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cdist
from fly.fly_utils import hash_dataset_, read_vocab, encode_docs
from scipy.sparse import csr_matrix, vstack
from pathlib import Path
from random import shuffle
import joblib
from os.path import join
from docopt import docopt
from glob import glob
import gzip
import re

def read_xml(f_dataset):
  vocab, reverse_vocab, logprobs = read_vocab(spm_path.replace(".model", ".vocab"))
  vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')
  doc=""
  c=0
  docs_encoded, titles, urls, docs = [],[],[], []
  with gzip.open(f_dataset,'rt') as f:
    for l in f:
      l = l.rstrip('\n')
      if l[:4] == "<doc":
        m = re.search(".*title=([^ ]*) ",l)
        ID=m.group(1)
        titles.append(titles)
        m = re.search(".*url=([^ ]*) ",l)
        url=m.group(1)
        urls.append(url)
        continue

      if l[:5] != "</doc" and l[:4] != "<doc":
        doc += l + " " 
        continue

      if l[:5] == "</doc" and doc != "":
        ll = " ".join(sp.encode_as_pieces(doc))
        docs_encoded.append(ll); docs.append(doc)
        doc=""
        c+=1
        if c % 100 == 0:
          print(f"{c} documents processed so far...")
        continue

  X=encode_docs(docs_encoded, vectorizer, logprobs)

  return X, docs, urls


def apply_hacked_umap(data, ridge_model, save=False):
    data=data.todense()
    m = csr_matrix(ridge_model.predict(data[:20000,:]))

    for i in range(20000,data.shape[0],20000):
        print("Reducing",i,"to",i+20000)
        m2=csr_matrix(ridge_model.predict(data[i:i+20000,:]))
        m = vstack((m,m2))
    dataset = np.nan_to_num(m)

    if save:
        Path(join(path_wet, file)).mkdir(parents=True, exist_ok=True)
        dfile = join(path_wet, file+'.umap.m')
        joblib.dump(dataset, dfile)
    return dataset


def identify_class(new_data, docs):
    print("\n## Showing sample docs from fruit fly buckets) ##")
    fh_files = glob(f"./datasets/data/{lang}/{lang}wiki-latest-pages-articles1.*fh")
    shuffle(fh_files)
    random_fh_file = fh_files[0]
    k = 1
    titles2fh = joblib.load(random_fh_file)  # title as key and binary hash as value

    """
    Stuck in this part of the code. I want to be able to retrieve the name of the clusters in order to 
    compare the texts against the name of the cluster and check whether they make sense. 
    """
    # cats = joblib.load(random_fh_file.replace(".fh", ".cats.pkl"))  # titles as keys and categories as values, I guess
    # clusters = joblib.load(random_fh_file.replace(".fh", ".idx2cl.pkl"))  # indices only

    all_titles = list(titles2fh.keys())
    fhs = np.vstack([[i for i in fh] for fh in titles2fh.values()])

    for enu, data in enumerate(new_data):
        print(docs[enu])
        hams = 1 - cdist(fhs, data, 'hamming')
        hams = hams.reshape(1, -1)
        inds = np.argpartition(hams, -k, axis=1)[:, -k:].flatten()
        for neigh in inds:
            print(all_titles[neigh],hams[:, neigh])  #this prints the hamming distance with the most similar documents.
            print("\n\n")
    exit()


def label_documents(f_dataset, save=False):
  m, docs, urls = read_xml(f_dataset)
  m = apply_hacked_umap(m, ridge_model)

  hashed_data, _, _ = hash_dataset_(dataset_mat=m, weight_mat=fly_model.projections,
                                                        percent_hash=fly_model.wta, top_words=fly_model.top_words)
  hashed_data = hashed_data.todense()
  if save:
    Path(join(path_wet, file)).mkdir(parents=True, exist_ok=True)
    dfile = join(path_wet, file + '.hs')
    joblib.dump(hashed_data, dfile)

  identify_class(hashed_data, docs)



if __name__ == '__main__':
    args = docopt(__doc__, version='Labelling documents according to wiki categories, ver 0.1')

    lang = args['--lang']
    path_wet = "./"+args["--path_wet"]

    spm_path=f'./spm/{lang}wiki.model'
    sp = spm.SentencePieceProcessor()
    sp.load(spm_path)

    ridge_model = joblib.load(glob(f"./fly/models/umap/{lang}/*hacked.umap")[0])
    fly_model = joblib.load(glob(join(f'./fly/models/flies/{lang}', '*fly.m'))[0])

    files_wet = glob(path_wet+"/*.gz")

    for file in files_wet:
      label_documents(file)
