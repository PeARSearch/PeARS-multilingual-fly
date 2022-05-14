import joblib
import numpy as np
from os.path import join
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from utils import read_vocab, hash_dataset_, read_n_encode_dataset, encode_docs

def init_vectorizer(lang):
    spm_vocab = f"../../PeARS-fruit-fly/spm/spm.wiki.vocab"
    vocab, reverse_vocab, logprobs = read_vocab(spm_vocab)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern='[^ ]+')
    return vectorizer, reverse_vocab, logprobs

def apply_hacked_model(lang,ridge,dataset):
    vectorizer, reverse_vocab, logprobs = init_vectorizer(lang)
    logprob_power=7
    train_set, train_titles, train_labels = read_n_encode_dataset(dataset, vectorizer, logprobs, logprob_power)
    train_set = train_set.todense()
    scaler = preprocessing.MinMaxScaler().fit(train_set)
    train_set = scaler.transform(train_set)
    umap_m = joblib.load(dataset.replace('sp','umap.m')).todense()
    pred = ridge.predict(train_set)
    print(mean_squared_error(pred, umap_m))
    print(pred[0],'\n',umap_m[0],'\n')

def hack_umap_model(lang=None, dataset=None):
    print('--- Learning regression model over UMAP ---')
    vectorizer, reverse_vocab, logprobs = init_vectorizer(lang)
    logprob_power=7
    train_set, train_titles, train_labels = read_n_encode_dataset(dataset, vectorizer, logprobs, logprob_power)
    train_set = train_set.todense()
    scaler = preprocessing.MinMaxScaler().fit(train_set)
    train_set = scaler.transform(train_set)
    umap_m = joblib.load(dataset.replace('sp','umap.m')).todense()
    ridge = Ridge(alpha = 0.9)
    ridge.fit(train_set, umap_m)
    pred = ridge.predict(train_set)
    print(mean_squared_error(pred, umap_m))
    print(pred[0],'\n',umap_m[0],'\n')
    joblib.dump(ridge,'ridge.tmp')
    return ridge

ridge = hack_umap_model('en', '../../PeARS-fruit-fly/web_map/umap/processed/enwiki-latest-pages-articles1.xml-p1p41242.sp')
apply_hacked_model('en',ridge,'../../PeARS-fruit-fly/web_map/umap/processed/enwiki-latest-pages-articles20.xml-p34308443p35522432.sp')
