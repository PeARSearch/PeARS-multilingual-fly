import re
import numpy as np
from sklearn import linear_model
from scipy.sparse import csr_matrix, vstack

def read_n_encode_dataset(path, vectorizer, logprobs):
    # read
    doc_list, label_list = [], []
    doc = ""
    with open(path) as f:
        for l in f:
            l = l.rstrip('\n')
            if l[:4] == "<doc":
                m = re.search(".*class=([^ ]*)>", l)
                label = m.group(1)
                label_list.append(label)
            elif l[:5] == "</doc":
                doc_list.append(doc)
                doc = ""
            else:
                doc += l + ' '

    # encode
    X = vectorizer.fit_transform(doc_list)
    X = csr_matrix(X)
    X = X.multiply(logprobs)

    return X, label_list

def read_vocab(vocab_file):
    c = 0
    vocab = {}
    reverse_vocab = {}
    logprobs = []
    with open(vocab_file) as f:
        for l in f:
            l = l.rstrip('\n')
            wp = l.split('\t')[0]
            logprob = -(float(l.split('\t')[1]))
            #logprob = log(lp + 1.1)
            if wp in vocab or wp == '':
                continue
            vocab[wp] = c
            reverse_vocab[c] = wp
            logprobs.append(logprob**3)
            c+=1
    return vocab, reverse_vocab, logprobs

def wta_vectorized(feature_mat, k, percent=True):
    # thanks https://stackoverflow.com/a/59405060
    m, n = feature_mat.shape
    if percent:
        k = int(k * n / 100)
    # get (unsorted) indices of top-k values
    topk_indices = np.argpartition(feature_mat, -k, axis=1)[:, -k:]
    # get k-th value
    rows, _ = np.indices((m, k))
    kth_vals = feature_mat[rows, topk_indices].min(axis=1)
    # get boolean mask of values smaller than k-th
    is_smaller_than_kth = feature_mat < kth_vals[:, None]
    # replace mask by 0
    feature_mat[is_smaller_than_kth] = 0
    return feature_mat

def hash_input_vectorized_(pn_mat, weight_mat, percent_hash):
    kc_mat = pn_mat.dot(weight_mat.T)
    m, n = kc_mat.shape
    wta_csr = csr_matrix(np.zeros(n))
    for i in range(0, m, 2000):
        part = wta_vectorized(kc_mat[i: i+2000].toarray(), k=percent_hash)
        wta_csr = vstack([wta_csr, csr_matrix(part, shape=part.shape)])
    hashed_kenyon = wta_csr[1:]
    return hashed_kenyon


def hash_dataset_(dataset_mat, weight_mat, percent_hash, top_words):
    m, n = dataset_mat.shape
    dataset_mat = csr_matrix(dataset_mat)
    wta_csr = csr_matrix(np.zeros(n))
    for i in range(0, m, 2000):
        part = wta_vectorized(dataset_mat[i: i+2000].toarray(), k=top_words, percent=False)
        wta_csr = vstack([wta_csr, csr_matrix(part, shape=part.shape)])
    hs = hash_input_vectorized_(wta_csr[1:], weight_mat, percent_hash)
    hs = (hs > 0).astype(np.int_)
    return hs

def train_model(m_train,classes_train,m_val,classes_val,C,num_iter):
    lm = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear',
                                         max_iter=num_iter, C=C, verbose=0)
    lm.fit(m_train, classes_train)
    score = lm.score(m_val,classes_val)
    #print(lm.predict(m_val)[:10])
    #print(classes_val[:10])
    return score, lm

