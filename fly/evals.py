import numpy as np
from scipy.spatial.distance import cdist
from fly.vectorizer import vectorize


def wiki_cat_purity(lang=None, spf=None, m=None, num_nns=20, metric="cosine", verbose=False):

    _, _, wikicats = vectorize(lang, spf)
    for i in range(len(wikicats)):
        try:
            wikicats[i].remove('')
            wikicats[i].remove(' ')
        except:
            pass
    batch = 1024
    scores = [] # List for purity of each nearest neighbour set
    n_batches = int(m.shape[0] / batch)
    for i in range(0, n_batches * batch, batch): #Looping through batch
        sims = 1 - cdist(m[i:i+batch:,], m, metric=metric)
        inds = np.argpartition(sims, -num_nns, axis=1)[:, -num_nns:]
        for j in range(sims.shape[0]): #Looping through nearest neighbours of one batch
            c = 0     #Number of instances with categories for target and NNs
            score = 0 #Score for that batch
            if len(wikicats[i+j]) == 0:
                continue
            for nn in inds[j]:
                if len(wikicats[nn]) == 0:
                    continue
                c+=1
                if verbose:
                    print(wikicats[i+j],wikicats[nn])
                if len(list(set(wikicats[nn]) & set(wikicats[i+j]) )) > 0:
                    score+=1
            if c > 0:
                if verbose:
                    print(score,c)
                scores.append(score / c)
    return np.mean(scores)


