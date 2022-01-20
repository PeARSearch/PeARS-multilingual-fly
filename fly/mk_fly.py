"""Create fly from a dataset in some language

Usage:
  mk_fly.py --data_dir=<dir path> --spm_model=<model path>
  mk_fly.py (-h | --help)
  mk_fly.py --version

Options:
  --data_dir=<dir path>         Path to the data to use to train the fly, split into categories.
  --spm_model=<model path>      Path to the spm model for the language under consideration.
  -h --help                     Show this screen.
  --version                     Show version.

"""

from docopt import docopt
from glob import glob
from random import shuffle
from os import mkdir
from os.path import join,isdir
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import CountVectorizer
from utils import read_vocab, read_n_encode_dataset, hash_dataset_, train_model

class Fly:
    def __init__(self):
        self.kc_size = KC_SIZE
        self.wta = WTA
        self.proj_size = PROJ_SIZE
        weight_mat, self.shuffled_idx = self.create_projections(self.proj_size)
        self.projections = lil_matrix(weight_mat)
        self.val_score = 0
        self.is_evaluated = False

    def create_projections(self,proj_size):
        weight_mat = np.zeros((self.kc_size, PN_SIZE))
        idx = list(range(PN_SIZE))
        shuffle(idx)
        c = 0
        while c < self.kc_size:
            for i in range(0,len(idx),proj_size):
                p = idx[i:i+proj_size]
                for j in p:
                    weight_mat[c][j] = 1
                c+=1
                if c >= self.kc_size:
                    break
        return weight_mat, idx

    def get_coverage(self):
        ps = self.projections.toarray()
        vocab_cov = (PN_SIZE - np.where(~ps.any(axis=0))[0].shape[0]) / PN_SIZE
        kc_cov = (self.kc_size - np.where(~ps.any(axis=1))[0].shape[0]) / self.kc_size
        return vocab_cov, kc_cov

    def get_fitness(self):
        if not self.is_evaluated:
            return 0
        else:
            return self.val_score

    def evaluate(self):
        val_score = 0
        hash_train = hash_dataset_(dataset_mat=train_set, weight_mat=self.projections,
                                       percent_hash=self.wta, top_words=TOP_WORDS)
        hash_val = hash_dataset_(dataset_mat=val_set, weight_mat=self.projections,
                                     percent_hash=self.wta, top_words=TOP_WORDS)
        self.val_score, _ = train_model(m_train=hash_train, classes_train=train_labels,
                                       m_val=hash_val, classes_val=val_labels,
                                       C=C, num_iter=NUM_ITER)
        self.is_evaluated = True
        print("\nCOVERAGE:",self.get_coverage())
        print("SCORE:",self.val_score)
        print("PROJECTIONS:",self.print_projections())
        return val_score, self.kc_size, self.wta

    def print_projections(self):
        words = ''
        for row in self.projections[:10]:
            cs = np.where(row.toarray()[0] == 1)[0]
            for i in cs:
                words+=reverse_vocab[i]+' '
            words+='|'
        return words



def mk_train_test(data_dir,vectorizer):
    '''Gather data, read categories and generate train/test data'''
    '''This assumes the dataset is small enough to fit in memory.'''

    def write_docs(ds,path):
        shuffle(ds)
        f_out = open(path,'w')
        for d in ds:
            f_out.write(d+'\n')
        f_out.close()

    for f in ['./train','./val']:
        if not isdir(f):
            mkdir(f)
    all_cats = glob(join(data_dir,'*'))
    train_docs = []
    val_docs = []
    for cat in all_cats:
        print(cat)
        in_filename = join(cat,'spm.txt')
        docs = []
        current_doc = ""
        with open(in_filename) as in_f:
            for l in in_f:
                if "<doc" in l:
                    current_doc = l
                elif "</doc" in l:
                    current_doc+=l
                    docs.append(current_doc)
                else:
                    current_doc+=l
        shuffle(docs)
        train_n = int(len(docs) * 70 / 100)
        train_docs.extend(docs[:train_n])
        val_docs.extend(docs[train_n:])
    path = data_dir.replace('/','_').replace('.','')+'.'
    write_docs(train_docs,'./train/'+path+'train.sp')
    write_docs(val_docs,'./val/'+path+'val.sp')

    train_set, train_labels = read_n_encode_dataset('./train/'+path+'train.sp', vectorizer, logprobs)
    val_set, val_labels = read_n_encode_dataset('./val/'+path+'val.sp', vectorizer, logprobs)
    return train_set, train_labels, val_set, val_labels
                

if __name__ == '__main__':
    args = docopt(__doc__, version='Make a fly out of a preprocessed corpus, ver 0.1')
    data_dir = args['--data_dir']
    spm_model = args['--spm_model']
    vocab, reverse_vocab, logprobs = read_vocab(spm_model.replace("model","vocab"))
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern='[^ ]+')
    train_set, train_labels, val_set, val_labels = mk_train_test(data_dir,vectorizer)

    #Fly parameters
    PN_SIZE = len(vocab)
    KC_SIZE = 300
    PROJ_SIZE = 4
    WTA = 10
    TOP_WORDS = 350
    
    #Hyperparameters for classifier, kept fixed
    C = 100
    NUM_ITER = 2000
    

    fly = Fly()
    print(fly.get_coverage())
    fly.evaluate()
