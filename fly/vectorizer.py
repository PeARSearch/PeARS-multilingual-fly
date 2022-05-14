from sklearn.feature_extraction.text import CountVectorizer
from fly.utils import read_vocab, read_n_encode_dataset
from sklearn import preprocessing

def init_vectorizer(lang): 
    spm_vocab = f"./spm/{lang}/{lang}wiki.vocab"
    vocab, reverse_vocab, logprobs = read_vocab(spm_vocab)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern='[^ ]+')
    return vectorizer, logprobs

def vectorize(lang, spf):
    '''Takes input file and return vectorized /scaled dataset'''
    logprob_power=7 #From BO on wikipedia
    vectorizer, logprobs = init_vectorizer(lang)
    dataset, wikititles, wikicats = read_n_encode_dataset(spf, vectorizer, logprobs, logprob_power)
    dataset = dataset.todense()
    return dataset, wikititles, wikicats

def scale(dataset):
    scaler = preprocessing.MinMaxScaler().fit(dataset)
    return scaler.transform(dataset)

def vectorize_scale(lang,spf):
    dataset, titles, _ = vectorize(lang,spf)
    return scale(dataset), titles
