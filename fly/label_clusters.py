import pickle
from glob import glob
from os.path import join
from nltk.corpus import stopwords
from collections import Counter

def generate_cluster_labels(lang=None, verbose=False):
    #Merge all cl2titles dictionary files
    print('--- Generating cluster labels ---')
    txt_f = open("./fly/langs_list.txt", "r")
    dic={}
    for line in txt_f:
        line = line.rstrip("\n").split("\t")
        dic[line[0]]=line[1]
    if lang in dic:
        stop_words=set(stopwords.words(dic[lang]))
    else:
        stop_words=set()
    cl2titles_files = glob(join(f'./datasets/data/{lang}','*.cl2titles.pkl'))
    clusters2titles = pickle.load(open(cl2titles_files[0],'rb'))
    if len(cl2titles_files) > 1:
        for f in cl2titles_files[1:]:
            tmp = pickle.load(open(f,'rb'))
            for cl,titles in tmp.items():
                if cl in clusters2titles:
                    clusters2titles[cl].extend(titles)
                else:
                    clusters2titles[cl] = titles

    #Associate a single category label with each cluster
    cluster_titles = {}
    for k,v in clusters2titles.items():
        keywords = []
        for title in v:
            keywords.extend([w for w in title.split() if w not in stop_words])
        c = Counter(keywords)
        #category = ' '.join([pair[0]+' ('+str(pair[1])+')' for pair in c.most_common()[:5]])
        category = ' '.join([pair[0] for pair in c.most_common()[:5]])
        if verbose:
            print('\n',k,len(v),category,'\n',v[:20])
        cluster_titles[k] = category

    return cluster_titles

