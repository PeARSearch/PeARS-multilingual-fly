"""Fruit Fly dataset preparation from Wikipedia in some language

Usage:
  wiki_process.py --lang=<language> --spm=<model_path> (--get_pages|--get_categories)
  wiki_process.py (-h | --help)
  wiki_process.py --version

Options:
  --lang=<language>         The language of the Wikipedia to process (e.g. pt).
  --spm=<model_path>       The path to the sentencepiece model to use (e.g. ../spm/enwiki.model).
  --get_categories   Get categories and metacategories for the given language.
  --get_pages        Get pages for the categories selected by the user. To be run after --get_categories.
  -h --help          Show this screen.
  --version          Show version.

"""

from docopt import docopt
import nltk
from nltk.util import ngrams
from os import mkdir
from os.path import join,isdir,isfile
from collections import Counter, defaultdict
from glob import glob
from random import shuffle
import requests
import pickle
import re

import sentencepiece as spm


'''Util functions for finding metacategories'''
def read_category_file(cat_file):
    cats=[]
    with open(cat_file, 'r') as f:
        for line in f.read().splitlines():
            cats.append((line, "0"))
    f.close()
    return cats

def ignore(cat_name):
    ig = False
    ignored_classes = ["wikipedia","articles","category","templates","stubs","a-class","b-class","c-class","draft-class","stub-class","list-class","start-class","wikiproject","low-importance","mid-importance","high-importance","redirects","infobox","files","wikipedians"]
    for c in ignored_classes:
        if c in cat_name:
            ig = True
            break
    return ig

def get_ngram_freqs(ngram_dir,cats,n):
    cats=[i[0] for i in cats if i[0]!=""]
    ngrams_l=[]
    dic=defaultdict(list)
    for cat in cats:
        if ignore(cat.lower()): # Get rid of Wikipedia meta-classes
            continue
        cat_pieces = sp.encode_as_pieces(cat.lower())
        all_ngrams = list(ngrams(cat_pieces, n))
        for ngram in all_ngrams:
            ngrams_l.append(ngram)
            if ngram in dic:
                dic[ngram].append(cat)
            else:
                dic[ngram] = [cat]
    fdist = nltk.FreqDist(ngrams_l)
    sort_orders = sorted(fdist.items(), key=lambda x: x[1], reverse=True)
    pkl_dir = join(ngram_dir,"pkl")
    pickle.dump(sort_orders, open(join(pkl_dir,"ngram"+str(n)+".p"), 'wb'))
    pickle.dump(dic, open(join(pkl_dir,"dic"+str(n)+".p"), 'wb'))

    readable_dir = join(ngram_dir,"readable")
    with open(join(readable_dir,str(n)+"grams.txt"), 'w') as f:
        for k in sort_orders:
            f.write(",".join(k[0])+"\t"+str(k[1])+'\n')
    f.close()

def create_metacategories(pkl_dir,threshold):
    """
    Takes the threshold of the frequency of the ngrams. 
    Returns a .txt file with categories that belong to the same meta-category on each line.
    """
    ngrams_freq_files=glob(join(pkl_dir,"ngram*.p"))
    clustered_cats = []
    orphan_cats = []
    for ngram_file in ngrams_freq_files:
        ngram_size=re.findall(r'\d+', ngram_file)[0]
        ngram_freq=pickle.load(open(ngram_file, 'rb'))
        dic=pickle.load(open(join(pkl_dir,"dic"+ngram_size+".p"), 'rb'))
        for k in ngram_freq:
            cats=set()
            if k[1] < threshold:
                for cat in set(dic[k[0]]):
                    if cat not in orphan_cats:
                        orphan_cats.append(cat)
                continue
            for cat in set(dic[k[0]]):
                cats.add(cat)
            clustered_cats.append(list(cats))
    return clustered_cats,orphan_cats


def name_metacategories(lang_dir,clustered_cats,orphan_cats):
    """
    Takes the .txt file where the categories have been divided by line
    Return a .txt file with the name of the meta-categories and their repective categories.
    """
    metacategories = []
    f_out=open(join(lang_dir,lang+'wiki.metacategory_clusters.txt'), 'w')
    for cluster in clustered_cats:
        n = 6
        name_found = False
        while not name_found and n > 0:
            ngram_strs = []
            for cat in cluster:
                cat_pieces = sp.encode_as_pieces(cat.lower())
                ngram_str = [' '.join(ngram) for ngram in ngrams(cat_pieces, n)]
                ngram_strs.extend(ngram_str)
            cou=Counter(ngram_strs) #count ngrams occurring most often in subcategory names
            comm=cou.most_common(len(cou))
            tops = [p[0] for p in comm if p[1] > 0.99 * len(cluster) and len(p[0]) > 1] # len(p[0]) is there to avoid single letter ngrams
            if len(tops) > 0:
                name_found = True
                metacategories.extend(tops)
                f_out.write("TOPICS: "+" | ".join(tops)+'\t')
                f_out.write("CATEGORIES: "+" | ".join(cluster)+'\n')
            else:
                n-=1
    for cat in orphan_cats:
        f_out.write("TOPICS: "+cat+'\t')
        f_out.write("CATEGORIES: "+cat+'\n')
    f_out.close()
    print(clustered_cats)
    print(orphan_cats)
    metacategories = list(set(metacategories + orphan_cats)) # Add orphans and eliminate duplicates

    # Write metacategories.txt with just category names
    f_out=open(join(lang_dir,lang+'wiki.metacategories.txt'), 'w')
    for m in metacategories:
        f_out.write(m+'\n')
    f_out.close()

def metacat(lang_dir,cat_file_path):
    print("\n--- Creating metacategories ---")
    cats=read_category_file(cat_file_path)
    #Create ngram dir
    ngram_dir = join(lang_dir,"ngrams")
    if not isdir(ngram_dir):
        mkdir(ngram_dir)
    #Create subdirs of ngram dir:
    pkl_dir = join(ngram_dir,"pkl")
    if not isdir(pkl_dir):
        mkdir(pkl_dir)
    readable_dir = join(ngram_dir,"readable")
    if not isdir(readable_dir):
        mkdir(readable_dir)
    #Compute ngrams
    for i in [2,3,4,5,6]:
        get_ngram_freqs(ngram_dir,cats,i)
    clustered_cats,orphan_cats = create_metacategories(pkl_dir,5) #Metacats must have at least 5 subcats
    name_metacategories(lang_dir,clustered_cats,orphan_cats)






'''Utils for getting category list from Wikipedia'''
def get_wiki_categories():
    print("\n--- Getting category list from Wikipedia ---")

    PARAMS = {
        "action": "query",
        "format": "json",
        "list": "allcategories",
        "acmin": 100,
        "aclimit": 500
    }

    cat_file_path = join(lang_dir,lang+"wiki_categories.txt")
    f = open(cat_file_path,'w')

    for i in range(1000): #1000 should be enough to get all categories, even from enwiki
        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()

        CATEGORIES = DATA["query"]["allcategories"]

        for cat in CATEGORIES:
            cat_name = cat["*"]
            m = re.search("[0-9]{4}",cat_name)
            if not m:
                f.write(cat_name+'\n')
        
        if "continue" in DATA:
            PARAMS["acfrom"] = DATA["continue"]["accontinue"]
        else:
            break
    f.close()
    return cat_file_path



'''Utils for getting page titles for particular categories from Wikipedia'''
def read_metacategory_names(lang_dir,metacat):
    categories = []
    f = open(join(lang_dir,lang+"wiki.metacategory_clusters.txt"))
    for l in f:
        l = l.rstrip('\n')
        topics,cats = l.split('\t')
        for cat in cats[12:].split('|'):
            if metacat.lower() in cat.lower() and cat not in categories:
                categories.append(cat.strip())
    return list(set(categories))

def get_wiki_page_titles(metacat):
    data_dir = "./data/"+lang
    if not isdir(data_dir):
        mkdir(data_dir)
    cat_dir = join(data_dir,"categories")
    if not isdir(cat_dir):
        mkdir(cat_dir)

    max_number_pages = 1000

    categories = read_metacategory_names(lang_dir,metacat)
    print(metacat,categories)
    if len(categories) == 0:
        print("Category not found!")
        return -1

    metacat_dir = join(cat_dir,metacat.lower().replace(' ','_'))
    if not isdir(metacat_dir):
        mkdir(metacat_dir)

    c = 0
    title_file = open(join(metacat_dir,"titles.txt"),'w')
    for cat in categories:

        PARAMS = {
            "action": "query",
            "list": "categorymembers",
            "format": "json",
            "cmtitle": "Category:"+cat,
            "cmlimit": "200"
        }

        for i in range(5):    #increase 1 to more to get additional data
            R = S.get(url=URL, params=PARAMS)
            DATA = R.json()

            PAGES = DATA["query"]["categorymembers"]

            for page in PAGES:
                title = page["title"]
                ID = str(page["pageid"])
                if title[:9] != "Category:":
                    c+=1
                    if c <= max_number_pages and "File:" not in title:
                        title_file.write(ID+' '+title+'\n')

            if "continue" in DATA and c < max_number_pages:
                PARAMS["cmcontinue"] = DATA["continue"]["cmcontinue"]
            else:
                break
    return metacat_dir

'''Utils for getting page content'''

def read_wiki_page_titles(filename):
    IDs = []
    titles = []
    f = open(filename,'r')
    for l in f:
        l.rstrip('\n')
        IDs.append(l.split()[0])
        titles.append(' '.join(l.split()[1:]))
    return IDs,titles


def retrieve_page_content(metacat_dir,num_page_per_cat):

    print("--- Connecting to API. Retrieving page content for category",metacat_dir,"---")
    title_file = join(metacat_dir,"titles.txt")
    IDs, titles = read_wiki_page_titles(title_file)
    idx = list(range(len(titles)))
    shuffle(idx)
    IDs = [IDs[i] for i in idx]
    titles = [titles[i] for i in idx]

    content_file = open(join(metacat_dir,"linear.txt"),'w')

    max_titles = min(num_page_per_cat,len(titles))
    for i in range(max_titles):
        PARAMS = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "redirects": True,
            "titles": titles[i]
            }

        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()

        PAGES = DATA["query"]["pages"]
        cat = metacat_dir.replace("./data/"+lang+"/categories/","")

        for page in PAGES:
            extract = PAGES[page]["extract"]
            if len(extract) == 0:
                continue
            content_file.write("<doc id="+IDs[i]+" url=https://en.wikipedia.org/wiki/"+titles[i].replace(' ','_')+" class="+cat+">\n")
            content_file.write(extract+'\n')
            content_file.write("</doc>\n\n")

    content_file.close()

def apply_spm(metacat_dir):
    print("--- Applying sentencepiece to category",metacat_dir,"---")
    in_filename = join(metacat_dir,"linear.txt")
    spm_filename = in_filename.replace("linear.txt","spm.txt")
    in_f = open(in_filename,'r')
    spm_f = open(spm_filename,'w')
    for l in in_f:
        if "<doc" in l or "</doc" in l:
            spm_f.write(l)
        else:
            l = l.rstrip('\n').lower()
            spm_f.write(' '.join(sp.encode_as_pieces(l))+'\n')
    spm_f.close()
    in_f.close()
    print("--- All done for ",metacat_dir,"---\n")



if __name__ == '__main__':
    args = docopt(__doc__, version='Understanding and filtering Wikipedia categories, ver 0.1')
    lang = args['--lang']
    sp = spm.SentencePieceProcessor(model_file=args['--spm'])
    S = requests.Session()
    URL = "https://"+lang+".wikipedia.org/w/api.php"
    
    lang_dir = join("./category_lists",lang)
    if not isdir(lang_dir):
        mkdir(lang_dir)

    if args['--get_categories']:
        cat_file_path = get_wiki_categories()
        metacat(lang_dir, cat_file_path)

    if args['--get_pages']:
        datacategories_dir = glob("./data/"+lang+"/categories/*")
        while True:
            user_input = input("Please enter a category name (or q to quit): ")
            if user_input != 'q':
                metacat_dir = get_wiki_page_titles(user_input)
                retrieve_page_content(metacat_dir, 200) #200 pages per cat
                apply_spm(metacat_dir)
            else:
                break
