"""Process a whole Wikipedia dump with the fruit fly

Usage:
  run.py --lang=<language_code> --pipeline
  run.py --lang=<language_code> --train_tokenizer
  run.py --lang=<language_code> --download_wiki
  run.py --lang=<language_code> --train_umap
  run.py --lang=<language_code> --train_pca
  run.py --lang=<language_code> --cluster_data
  run.py --lang=<language_code> --train_fly
  run.py --lang=<language_code> --binarize_data
  run.py --s2e

  run.py (-h | --help)
  run.py --version

Options:
  --lang=<language code>         The language of the Wikipedia to process.
  --pipeline                     Run the whole pipeline (this can take several hours!)
  --train_tokenizer              Train a sentencepiece tokenizer for a language.
  --download_wiki                Download and preprocess Wikipedia for the chosen language.
  --train_umap                   Train the UMAP dimensionality reduction model.
  --train_pca                    Train a PCA dimensionality reduction model (alternative to UMAP).
  --cluster_data                 Learn cluster names and apply clustering to the entire Wikipedia.
  --train_fly                    Train the fruit fly over dimensionality-reduced representations.
  --binarize_data                Apply the fruit fly to the entire Wikipedia.
  --s2e                          Hack to apply simple wiki models to en wiki.
  -h --help                      Show this screen.
  --version                      Show version.

"""

import configparser
from os.path import exists
from docopt import docopt
from glob import glob
from random import shuffle
import joblib

from codecarbon import EmissionsTracker
from spm.spm_train_on_wiki import mk_spm
from datasets.get_wiki_data import mk_wiki_data
from fly.train_models import train_umap, hack_umap_model, run_pca, hack_pca_model, train_birch, train_fly
from fly.apply_models import apply_dimensionality_reduction, apply_fly
from fly.prepare_clusters import generate_cluster_labels, generate_cluster_centroids


def get_training_data(lang, train_spf_path):

    def get_n_docs(input_file_path, output_file, n):
        article_count = 0
        article = ""
        input_file = open(input_file_path)
        for l in input_file:
            if "</doc" in l:
                article+=l
                output_file.write(article)
                article = ""
                article_count+=1
                if article_count == n:
                    break
            else:
                article+=l
        input_file.close()
        return article_count

    print("--- Gathering training data from sample of dump files ---")
    required_article_count = 50000
    train_spf = open(train_spf_path,'w')

    '''Get first dump file, which usually contains 'core' articles.'''
    dump_split = True
    try:
        first_sp_file = glob(f"./datasets/data/{lang}/{lang}wiki-latest-pages-articles1.*sp")[0]
    except:
        first_sp_file = f"./datasets/data/{lang}/{lang}wiki-latest-pages-articles.xml.sp"
        dump_split = False
    c = get_n_docs(first_sp_file, train_spf, 30000) #Up to 30,000 articles from first dump file
    print(">>> Gathered",c,"articles from ",first_sp_file)

    '''Get sample from other dump files, to get correct data distribution.'''
    required_article_count-=c
    if dump_split:
        spfs = glob(f"./datasets/data/{lang}/{lang}wiki-latest-pages-articles*sp")
        shuffle(spfs)
        for i in range(4):
            c = int(required_article_count / 4)
            get_n_docs(spfs[i], train_spf, c) #Articles from other dump files
            print(">>> Gathered", c,"articles from ",spfs[i])
    train_spf.close()
    print(">>> Finished building the training corpus ---")

def init_config(lang):
    config_path = lang+'.hyperparameters.cfg'
    if exists(config_path):
        return 1
    else:
        config = configparser.ConfigParser()
        config['GENERIC'] = {}
        config['GENERIC']['language'] = lang
        config['PREPROCESSING'] =  {}
        config['PREPROCESSING']['logprob_power'] = 'None'
        config['PREPROCESSING']['top_words'] =  'None'
        config['REDUCER'] =  {}
        config['REDUCER']['type'] =  'None'
        config['REDUCER']['dimensionality'] =  'None'
        config['REDUCER']['path'] = 'None'
        config['RIDGE'] =  {}
        config['RIDGE']['path'] = 'None'
        config['FLY'] =  {}
        config['FLY']['num_trials'] =  'None'
        config['FLY']['kc_size'] = 'None' 
        config['FLY']['neighbours'] =  'None'
        config['FLY']['path'] = 'None'
        with open(config_path, 'w+') as configfile:
            config.write(configfile)

def read_config(lang):
    config_path = lang+'.hyperparameters.cfg'
    config = configparser.ConfigParser()
    config.read(config_path)
    return config_path, config

def update_config(lang, section, k, v):
    config_path, config = read_config(lang)
    config[section][k] = str(v)
    with open(config_path, 'w+') as configfile:
        config.write(configfile)


if __name__ == '__main__':
    args = docopt(__doc__, version='Get Wikipedia in fruit fly vectors, ver 0.1')
    lang = args['--lang']
    #tracker = EmissionsTracker(output_dir="./emission_tracking", project_name="Multilingual Fly")
    #tracker.start()

    if args['--lang']:
        init_config(lang)
        train_path = f"./datasets/data/{lang}/{lang}wiki-latest-pages-articles.train.sp"

    if args['--train_tokenizer'] or args['--pipeline']:
        mk_spm(lang)
    
    if args['--download_wiki'] or args['--pipeline']:
        mk_wiki_data(lang, lang) #In the normal case, input and spm model case are the same language
        get_training_data(lang, train_path)

    if args['--train_umap'] or args['--pipeline']:
        umap_path, input_m, umap_m, best_logprob_power, best_top_words = train_umap(lang, train_path)
        print("UMAP LOG: BEST LOG POWER - ",best_logprob_power, "BEST TOP WORDS:", best_top_words)
        update_config(lang, 'PREPROCESSING', 'logprob_power', best_logprob_power)
        update_config(lang, 'PREPROCESSING', 'top_words', best_top_words)
        update_config(lang, 'REDUCER', 'path', umap_path)
        update_config(lang, 'REDUCER', 'type', 'UMAP')
        update_config(lang, 'REDUCER', 'dimensionality', str(umap_m.shape[1]))
        hacked_path, hacked_m = hack_umap_model(lang, train_path, best_logprob_power, best_top_words, input_m, umap_m)
        update_config(lang, 'RIDGE', 'path', hacked_path)

    if args['--train_pca']:
        pca_path, input_m, pca_m, best_logprob_power, best_top_words = run_pca(lang, train_path)
        print("PCA LOG: BEST LOG POWER - ",best_logprob_power, "BEST TOP WORDS:", best_top_words)
        update_config(lang, 'PREPROCESSING', 'logprob_power', best_logprob_power)
        update_config(lang, 'PREPROCESSING', 'top_words', best_top_words)
        update_config(lang, 'REDUCER', 'path', pca_path)
        update_config(lang, 'REDUCER', 'type', 'PCA')
        update_config(lang, 'REDUCER', 'dimensionality', str(pca_m.shape[1]))
        hacked_path, hacked_m = hack_pca_model(lang, train_path, best_logprob_power, best_top_words, input_m, pca_m)
        update_config(lang, 'RIDGE', 'path', hacked_path)

    if args['--cluster_data'] or args['--pipeline']:
        _ , config = read_config(lang)
        best_logprob_power = int(config['PREPROCESSING']['logprob_power'])
        best_top_words = int(config['PREPROCESSING']['top_words'])
        hacked_path = config['RIDGE']['path']
        hacked_m = joblib.load(hacked_path+'.m') 
        brm, labels = train_birch(lang, hacked_m)
        generate_cluster_labels(lang, train_path, labels, best_logprob_power, best_top_words)
        apply_dimensionality_reduction(lang, hacked_path, best_logprob_power, best_top_words, brm)
        if lang == 'simple':
            apply_dimensionality_reduction('en', hacked_path, best_logprob_power, best_top_words)

    if args['--train_fly'] or args['--pipeline']:
        num_trials = 10
        kc_size = 256
        k = 20
        update_config(lang, 'FLY', 'num_trials', num_trials)
        update_config(lang, 'FLY', 'kc_size', kc_size)
        update_config(lang, 'FLY', 'neighbours', k)
        fly_path, _ = train_fly(lang=lang, dataset=train_path, num_trials=num_trials, kc_size=kc_size, k=k)
        update_config(lang, 'FLY', 'path', fly_path)
    
    if args['--binarize_data'] or args['--pipeline']:
        _, config = read_config(lang)
        generate_cluster_centroids(train_path)
        best_logprob_power = int(config['PREPROCESSING']['logprob_power'])
        best_top_words = int(config['PREPROCESSING']['top_words'])
        apply_fly(lang, best_logprob_power, best_top_words)
        if lang == 'simple':
            apply_fly('simple', best_logprob_power, best_top_words, 'en')


    #tracker.stop()

