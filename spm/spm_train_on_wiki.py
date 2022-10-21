"""Train a sentencepiece model on some Wikipedia data

Usage:
  spm_train_on_wiki.py --lang=<language>
  spm_train_on_wiki.py (-h | --help)
  spm_train_on_wiki.py --version

Options:
  --lang=<language>                   The language of the Wikipedia to process, e.g. pt for Portuguese.
  -h --help                      Show this screen.
  --version                      Show version.

"""

from docopt import docopt
import os
from os.path import join
import re
import bz2
import sys
import gzip
import shutil
import pathlib
import requests
import subprocess
import sentencepiece as spm

def bz2_uncompress(filepath):
    print("--- Uncompressing downloaded bz2:",filepath,"---")
    newfilepath = filepath.replace(".bz2","")
    with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filepath, 'rb') as file:
        for data in iter(lambda : file.read(100 * 1024), b''):
            new_file.write(data)
    return newfilepath

def read_wiki_links(lang):
    links_dir = join(pathlib.Path(__file__).parent.resolve(),'wiki_dump_links')
    with open(join(links_dir,lang+"_wiki_dump_links.txt")) as f:
        return f.read().splitlines()

def get_wiki_links(lang):
    print("\n--- Getting wiki links for download ---")
    html = requests.get(url = 'https://dumps.wikimedia.org/'+lang+'wiki/latest/').text
    match = re.findall(lang+'wiki-latest-pages-articles[0-9]*\.xml-p[0-9]*p[0-9]*\.bz2', html)
    if len(match) == 0:
        match = re.findall(lang+'wiki-latest-pages-articles.xml.bz2', html) #For wikis with only one dump file.
    match = list(set(match))

    links_dir = join(pathlib.Path(__file__).parent.resolve(),'wiki_dump_links')
    filename = join(links_dir,lang+"_wiki_dump_links.txt")
    outf = open(filename,'w')
    for url in match:
        outf.write("https://dumps.wikimedia.org/"+lang+"wiki/latest/"+url+"\n")
    outf.close()
    return filename


def extract_xml(lang,wiki_paths):
    print("\n--- Downloading and extracting XML version of corpus (5M words) ---")
    out_file = open(lang+'wiki.raw.xml','w')

    bz2_file = wiki_paths[0] # We only need one bz2 file
    print(bz2_file)
    #subprocess.run(["wget",bz2_file, "-P",processed_dir])
    subprocess.run(["wget",bz2_file])
    local_file = bz2_file.split('/')[-1]
    uncompressed = bz2_uncompress(local_file)
    os.remove(local_file)
    f=open(uncompressed,'r')

    word_count = 0
    content = ""
    for l in f:
        if "</page" in l:
            out_file.write(l)
            content = ""
            if word_count > 5000000: #5M words only
                break
        else:
            out_file.write(l)
            word_count+=len(l.split()) #Very rough, will include markup. But doesn't matter.

    out_file.write("</mediawiki>")
    print("Word count:",word_count)
    f.close()
    os.remove(uncompressed)
    out_file.close()

def mk_linear(lang):
    print("\n--- Generating linear version of corpus ---")

    xml_file = lang+'wiki.raw.xml'
    tmp_linear_file = lang+'wiki.raw.tmp'
    command = ['python3','-m','wikiextractor.WikiExtractor','--output',tmp_linear_file,'--no-templates','--html-safe','False',xml_file]
    subprocess.run(command)

    os.remove(xml_file)
    tmpf = open(tmp_linear_file,'r')
    linear_filename = tmp_linear_file.replace('tmp','txt')
    linear_file = open(linear_filename,'w')
    for l in tmpf:
        if "<doc" not in l and "</doc" not in l:
            linear_file.write(l.lower())
    linear_file.close()
    tmpf.close()
    os.remove(tmp_linear_file)
    return linear_filename


def train_sentencepiece(txt_path,lang, vocab_size):
    print("\n--- Training sentencepiece on corpus ---")
    spm.SentencePieceTrainer.train(input=txt_path, model_prefix=join(join('spm',lang),txt_path.replace('.raw.txt','')), vocab_size=vocab_size, minloglevel=2)
    os.remove(txt_path)
    print("\n All done!! Your sentence piece model is at",join(join('spm',lang),txt_path.replace('.raw.txt','.model')),".")

def mk_spm(lang, vocab_size):
    links_dir = join(pathlib.Path(__file__).parent.resolve(),'wiki_dump_links')
    model_dir = join(pathlib.Path(__file__).parent.resolve(),lang)
    pathlib.Path(links_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    link_file = get_wiki_links(lang)
    wiki_paths = read_wiki_links(lang)
    extract_xml(lang,wiki_paths)
    linear_file = mk_linear(lang)
    train_sentencepiece(linear_file, lang, vocab_size)

if __name__ == '__main__':
    args = docopt(__doc__, version='Train a sentencepiece model on Wikipedia, ver 0.1')
    lang = args['--lang']
    vocab_size = int(args['--vocab'])
    sp = spm.SentencePieceProcessor()
    mk_spm(lang, vocab_size)
