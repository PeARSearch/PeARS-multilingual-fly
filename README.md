# A PeARS fly in any language

This repo contains code to train a fruit fly document classifier for data in a given language. The provided pipeline assumes the user will be training the fly on data collected from Wikipedia. But it is also possible to use one's own dataset.


## Training a sentencepiece model

Before doing anything, we need to train a tokeniser to deal with the particular language of interest. This is done in the *spm* directory, by running:

    python3 spm_train_on_wiki.py --lang=<language>

The language code is the prefix to your Wikipedia url. For instance, The English Wikipedia is at *https://en.wikipedia.org/* so its language code is *en*. The Hindi Wikipedia is at *https://hi.wikipedia.org/*, so its language code is *hi*.

The result of this process is a sentencepiece model and vocabulary, to be found in the *spm* folder. You can inspect the created vocabulary file to make sure everything is okay.


## Downloading and pre-processing Wikipedia

In the *datasets* directory, we provide a script to download and pre-process an entire Wikipedia dump, using our trained sentencepiece model. As long as an appropriate sentencepiece model is available, the script can be run for any language, using the following command:

    python3 get_wiki_data.py --lang=<str>

where the argument of *--lang* is the desired language (e.g. *en* for English, *ml* for Malayalam, etc). 


## Applying dimensionality reduction and fruit fly to the Wiki corpus

We use UMAP for dimensionality reduction and Birch for clustering. The first thing we will have to do is train a UMAP and Birch model from one subset of Wikipedia. This can be done with the following command:

    python3 apply_umap_fly.py train --dataset=processed/enwiki-latest-pages-articles1.xml-p1p41242

(Here, we are training on the first file of the dump. This is usually a good choice, as older articles cover a range of fundamental topics.)

Next, we will dimensionality-reduce and cluster the Wikipedia data, file by file, using the models we have trained: 

    python3 apply_umap_fly.py reduce --model=models/umap/enwiki-latest-pages-articles1.xml-p1p41242

(The script figures out the path for the Birch model from the UMAP path.)

If desired, it is possible to get an interpretable representation of the UMAP clusters using:

    python3 apply_umap_fly.py label --lang=en

This will gather documents from all dump files, together with their respective cluster IDs, and derive keywords from them to describe each cluster.

The next and final step is to put the UMAP representations through the fruit fly. To do this, run e.g.:

    python3 apply_umap_fly.py fly --dataset=processed/enwiki-latest-pages-articles1.xml-p1p41242.sp --model=models/umap/enwiki-latest-pages-articles1.xml-p1p41242

where the argument of *--dataset* is the dump file the fly should be trained on (probably again the first dump file), and the argument of *--model* is the path to the previously trained UMAP model. In principle, it is not necessary to train the fly on the file that UMAP was trained on, but it makes good sense. 
