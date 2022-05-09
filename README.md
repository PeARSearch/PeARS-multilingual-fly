# A PeARS fly in any language

This repo contains code to train a fruit fly document classifier for data in a given language. The provided pipeline assumes the user will be training the fly on data collected from Wikipedia and return a representation of the entire dump as 256-bit locality-sensitive binary vectors. 

To use, simply run the following:

    python3 run.py --lang=<language>

The language code is the prefix to your Wikipedia url. For instance, The English Wikipedia is at *https://en.wikipedia.org/* so its language code is *en*. The Hindi Wikipedia is at *https://hi.wikipedia.org/*, so its language code is *hi*.



## What happens in the background

For those interested, here is what the pipeline does!

* **Training a sentencepiece model:** Before doing anything, we need to train a tokeniser to deal with the particular language of interest. This is done in the *spm* directory. The result of this process is a sentencepiece model and vocabulary, to be found in the *spm* folder, under the language of interest. You can inspect the created vocabulary file to make sure everything is okay.

* **Downloading and pre-processing Wikipedia:** Running code from the *datasets* directory, the system downloads and pre-processes an entire Wikipedia dump, using our trained sentencepiece model. 

* **Applying dimensionality reduction and fruit fly to the Wiki corpus:**

We use UMAP for dimensionality reduction and Birch for clustering. The first thing we will have to do is train a UMAP and Birch model from one subset of Wikipedia. 

Next, we will dimensionality-reduce and cluster the Wikipedia data, file by file, using the models we have trained.

We also get an interpretable representation of the UMAP clusters, generating keywords to describe each cluster.

The next and final step is to put the UMAP representations through the fruit fly to produce binary vectors. 

