# A PeARS fly in any language

This repo contains all the tools needed to process the documents of an entire Wikipedia snapshot and convert them into PeARS document representations, labeled with cluster names. The high-level idea is that you input Wikipedia and you output a map of current knowledge organised into topics (the clusters) and represented in binary form. The binary vectors are locality-sensitive, i.e. you can compute semantic similarity over them. The result of this processing step is a stable semantic grid, which you can attach new documents to. For instance, if you index a Medium post on AI, the system should give it a representation located in the AI/ML cluster of the Wikipedia map. In the future, this will allow for faster search, as a query on AI is only compared to documents in the AI cluster, rather than all documents available in the PeARS collection.

**Output:** a representation of an entire Wikipedia dump as 256-bit locality-sensitive binary vectors. 


## Install

We recommend to run the code in a virtual environment:

    git clone https://github.com/PeARSearch/PeARS-multilingual-fly.git
    virtualenv PeARS-multilingual-fly/env
    cd PeARS-multilingual-fly
    source env/bin/activate

Install the requirements:

    pip install -r requirements.txt


## Running the entire pipeline in one go

The entire pipeline can be run using the following command:

    python3 run.py --lang=<language> --pipeline

The language code is the prefix to your Wikipedia url. For instance, The English Wikipedia is at *https://en.wikipedia.org/* so its language code is *en*. The Hindi Wikipedia is at *https://hi.wikipedia.org/*, so its language code is *hi*.

If you are curious about what happens in the background, go and check out [the wiki](https://github.com/PeARSearch/PeARS-multilingual-fly/wiki). 

**NB:** for large Wikipedia snapshots, processing will take a long time. For the largest wikis (English, German, French, Dutch...) this can easily run into an entire day, depending on the number and strength of CPUs on your machine. You may want to try out of the pipeline on a smaller wiki first, to get a feel for the system. (The [simple English Wikipedia](https://simple.wikipedia.org/wiki/Main_Page) may be a good choice, and can be run using *--lang=simple*. But in all likelihood, the processing will still take a few hours.)


## Running the pipeline in steps

If running the entire pipeline seems overwhelming, you can also do it in steps. Usage for the *run.py* is defined as follow:

```
Usage:
  run.py --lang=<language_code> --pipeline
  run.py --lang=<language_code> --train_tokenizer
  run.py --lang=<language_code> --download_wiki
  run.py --lang=<language_code> --train_umap
  run.py --lang=<language_code> --train_pca
  run.py --lang=<language_code> --cluster_data
  run.py --lang=<language_code> --train_fly
  run.py --lang=<language_code> --binarize_data

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
  -h --help                      Show this screen.
  --version                      Show version.

```

We recommend running the following in this order:

```
run.py --lang=<language_code> --train_tokenizer
run.py --lang=<language_code> --download_wiki
run.py --lang=<language_code> --train_umap
run.py --lang=<language_code> --cluster_data
run.py --lang=<language_code> --train_fly
run.py --lang=<language_code> --binarize_data

```

If for any reason, you have trouble running UMAP on your machine, you can replace the *--train_umap* step with the following:

```
run.py --lang=<language_code> --train_pca
```

This will train a PCA model rather than UMAP, which give slightly worse document representations but may be kinder to your machine.


## Inspecting the output

You can inspect the configuration of the representations built over Wikipedia, by using the *analyze.py* tool. Please check [the wiki](https://github.com/PeARSearch/PeARS-multilingual-fly/wiki/Analysis) for more details.
