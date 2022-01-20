# A PeARS fly in any language

This repo contains code to train a fruit fly document classifier for data in a given language. The provided pipeline assumes the user will be training the fly on data collected from Wikipedia. But it is also possible to use one's own dataset.


## Training a sentencepiece model

Before doing anything, we need to train a tokeniser to deal with the particular language of interest. This is done in the *spm* directory, by running:

    python3 spm_train_on_wiki.py --lang=<language code>

The language code is the prefix to your Wikipedia url. For instance, The English Wikipedia is at *https://en.wikipedia.org/* so its language code is *en*. The Hindi Wikipedia is at *https://hi.wikipedia.org/*, so its language code is *hi*.

The result of this process is a sentencepiece model and vocabulary, to be found in the *spm* folder. You can inspect the created vocabulary file to make sure everything is okay.


## Wikipedia data collection and preprocessing 

**NB:** if you want to use your own data, please skip this step.

The *datasets* folder contains code to get a list of Wikipedia categories for a given language. The code also analyses category names to check whether some categories could be merged into one larger metacategory (this is useful for highly split categories such as *moths* in the English Wikipedia, which is divided into subcategories such as *Moths of South America*, *Moths described in 1776*, etc.

The code can be called in the following way:

    wiki_process.py --lang=<language code> --spm=<spm model path> --get_categories

