"""Process a whole Wikipedia dump with the fruit fly

Usage:
  run.py --lang=<language_code>
  run.py (-h | --help)
  run.py --version

Options:
  --lang=<language code>         The language of the Wikipedia to process.
  -h --help                      Show this screen.
  --version                      Show version.

"""

from docopt import docopt
from glob import glob
import joblib

from codecarbon import EmissionsTracker
from spm.spm_train_on_wiki import mk_spm
from datasets.get_wiki_data import mk_wiki_data
from fly.train_models import train_umap, hack_umap_model, train_birch, train_fly
from fly.apply_models import apply_dimensionality_reduction, apply_fly
from fly.label_clusters import generate_cluster_labels

if __name__ == '__main__':
    args = docopt(__doc__, version='Get Wikipedia in fruit fly vectors, ver 0.1')
    lang = args['--lang']
    #tracker = EmissionsTracker(output_dir="./emission_tracking", project_name="Multilingual Fly")
    #tracker.start()

    #mk_spm(lang)
    #mk_wiki_data(lang)

    try:
        first_sp_file = glob(f"./datasets/data/{lang}/{lang}wiki-latest-pages-articles1.*sp")[0]
    except:
        first_sp_file = f"./datasets/data/{lang}/{lang}wiki-latest-pages-articles.xml.sp"
    input_m, umap_m, best_logprob_power, best_top_words = train_umap(lang, first_sp_file)
    hacked_m = hack_umap_model(lang, first_sp_file, best_logprob_power, best_top_words, input_m, umap_m)
    #brm, labels = train_birch(lang, hacked_m)
    #apply_dimensionality_reduction(lang, brm, best_logprob_power)
    generate_cluster_labels(lang, first_sp_file, labels, best_logprob_power, best_top_words)

    #train_fly(lang, first_sp_file, 32)
    #apply_fly(lang, best_logprob_power, best_top_words)

    #tracker.stop()

