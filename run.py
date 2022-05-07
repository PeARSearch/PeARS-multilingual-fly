"""Process a whole Wikipedia dump with the fruit fly

Usage:
  run.py --lang=<language code>
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
from fly.train_models import train_umap, train_birch, train_fly
from fly.reduce import reduce
from fly.apply_models import apply_trained_models

if __name__ == '__main__':
    args = docopt(__doc__, version='Get Wikipedia in fruit fly vectors, ver 0.1')
    lang = args['--lang']
    tracker = EmissionsTracker(output_dir="./emission_tracking", project_name="Multilingual Fly")
    tracker.start()

    mk_spm(lang)
    mk_wiki_data(lang)

    try:
        first_sp_file = glob(f"./datasets/data/{lang}/{lang}wiki-latest-pages-articles1.*sp")[0]
    except:
        first_sp_file = f"./datasets/data/{lang}/{lang}wiki-latest-pages-articles.xml.sp"
    m = train_umap(lang, first_sp_file)
    train_birch(lang, m)
    reduce(lang)
    train_fly(lang, first_sp_file)
    apply_trained_models(lang)

    tracker.stop()
