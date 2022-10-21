"""Export fly models for deployment

Usage:
  export_models_for_deployment.py --lang=<language_code>
  export_models_for_deployment.py (-h | --help)
  export_models_for_deployment.py --version
Options:
  -h --help                           Show this screen.
  --version                           Show version.
  --lang=<language_code>              Language of trained models.

"""

import configparser
from docopt import docopt
import joblib
import numpy as np
from glob import glob
from os.path import join, dirname
from pathlib import Path
from shutil import copy, make_archive
from fly.fly import Fly

def read_config(lang):
    config_path = lang+'.hyperparameters.cfg'
    config = configparser.ConfigParser()
    config.read(config_path)
    return config_path, config


class DeployedFly:
    def __init__(self):
        self.pn_size = None
        self.kc_size = None
        self.wta = None
        self.projections = None
        self.top_words = None
        self.hyperparameters = None

def export_fly(fly):
    dfly = DeployedFly()
    dfly.kc_size = fly.kc_size
    dfly.wta = fly.wta
    dfly.projections = fly.projections
    dfly.top_words = fly.top_words
    dfly.hyperparameters = fly.hyperparameters
    return dfly

args = docopt(__doc__, version='Deploying the fruit fly, ver 0.1')
lang = args['--lang']
current_path = Path().resolve()
deployment_path = join(current_path, "./pears_deployment", lang)
Path(deployment_path).mkdir(exist_ok=True, parents=True)

config_path, config = read_config(lang)
ridge_path = config['RIDGE']['path']
expander_path = join(dirname(ridge_path), lang+'wiki.expansion.m')
print(expander_path)
fly_path = config['FLY']['path']

# Copy spm model
copy(join(current_path, f'spm/{lang}/{lang}wiki.model'), deployment_path)
copy(join(current_path, f'spm/{lang}/{lang}wiki.vocab'), deployment_path)

# Copy ridge models
copy(ridge_path, deployment_path)
copy(expander_path, deployment_path)

# Prepare fly model
with open(fly_path, 'rb') as f:
    fly_model = joblib.load(glob(join(f'./fly/models/flies/{lang}','*fly.m'))[0])

deployment_fly = export_fly(fly_model)

with open(join(deployment_path,"fly.m"), "wb") as f:
    joblib.dump(deployment_fly, f)


# Throw in the config file
copy(join(current_path, config_path), deployment_path)

# Zip folder
make_archive(lang+'_deployment', 'zip', deployment_path)
