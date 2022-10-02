import os
import wandb
import pandas as pd
import argparse
import numpy as np
import json

COUNT=15
ENTITY="cardiac-ml"
PROJECT="survival"


api = wandb.Api()

parser = argparse.ArgumentParser()
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--models", choices=['cox', 'dt', 'rf', 'gbm', 'svm', 'xgboost'], nargs='+')
args = parser.parse_args()

sweep_config_base = {
    "program":"train.py",
    "name":"dataset",
    "method":"bayes",
    "metric":{"goal":"maximize","name":"validation_cindex.mean"},
    "parameters":{
     },
    "command":[
        "${env}",
        "${interpreter}",
        "${program}",
        "--sweep",
        "--dataset",
        "ANZSCTS",
        "--k",
        "2",
        "--repeats",
        "2",
        "--model"
    ]
} 
parameters = {
    'cox':{
        # 'alpha':{"max":10.0,"min":0.0001,"distribution":"log_uniform_values"},
        'l1_ratio':{"max":1.0,"min":0.0,"distribution":"uniform"}
    }, 
    'dt':{
        'max_depth':{"max":12,"min":2,"distribution":"int_uniform"},
        'min_samples_split':{"max":100,"min":1,"distribution":"int_uniform"},
        'min_samples_leaf':{"max":199,"min":1,"distribution":"int_uniform"},
    },
    'rf':{
        'max_depth':{"max":12,"min":2,"distribution":"int_uniform"},
        'min_samples_split':{"max":100,"min":2,"distribution":"int_uniform"},
        'min_samples_leaf':{"max":100,"min":1,"distribution":"int_uniform"},
        'max_samples':{"max":1.0,"min":0.0,"distribution":"uniform"},
    },
    'gbm':{
        'learning_rate':{"max":1.0,"min":0.0001,"distribution":"log_uniform_values"},
        'max_depth':{"max":12,"min":2,"distribution":"int_uniform"},
        'min_samples_split':{"max":100,"min":1,"distribution":"int_uniform"},
        'min_samples_leaf':{"max":100,"min":1,"distribution":"int_uniform"},
        'subsample':{"max":1.0,"min":0.0,"distribution":"uniform"},
    },
    'svm':{
        'kernel':{"values":["linear", "poly", "rbf", "sigmoid", "cosine"],"distribution":"categorical"},
        'alpha':{"max":1.0,"min":0.00000001,"distribution":"log_uniform_values"}
    },
    'xgboost':{
        'learning_rate':{"max":10.0,"min":0.001,"distribution":"log_uniform_values"},
        # 'max_depth':{"max":12,"min":2,"distribution":"int_uniform"},
        'subsample':{"max":1.0,"min":0.0,"distribution":"uniform"},
        'n_estimators':{"max":1000,"min":10,"distribution":"int_uniform"},
    },
}

def get_store(path):
    try:
        with open(path) as file:
            store = json.load(file)
        return store
    except:
        return dict()

def save_store(store, path):
    with open(path, 'w') as file:
        json.dump(store, file)

# import datainfo
store_path = "results/store.json"
store = get_store(store_path)

def check_sweep(entity, project, sweep_id, config):
    try:
        sweep = api.sweep("{}/{}/{}".format(entity,project,sweep_id))
        return sweep_id, len(sweep.runs)
    except Exception as e:
        print(e)
        sweep_id = wandb.sweep(config,entity=ENTITY,project=PROJECT)
        return sweep_id, 0

def run_hpsearch():
    for model in args.models:
        config = sweep_config_base.copy()
        config["command"] = sweep_config_base["command"].copy()

        config["parameters"] = parameters[model]
        config["command"].append(model)
        config["command"].append("${args}")
        config["name"] = model

        if model not in store or args.overwrite:
            store[model] = ""

        store[model], count = check_sweep(ENTITY, PROJECT, store[model], config)

        if count < COUNT:
            os.environ["SWEEPID"] = store[model]
            os.system("wandb agent --count {} cardiac-ml/survival/{}".format(COUNT - count, store[model]))
            save_store(store, store_path)

run_hpsearch()
