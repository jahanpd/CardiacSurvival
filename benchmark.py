import os
import json
import wandb
import pandas as pd
import argparse
import numpy as np

ENTITY="cardiac-ml"
PROJECT="survival"
K = 2
REPEATS = 5

api = wandb.Api()

parser = argparse.ArgumentParser()
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--seed", type=int, default=69)
parser.add_argument("--models", choices=['cox', 'dt', 'rf', 'gbm', 'svm', 'xgboost'], nargs='+')

args = parser.parse_args()

def get_store(path):
    try:
        with open(path) as file:
            store = json.load(file)
        return store
    except Exception as e:
        print(e)
        return dict()

def save_store(store, path):
    with open(path, 'w') as file:
        json.dump(store, file)

# import datainfo
store_path = "./results/store.json"
store = get_store(store_path)
print(store)
rng = np.random.default_rng(args.seed)

def get_sweep(entity, project, sweep_id):
    try:
        sweep = api.sweep("{}/{}/{}".format(entity,project,sweep_id))
        return sweep.best_run().config
    except Exception as e:
        print(e)
        return "FAILED"

for model in args.models:
    if model not in store:
        print("Model hyperparameters not available for {}".format(model))
        continue

    sweep = get_sweep(ENTITY, PROJECT, store[model])
    if sweep == "FAILED":
        print("Failed to retrieve sweep for {}".format(model))
        continue

    seed = int(rng.integers(1,9999))
    parameters = {
        'alpha':sweep["alpha"],
        'l1_ratio':sweep["l1_ratio"],
        'learning_rate':sweep["learning_rate"],
        'loss':sweep["loss"],
        'n_estimators':sweep["n_estimators"],
        'max_depth':sweep["max_depth"],
        'min_samples_leaf':sweep["min_samples_leaf"],
        'min_samples_split':sweep["min_samples_split"],
        'kernel':sweep["kernel"],
        'model':model,
        'run_name':model,
        'k': K,
        'repeats':REPEATS,
        'seed':seed
    }
    command = ("python train.py "
                "--alpha {alpha} "
                "--l1_ratio {l1_ratio} "
                "--learning_rate {learning_rate} "
                "--loss {loss} "
                "--n_estimators {n_estimators} "
                "--max_depth {max_depth} "
                "--min_samples_leaf {min_samples_leaf} "
                "--min_samples_split {min_samples_split} "
                "--kernel {kernel} "
                "--model {model} "
                "--dataset ANZSCTS "
                "--seed {seed} "
                "--run_name {run_name} "
                "--k {k} "
                "--repeats {repeats} "
                "--bias "
            ).format(**parameters)
    print(command)
    os.system(command)
