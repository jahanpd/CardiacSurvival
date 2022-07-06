import argparse
import wandb
import numpy as np
import pandas as pd
from models import make_model
from data import datasets
from sklearn.model_selection import RepeatedKFold
from numpy.random import default_rng

# Build your ArgumentParser however you like
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=['test', 'ANZSCTS'], default='test')
# Add arg for model type
parser.add_argument("--model", choices=['cox', 'dt', 'rf', 'gbm', 'svm'], default='cox')
# This argument is for Cox regression
parser.add_argument("--alpha", type=float, default=0.00000000001) # also in SVM
# Arguments for trees
parser.add_argument("--learning_rate", type=float, default=0.1) # gbm only
parser.add_argument("--loss", choices=['coxph', 'squared', 'ipcwls'], default='coxph') # gbm only
parser.add_argument("--n_estimators", type=int, default=100) # both
parser.add_argument("--max_depth", type=int, default=20) # both/all
parser.add_argument("--min_samples_split", type=int, default=6) # both/all
parser.add_argument("--min_samples_leaf", type=int, default=3) # both/all
parser.add_argument("--min_weight_fraction_leaf", type=float, default=0) # both/all
# Agruments for SVM
parser.add_argument("--kernel", choices=["linear", "poly", "rbf", "sigmoid", "cosine"], default="linear") # both/all
# Randomness seed
parser.add_argument("--seed", type=int, default=69) 

# Get the experiment info, hyperparameters, and seed
args = parser.parse_args()

# Pass them to wandb.init
wandb.init(config=args, name=args.dataset, entity="cardiac-ml", project="survival")
# Access all hyperparameter values through wandb.config
config = wandb.config

# define recorded metrics
wandb.define_metric("training_cindex", summary="mean")
wandb.define_metric("validation_cindex", summary="mean")
wandb.define_metric("validation_cindex_mean")

rng = np.random.default_rng(config.seed)

# Get data
X, y = datasets(args.dataset)

rkf = RepeatedKFold(n_splits=4, n_repeats=5)
i = 0
mean = []
metrics = {}
for train_index, test_index in rkf.split(X):
    seed = int(rng.integers(1,9999))
    model = make_model(config, seed=seed)
    model.fit(X[train_index, :], y[train_index])
    scoretrain = model.score(X[train_index, :], y[train_index])
    scoreval = model.score(X[test_index, :], y[test_index])
    metrics[""] = {
        "training_cindex": scoretrain,
        "validation_cindex": scoreval
    }
    wandb.log(metrics, step=i)
    mean.append(scoreval)
    i += 1

wandb.log({"validation_cindex_mean": np.nanmean(mean)})
