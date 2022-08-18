import argparse
import wandb
import numpy as np
import pandas as pd
import json
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
# k fold params
parser.add_argument("--k", type=int, default=4) 
parser.add_argument("--repeats", type=int, default=5) 
# check for bias
parser.add_argument("--bias", action='store_true')

# Get the experiment info, hyperparameters, and seed
args = parser.parse_args()

# Pass them to wandb.init
wandb.init(config=args, name=args.dataset, entity="cardiac-ml", project="survival")
# Access all hyperparameter values through wandb.config
config = wandb.config

# define recorded metrics
wandb.define_metric("training_cindex", summary="mean")
wandb.define_metric("validation_cindex", summary="mean")

rng = np.random.default_rng(config.seed)

# Get data
with open('data/data_dictionary.json') as file:
    data_dict = json.load(file)
    
anzscts_path = './temp/anzscts.pickle'
X, y = datasets(args.dataset, anzscts_path)
cat = []
if args.dataset == "ANZSCTS":
    for c in list(X):
        if c in data_dict:
            if data_dict[c]["type"] == "categorical":
                cat.append(0)
            else:
                cat.append(1)
        else:
            cat.append(1)
print(X.shape)
assert len(cat) == 0 or len(cat) == X.shape[1]
bias = {
    "Sex":{'values': ['Male', 'Female']},
    "Race1":{'values': ['No', 'Yes']}
}
for k in bias.keys():
    bias[k]["index"] = list(X).index(k)
    wandb.define_metric("{}_cindex".format(k), summary="mean")

rkf = RepeatedKFold(n_splits=args.k, n_repeats=args.repeats)
for train_index, test_index in rkf.split(X):
    seed = int(rng.integers(1,9999))
    model = make_model(config, seed=seed, cat="auto" if len(cat)==0 else cat)
    model.fit(X.values[train_index, :], y[train_index])
    scoretrain = model.score(X.values[train_index, :], y[train_index])
    scoreval = model.score(X.values[test_index, :], y[test_index])
    metrics = {
        "training_cindex": scoretrain,
        "validation_cindex": scoreval
    }
    if args.bias:
        for k in bias.keys():
            Xtest = X.values[test_index, :]
            ytest = y.values[test_index]
            bidx = Xtest[:, bias[k]["index"]] == 1.0
            if bidx.sum() > 1:
                scoreval = model.score(Xtest[bidxd, :], ytest[bidx])
            else:
                scoreval = np.nan
            metrics["{}_cindex".format(k)] = scoreval
            
    wandb.log(metrics)

