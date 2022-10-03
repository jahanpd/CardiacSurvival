#!/usr/bin/env python3
# feature importance from GBM

import wandb
import numpy as np
import pandas as pd
import json
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from data import datasets
from models import make_model, Imputer
from docx import Document
from argparse import Namespace

ENTITY="cardiac-ml"
PROJECT="survival"

api = wandb.Api()

anzscts_path = './temp/anzscts.pickle'
X, y = datasets("ANZSCTS", anzscts_path)

# Get data
with open('data/data_dictionary.json') as file:
    data_dict = json.load(file)

def get_store(path):
    try:
        with open(path) as file:
            store = json.load(file)
        return store
    except Exception as e:
        print(e)
        return dict()

store_path = "./results/store.json"
store = get_store(store_path)

cat = []
for c in list(X):
    if c in data_dict:
        if data_dict[c]["type"] == "categorical":
            cat.append(0)
        else:
            cat.append(1)
    else:
        cat.append(1)

print(X.shape, y.shape)
assert len(cat) == 0 or len(cat) == X.shape[1]

def get_sweep(entity, project, sweep_id):
    try:
        sweep = api.sweep("{}/{}/{}".format(entity,project,sweep_id))
        return sweep.best_run().config
    except Exception as e:
        print(e)
        return "FAILED"

config = Namespace(**get_sweep(ENTITY, PROJECT, store["gbm"]))

print(config)
model = make_model(
    config, seed=1234, cat="auto" if len(cat)==0 else cat,
    imputer=None
)


imputer = Imputer(cat=cat)
X = imputer.transform(X)


model.fit(X.values, y)

fi = model.model.feature_importances_

fi = list(zip(fi, list(X)))

fi.sort(reverse=True, key=lambda x: x[0])

doc = Document()

table = doc.add_table(rows=1, cols=2)

hdr_cells = table.rows[0].cells
hdr_cells[0].text = 'Feature'
hdr_cells[1].text = 'Score'

print(fi)

for f in fi:
    row_cells = table.add_row().cells
    try:
        row_cells[0].text = data_dict[f[1]]["name"]
    except:
        row_cells[0].text = f[1]

    row_cells[1].text = str(f[0])

doc.save('importance.docx')
