import argparse
from data import datasets
from models import make_model
import miceforest as mf

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


# test datasets
X, y = datasets(name="test")

print(X.shape, y.shape)
X = mf.ampute_data(X,perc=0.25,random_state=1991)
# test model
args = parser.parse_args()
model = make_model(args, args.seed)
model.fit(X, y)
c = model.score(X, y)
print(c)
