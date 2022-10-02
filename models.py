import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
# from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.tree import SurvivalTree
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.metrics import concordance_index_censored
import xgboost as xgb
import miceforest as mf
from miceforest import mean_match_fast_cat
from numpy.random import default_rng


class XGBSurv:
    def __init__(
        self,
        params,
        num_round
        ):
        self.params = params
        self.num_round = num_round
        self.bst = None
    def fit(self, X, y):
        label = np.array([-x[1] if x[0] else x[1] for x in y])
        dtrain = xgb.DMatrix(X, label=label)
        self.bst = xgb.train(self.params, dtrain, self.num_round)

    def predict(self, X):
        assert self.bst is not None
        dtest = xgb.DMatrix(X)
        return self.bst.predict(dtest, output_margin=True)

    def score(self, X, y):
        assert self.bst is not None
        risk = self.predict(X) * -1
        return concordance_index_censored(
            y["event"],
            y["days"],
            risk
        )[0]

    def feature_importance(self):
        assert self.bst is not None
        return self.bst.get_score(importance_type='gain')

mean_match = mean_match_fast_cat.copy()
mean_match.set_mean_match_candidates(3)

class FeatureSelect(BaseEstimator, TransformerMixin):
    def __init__(self, correlation=True, cat=None):
        assert cat is not None, "need array of values for categorical feature to feature select"
        self.corrbool = correlation
        self.cat = cat
        self.mi = None
        self.corr = None
        self.selected = None

    def fit(self, X, y):
        # get mi and corr matrix
        assert X.shape[1] == len(self.cat), "cat must be array of size nfeat but is {}".format(self.cat)
        idx = np.arange(X.shape[1])
        rs = np.corrcoef(X, rowvar=False)
        mi = mutual_info_classif(X, y["event"].astype(float),
                                 discrete_features=[x == 1 for x in self.cat])
        if self.corrbool:
            selected = []
            for i in idx:
                if i in selected:
                    continue
                bool = rs[i, :] > 0.6
                if np.sum(bool) == 0:
                    selected.append(i) 
                else:
                    sub = np.flatnonzero(bool)
                    submi = mi[sub]
                    best = np.argmax(submi)
                    if best not in selected:
                        selected.append(sub[best])
            self.mi = mi
            self.corr = rs
            self.selected = np.array(selected)
            print("selected {} features".format(len(selected)))
        else:
            bool = mi > 1e-5
            self.selected = np.flatnonzero(bool)
            self.corr = rs
            self.mi = mi
            print("selected {} features".format(len(self.selected)))

    def transform(self, X):
        if self.selected is None:
            assert False, 'Must fit feature select before call transform'
        return X[:, self.selected]

class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=1234, rounds=1, cat="auto"):
        self.random_state=random_state
        self.kds = None
        self.rounds = rounds
        print("cat is not auto = {}".format(cat == "auto"))
        self.cat = cat
                           
    def fit(self, X):
        self.kds = mf.ImputationKernel(
          X,
          datasets=1,
          save_all_iterations=True,
          random_state=int(self.random_state),
          data_subset=5,
          categorical_feature=self.cat,
          mean_match_scheme=mean_match
        )
        print("Imputing")
        self.kds.mice(self.rounds)
        return self

    def transform(self, X):
        try:
            if self.kds is None:
                self.kds = mf.ImputationKernel(
                  X,
                  datasets=1,
                  save_all_iterations=True,
                  random_state=int(self.random_state)
                )
                print("Imputing")
                self.kds.mice(
                    self.rounds,
                    verbose=True,
                    compile_candidates=True,
                    max_depth=10,
                    n_estimators=50,
                    num_leaves=50,
                    max_bin=20,
                    num_iterations=100,
                    verbosity=-1,
                    force_row_wise=True
                )
            Xp = self.kds.impute_new_data(new_data=X)
            Xpf = Xp.complete_data(dataset=0, inplace=False)
            return Xpf
        except Exception as e:
            print("Imputation failed")
            print(e)
            return X


class Oversampler(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=1234):
        self.random_state=random_state
        self.sampler = RandomOverSampler(random_state=random_state)
                           
    def fit(self, X, y):
        return self

    def transform(self, X, y):
        ytimes = [i[1] for i in y]
        _, bins = np.histogram(ytimes)
        ybinned = np.digitize(ytimes, bins)
        idx = np.arange(X.shape[0])
        idx, _ = self.sampler.fit_resample(idx.reshape(-1,1), ybinned)
        return X[idx.flatten(), :], y[idx.flatten()]

class Pipeline:
    def __init__(
            self,
            feature_selection = FeatureSelect,
            correlation = True,
            scaler = StandardScaler,
            imputer = Imputer,
            oversampler = Oversampler,
            model = CoxnetSurvivalAnalysis(),
            rng = np.random.default_rng(69),
            cat = "auto"
        ):
        if feature_selection is not None:
            self.features = FeatureSelect(correlation=correlation, cat=cat)
        else:
            self.features = None
        if scaler is not None:
            self.scaler = scaler()
        else:
            self.scaler = None
        if imputer is not None:
            self.imputer = imputer(random_state=rng.integers(1, 9999), cat=cat)
        else:
            self.imputer = None
        self.oversampler = oversampler(random_state=rng.integers(1,9999))
        self.model = model

    def fit(self, X, y):
        if self.imputer is not None:
            X = self.imputer.transform(X)
        if self.features is not None:
            print("feature select")
            self.features.fit(X, y)
            X = self.features.transform(X)
            print("features selected {}".format(X.shape))
        if self.scaler is not None:
            print("scaling")
            X = self.scaler.fit_transform(X)
        print("oversampling")
        X, y = self.oversampler.transform(X, y)
        print("fitting model")
        self.model.fit(X, y)
        print("model fitted") 

    def predict(self, X):
        if self.imputer is not None:
            X = self.imputer.transform(X)
        if self.features is not None:
            print("feature select")
            X = self.features.transform(X)
            print("features selected {}".format(X.shape))
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def score(self, X, y):
        if self.imputer is not None:
            X = self.imputer.transform(X)
        if self.features is not None:
            print("feature select")
            X = self.features.transform(X)
            print("features selected {}".format(X.shape))
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.score(X, y)

def make_model(config, seed=69, cat="auto", imputer=None):
    rng = np.random.default_rng(seed)
    if config.model == 'cox':
        model = CoxnetSurvivalAnalysis(
            l1_ratio=config.alpha, tol=1e-7
        )
        pipeline = Pipeline(
            model=model, cat=cat, imputer=imputer)
    elif config.model == 'dt':
        model = SurvivalTree(
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            min_weight_fraction_leaf=config.min_weight_fraction_leaf,
            random_state=rng.integers(1,9999)
        )
        pipeline = Pipeline(
            model=model, cat=cat, correlation=False, scaler=None, imputer=imputer)
    elif config.model == 'rf':
        model = RandomSurvivalForest(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            min_weight_fraction_leaf=config.min_weight_fraction_leaf,
            max_samples=config.max_samples,
            random_state=rng.integers(1,9999),
            verbose=1,
            n_jobs=-1
        )
        pipeline = Pipeline(
            model=model, cat=cat, feature_selection=None, scaler=None, imputer=imputer)
    elif config.model == 'gbm':
        model = GradientBoostingSurvivalAnalysis(
            learning_rate=config.learning_rate,
            loss=config.loss,
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            min_weight_fraction_leaf=config.min_weight_fraction_leaf,
            subsample=config.subsample,
            random_state=rng.integers(1,9999),
            verbose=1,
        )
        pipeline = Pipeline(
            model=model, cat=cat, feature_selection=None, scaler=None, imputer=imputer)
    elif config.model == 'xgboost':
        model = XGBSurv(
            params = dict(
            max_depth=0,
            # max_depth=config.max_depth,
            eta = config.learning_rate,
            subsample=config.subsample,
            objective = "survival:cox",
            seed=rng.integers(1,9999),
            tree_method="hist"
            ),
            num_round = config.n_estimators
        )
        pipeline = Pipeline(
            model=model, cat=cat, feature_selection=None, scaler=None, imputer=imputer)
    elif config.model == 'svm':
        model = FastKernelSurvivalSVM(
            alpha=config.alpha,
            kernel=config.kernel,
            random_state=rng.integers(1,9999),
            verbose=True
        )
        pipeline = Pipeline(
            model=model, cat=cat, correlation=False, scaler=None, imputer=imputer)
    else:
        pipeline = None
    return pipeline


