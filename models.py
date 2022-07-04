import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
# from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.tree import SurvivalTree
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.svm import FastKernelSurvivalSVM
import miceforest as mf
from numpy.random import default_rng

class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=1234, rounds=1):
        self.random_state=random_state
        self.kds = None
        self.rounds = rounds
                           
    def fit(self, X, y=None):
        self.kds = mf.ImputationKernel(
          X,
          datasets=1,
          save_all_iterations=True,
          random_state=int(self.random_state)
        )
        print("Imputing")
        self.kds.mice(self.rounds)
        return self

    def transform(self, X, y=None):
        try:
            if self.kds is None:
                self.kds = mf.ImputationKernel(
                  X,
                  datasets=1,
                  save_all_iterations=True,
                  random_state=int(self.random_state)
                )
                print("Imputing")
                self.kds.mice(self.rounds)
            Xp = self.kds.impute_new_data(new_data=X)
            Xpf = Xp.complete_data(dataset=0, inplace=False)
            return Xpf, y
        except Exception as e:
            print(e)
            return X, y


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
            scaler = StandardScaler,
            imputer = Imputer,
            oversampler = Oversampler,
            model = CoxPHSurvivalAnalysis(),
            rng = np.random.default_rng(69)
        ):
        self.scaler = scaler()
        self.imputer = imputer(random_state=rng.integers(1, 9999))
        self.oversampler = oversampler(random_state=rng.integers(1,9999))
        self.model = model

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        X, y = self.imputer.transform(X, y)
        X, y = self.oversampler.transform(X, y)
        self.model.fit(X, y)

    def predict(self, X):
        X = self.scaler.transform(X)
        X, _ = self.imputer.transform(X, y)
        return self.model.predict(X)
    
    def score(self, X, y):
        X = self.scaler.transform(X)
        X, _ = self.imputer.transform(X, y)
        return self.model.score(X, y)

def make_model(config, seed=69):
    rng = np.random.default_rng(seed)
    if config.model == 'cox':
        model = CoxPHSurvivalAnalysis(
            alpha=config.alpha,
        )
        pipeline = Pipeline(model=model)
    elif config.model == 'dt':
        model = SurvivalTree(
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            min_weight_fraction_leaf=config.min_weight_fraction_leaf,
            random_state=rng.integers(1,9999)
        )
        pipeline = Pipeline(model=model)
    elif config.model == 'rf':
        model = RandomSurvivalForest(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            min_weight_fraction_leaf=config.min_weight_fraction_leaf,
            random_state=rng.integers(1,9999)
        )
        pipeline = Pipeline(model=model)
    elif config.model == 'gbm':
        model = GradientBoostingSurvivalAnalysis(
            learning_rate=config.learning_rate,
            loss=config.loss,
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            min_weight_fraction_leaf=config.min_weight_fraction_leaf,
            random_state=rng.integers(1,9999)
        )
        pipeline = Pipeline(model=model)
    elif config.model == 'svm':
        model = FastKernelSurvivalSVM(
            alpha=config.alpha,
            kernel=config.kernel,
            random_state=rng.integers(1,9999)
        )
        pipeline = Pipeline(model=model)
    else:
        pipeline = None
    return pipeline


