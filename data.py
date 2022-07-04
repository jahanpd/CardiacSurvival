import numpy as np
import pandas as pd
from sksurv.preprocessing import OneHotEncoder
from sksurv.datasets import load_breast_cancer

def datasets(name="test"):
    if name == "test":
        X, y = load_breast_cancer()
        Xt = OneHotEncoder().fit_transform(X)
        return Xt.values, y
    elif name == "ANZSCTS":
        raise NotImplementedError("ANZSCTS not implemented yet")
    else:
        raise NotImplementedError("Dataset is not implemented yet")
