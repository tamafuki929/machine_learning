# %% import general libraries
import numpy as np
import pandas as pd

# %% import libraries for preprocessing
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# %% drop na and use only numerical columns
def make_baseline_dataset(data):
    data = data.select_dtypes(include = np.number)
    data.dropna(axis = 1)
    return data

# %% make pipleline to impute nan and create nan or not column
base_trans = FeatureUnion[
            ("simpleimputer", SimpleImputer(strategy = "median")), 
            ("missingindicator", MissingIndicator())
            ]
ct = ColumnTransformer(
        transfomers = [
            "fillna_and_missindicator", base_trans, ["xxx", "yyy"], 
            "onehotencode", OneHotEncoder(), "xxxx"
        ]
    )

ct.fit_transform(data)