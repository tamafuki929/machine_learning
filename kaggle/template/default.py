# %% initialize
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font = "IPAexGothic")

def load_dataset(path):
    data = pd.read_csv(path, index_col = 0)
    data.columns = data.columns.str.lower()
    d_id = data.index
    return data, d_id


# %% get dataset
train, train_id = load_dataset("/kaggle/input/XXX/train.csv")
test, test_id = load_dataset("/kaggle/input/XXX/test.csv")
original_data = pd.concat([train, test])
original_data["is_test"] = 0
original_data.loc[test_id, "is_test"] = 1


# %% submit
def submit_res(X_test, models, aggregator):
    y_pred = aggregator.predict(models, X_test)
    y_pred = np.where(y_pred < 0.5, 0, 1)
    submit = pd.DataFrame()
    submit["PassengerId"] = X_test.index
    submit["Survived"]    = y_pred
    print(submit)
    submit.to_csv("submission.csv", index = False)