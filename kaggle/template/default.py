import numpy as np 
import pandas as pd 
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
sns.set(font = "IPAexGothic")
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score
from sklearn.linear_model import Ridge
from sklearn.impute import MissingIndicator
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce
import lightgbm as lgb
import optuna

def load_dataset(path):
    data = pd.read_csv(path, index_col = 0)
    data.columns = data.columns.str.lower()
    d_id = data.index
    return data, d_id

train, train_id = load_dataset("/kaggle/input/titanic/train.csv")
test, test_id = load_dataset("/kaggle/input/titanic/test.csv")
original_data = pd.concat([train, test])
original_data["is_test"] = 0
original_data.loc[test_id, "is_test"] = 1


def submit_res(X_test, models, threshold = 0.5, aggregator = None):
    y_pred = pred_models(models, X_test, aggregator)
    y_pred = np.where(y_pred < threshold, 0, 1)
    submit = pd.DataFrame()
    submit["PassengerId"] = X_test.index
    submit["Survived"]    = y_pred
    print(submit.head(10))
    submit.to_csv("submission.csv", index = False)