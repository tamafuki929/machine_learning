# %% import liblaries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %% plot continuous distribution
def dist_con_plot(data, columns, n_col = 3, hue = None, figsize = None):
    if n_col > len(columns): n_col = len(columns)
    n_row = int(np.ceil(len(columns)/n_col))
    if figsize is None:
        figsize = (n_col * 5, n_row * 4)
    fig, ax = plt.subplots(n_row, n_col, figsize = figsize)
    ax = ax.flatten()
    for i in range(len(columns)):
        skew = data[columns[i]].skew()
        sns.histplot(data = data, x = columns[i], hue = hue, kde=True, stat='density', common_norm = False, ax=ax[i])
    # plt.tight_layout()
    plt.show()

# %% plot discrete distribution
def dist_cat_plot(data, columns, n_col = 3, hue = None, figsize = None):
    if n_col > len(columns): n_col = len(columns)
    n_row = int(np.ceil(len(columns)/n_col))
    if figsize is None:
        figsize = (n_col * 5, n_row * 4)
    fig, ax = plt.subplots(int(np.ceil(len(columns)/n_col)), n_col, figsize = figsize)
    ax = ax.flatten()
    for i in range(len(columns)):
        sns.histplot(data = data, x = columns[i], hue = hue, stat = "probability", common_norm = False, ax=ax[i])
    # plt.tight_layout()
    plt.show()
    
## plot binning result
def binnin_plot(data, col, hue, nbins = 10):
    processed_data = data.copy()
    processed_data[col] = pd.qcut(processed_data[col], nbins, 
                        labels = [i for i in range(nbins)])
    sns.histplot(data = processed_data, x = col, 
                 hue = hue, kde=True, stat="probability", common_norm = False)
    
# plot prediction distribution
def plot_pred_dist(models, X_train, X_test, proba = True, aggregator = None, bins = np.arange(0, 1, 0.05)):
    def pred_models(models, X, aggregator = None):
        if aggregator is None:
            y_pred = np.array([0.] * len(X))
            for model in models:
                y_pred += model.predict(X)
            return y_pred / len(models)
        else:
            return aggregator.predict(models, X)
    y_pred_test = pd.DataFrame(pred_models(models, X_test, aggregator), 
                               index = X_test.index, columns = ["proba"])
    y_pred_train = pd.DataFrame(pred_models(models, X_train, aggregator), 
                                index = X_train.index, columns = ["proba"])
    y_pred_df = pd.concat([y_pred_test, y_pred_train])
    y_pred_df["pred"] = (y_pred_df["proba"] > 0.5).astype(int)
    y_pred_df["is_test"] = 0
    y_pred_df.loc[y_pred_test.index, "is_test"] = 1
    if proba:
        sns.histplot(data = y_pred_df, x = "proba", hue = "is_test", bins = bins, 
                     stat = "proportion", common_norm = False)
    else:
        sns.histplot(data = y_pred_df, x = "pred", hue = "is_test", 
                     stat = "proportion", common_norm = False)