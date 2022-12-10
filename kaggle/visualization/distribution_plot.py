# %% import liblaries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %% plot continuous distribution
def dist_con_plot(data, columns, n_col = 3, hue = None, figsize = (15, 5)):
    if n_col > len(columns): n_col = len(columns)
    fig, ax = plt.subplots(int(np.ceil(len(columns)/n_col)), n_col, figsize = figsize)
    ax = ax.flatten()
    for i in range(len(columns)):
        skew = data[columns[i]].skew()
        sns.histplot(data = data, x = columns[i], hue = hue, kde=True, stat='density', common_norm = False, ax=ax[i])
        ax[i].set_title("Variable %s skew : %.4f"%(columns[i], skew))
    # plt.tight_layout()
    plt.show()

# %% plot discrete distribution
def dist_cat_plot(data, columns, n_col = 3, hue = None, figsize = (15, 15)):
    if n_col > len(columns): n_col = len(columns)
    fig, ax = plt.subplots(int(np.ceil(len(columns)/n_col)), n_col, figsize = figsize)
    ax = ax.flatten()
    for i in range(len(columns)):
        sns.histplot(data = data, x = columns[i], hue = hue, stat = "probability", common_norm = False, ax=ax[i])
    # plt.tight_layout()
    plt.show()