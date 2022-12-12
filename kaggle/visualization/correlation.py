import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn 


import scipy.stats as st
import itertools

# %% plot correlation
def corr(data):
    sns.heatmap(data.corr(), annot = True, vmax = 1.0, vmin = -1., cmap = "coolwarm")

# calculate cramer's V
def cramer_v(x_cat, y_cat):
    cross_tab = pd.crosstab(x_cat, y_cat)
    chi2, _, _, _ = st.chi2_contingency(cross_tab, False)
    n = cross_tab.sum().sum()
    r = np.sqrt(chi2 / (n * (np.min(cross_tab.shape) - 1)))
    return r

# calculate correlation ratio (between cat and con)
def correlation_ratio(x_cat, y_con):
    cross_tab = pd.pivot_table(pd.concat([x_cat, y_con], axis = 1), 
                               index = x_cat.name, 
                               values = y_con.name, 
                               aggfunc = [len, "mean"])

    # calculate total sum of deviation squares
    total_mean = y_con.dropna().mean()
    total_var  = ((y_con - total_mean)**2).sum()

    # calculate correlation
    cross_tab["div_sum"] = cross_tab["len"] * (cross_tab["mean"] - total_mean)**2
    r = cross_tab["div_sum"].sum() / total_var
    return r

def corr_cats(data, cat_cols):
    corr = np.identity(len(cat_cols))
    for inds in itertools.combinations(range(len(cat_cols)), 2):
        r = cramer_v(data[cat_cols[inds[0]]], data[cat_cols[inds[1]]])
        corr[inds[0], inds[1]] = r
    corr += corr.T - np.identity(len(cat_cols))
    corr_df = pd.DataFrame(corr, index = cat_cols, columns = cat_cols)
    return corr_df
    
def corr_cat_vs_con(data, cat_cols, con_cols):
    corr = np.zeros((len(cat_cols), len(con_cols)))
    for cat_i in range(len(cat_cols)):
        for con_i in range(len(con_cols)):
            r = correlation_ratio(data[cat_cols[cat_i]], data[con_cols[con_i]])
            corr[cat_i, con_i] = r
    corr_df = pd.DataFrame(corr, index = cat_cols, columns = con_cols)
    return corr_df