import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn 


import scipy.stats as st
import itertools
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

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


def corr_mutual_matrix(df, random_state = 42):
    con_cols = df.select_dtypes(include = np.number).columns.to_list()
    cat_cols = df.select_dtypes(exclude = np.number).columns.to_list()
    columns = con_cols + cat_cols
    corr = pd.DataFrame(np.identity(len(columns)), index = columns, columns = columns)
    for col in columns:
        target_df = df[col].dropna()
        if target_df.dtype == np.number:
            mi = mutual_info_regression(target_df.values.reshape(-1, 1), 
                                        target_df.values, random_state=random_state)
        else:
            mi = mutual_info_classif(target_df.astype(np.number).values.reshape(-1, 1), 
                                     target_df.values, random_state = random_state)
        corr.loc[col, col] = mi
    for cols in itertools.combinations(columns, 2):
        target_df = df[list(cols)].dropna()
        if target_df.iloc[:, 1].dtype == np.number:
            mi = mutual_info_regression(target_df[cols[0]].values.reshape(-1, 1), 
                                        target_df[cols[1]].values, 
                                        random_state=random_state)
        else:
            mi = mutual_info_classif(target_df[cols[0]].astype(np.number).values.reshape(-1, 1), 
                                     target_df[cols[1]].values, 
                                     random_state = random_state)
            corr.loc[cols[0], cols[1]] = 2 * mi
            corr.loc[cols[0], cols[1]] /= corr.loc[cols[0], cols[0]] + corr.loc[cols[1], cols[1]]
    corr += corr.T - np.diag(np.diag(corr))
    for col in columns:
        corr.loc[col, col] = 1.
    return pd.DataFrame(corr, index = columns, columns = columns)

def feature_selection_mutual_info(df, th = 0.9, random_state = 42):
    while True:
        mutual_list = corr_mutual_matrix(df, random_state = random_state).sum(axis = 1) - 1.
        max_mi, max_mi_id = mutual_list.max(), mutual_list.idxmax()
        if max_mi > th: 
            df = df.drop(max_mi_id, axis = 1)
        else:
            break
    return corr_mutual_matrix(df, random_state = random_state)