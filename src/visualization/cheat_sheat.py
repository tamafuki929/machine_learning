# %% import liblaries
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn 

# %% data loading
data = pd.read_csv("../../dataset/titanic_train.csv", index_col=0)


# %% plot all dist, hue: base column
sns.pairplot(data, hue = "Survived")

# %% plot heatmap
sns.heatmap(data.corr(), annot = True)