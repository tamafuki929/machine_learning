# %% import libraries
import numpy as np 
import pandas as pd 
import os
import sys

# machine learning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font = 'IPAexGothic')

# deep learning
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')
from timm import create_model
from fastai.vision.all import *


# %%
set_seed(42, reproducible=True)
dir_path = Path("../input/petfinder-pawpularity-score")

im = Image.open(train_df['path'][6])
width, height = im.size
print(width,height)

dls = ImageDataLoaders.from_df(train_df, #pass in train DataFrame
                               valid_pct=0.2, #80-20 train-validation random split
                               seed=999, #seed
                               fn_col='path', #filename/path is in the second column of the DataFrame
                               label_col='pawpularity', #label is in the first column of the DataFrame
                               y_block=RegressionBlock, #The type of target
                               bs=32, #pass in batch size
                               num_workers=8,
                               item_tfms=Resize(224), #pass in item_tfms
                               batch_tfms=setup_aug_tfms([Brightness(), Contrast(), Hue(), Saturation()]))