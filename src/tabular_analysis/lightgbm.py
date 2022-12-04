# %% 
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, r2_score

import lightgbm as lgb
import optuna

# %% baseline to use sllearn or manual cv
def manual_cv_lgbreg(X, y, cv, params = {}, random_state = 42):
    _params = {
        "objective"    : "regression",  
        "boosting_type": "gbdt", 
        "metric"       : "rmse", 
        "random_state" : random_state, 
        "verbose"      : -1
    }
    _params.update(params)
    
    scores = []
    models = []
    for _, (train_id, val_id) in tqdm(enumerate(cv.split(X, y))):
        dtrain = lgb.Dataset(X[train_id], label = y[train_id])
        dvalid = lgb.Dataset(X[val_id]  , label = y[val_id], free_raw_data = False)
        
        # if there is no improvement in dtrain or dvalid, stop learning
        model = lgb.train(
                _params, dtrain, 
                valid_sets = [dtrain, dvalid], 
                num_boost_rounds = 1000, 
                callbacks=[
                    lgb.early_stopping(stopping_rounds=10, verbose=True), 
                    lgb.log_evaluation(period=100)
                ]
        )
        
        y_pred = model.predict(dvalid.get_data())
        r2_score = r2_score(dvalid.get_label(), y_pred)
        
        scores.append(r2_score)
        models.append(model)
    
    print(f"validation result: {np.mean(scores)}")
    return scores, models

cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
manual_cv_lgbreg(X, y, cv = cv)


# %% baseline to use cv func. in lgb library
def lgb_cv_lgvreg(X, y, params = {}, random_state = 42):
    params = {
        "objective"    : "regression", 
        "boosting_type": "gbdt", 
        "metric"       : "rmse", 
        "random_state" : random_state, 
        "verbose"      : -1
    }.update(params)
    
    cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
    cv_data = lgb.Dataset(X, label = y)
    
    cv_result = lgb.cv(
        params, cv_data, 
        num_boost_rounds = 1000, 
        callbacks = [
                    lgb.early_stopping(stopping_rounds=10, verbose=True), 
                    lgb.log_evaluation(period=100)
        ], 
        folds = cv
    )
    print(f'RMSE mean = {cv_result["rmse-mean"][-1]}')
    return cv_result


# %% baseline to use sllearn or manual cv
def manual_cv_lgbcls(X, y, cv, params = {}, random_state = 42):
    _params = {
        "objective"    : "binary", # or multiclass, 
        # "num_class"  : 3, 
        "boosting_type": "gbdt", 
        "metrics"      : "binary_logloss", # multi_logloss
        "random_state" : random_state, 
        "verbose"      : -1
    }
    _params.update(params)
    
    scores = []
    models = []
    for _, (train_id, val_id) in tqdm(enumerate(cv.split(X, y))):
        dtrain = lgb.Dataset(X[train_id], label = y[train_id])
        dvalid = lgb.Dataset(X[val_id]  , label = y[val_id], free_raw_data = False)
        
        # if there is no improvement in dtrain or dvalid, stop learning
        model = lgb.train(
                _params, dtrain, 
                valid_sets = [dtrain, dvalid], 
                num_boost_round = 1000, 
                callbacks=[
                    lgb.early_stopping(stopping_rounds=10, verbose=True), 
                    lgb.log_evaluation(period=100)
                ]
        )
        
        y_pred = model.predict(dvalid.get_data())
        auc_score = roc_auc_score(dvalid.get_label(), y_pred)
        
        scores.append(auc_score)
        models.append(model)
    
    print(f"validation result: {np.mean(scores)}")
    return scores, models


# %% baseline to use cv func. in lgb library
def lgb_cv_lgbcls(X, y, params = {}, random_state = 42):
    _params = {
        "objective"    : "binary", # or multiclass, 
        # "num_class"  : 3,  
        "boosting_type": "gbdt", 
        "metric"       : "binary_logloss", # logloss, multi_logloss 
        "random_state" : random_state, 
        "verbose"      : -1
    }
    _params.update(params)
    
    cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
    cv_data = lgb.Dataset(X, label = y)
    
    cv_result = lgb.cv(
        _params, cv_data, 
        num_boost_round = 1000, 
        folds = cv, 
        callbacks = [
            lgb.early_stopping(stopping_rounds=10, verbose=True), 
            lgb.log_evaluation(period=100)
        ]
    )
    print(f'logloss mean = {cv_result["binary_logloss-mean"][-1]}')
    return cv_result
    
# hyperparameter tuning in lightGBM
def tuning_lightgbm(X, y, random_state = 42):
    
    def objective(trial, X, y):
        params = {
            "n_estimators" : trial.suggest_categorical("n_estimators", [2000]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20), 
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 1020, step=100), 
            "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5), 
            "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5), 
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15), 
            "bagging_freq": trial.suggest_float("bagging_freq", 0.5, 1.0, step = 0.1),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1),
        }
        cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
        scores, models = manual_cv_lgbcls(X, y, cv = cv)
        return np.mean(scores)
    
    study = optuna.create_study(
            direction = "minimize", # or ma 
            sampler   = optuna.samplers.TPESampler(seed = random_state), 
            study_name = "lgb classifier")
    func  = lambda trial: objective(trial, X, y)
    study.optimize(func, n_trials=50)
    
    return study
# %%
