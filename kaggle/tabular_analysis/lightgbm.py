# %% 
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score
from tqdm import tqdm
import lightgbm as lgb
import optuna


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


def random_holdout_lgbcls(X, y, cv, params = {}, n_iter = 10, random_state = 42):
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
    for i in range(n_iter):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.1, 
                                                              stratify = y, 
                                                              random_state = 42 + i)
    
        dtrain = lgb.Dataset(X_train, label = y_train)
        dvalid = lgb.Dataset(X_valid, label = y_valid, free_raw_data = False)
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
def manual_cv_lgbcls(X, y, cv, params = {}, random_state = 42):
    _params = {
        "objective"    : "binary", # or multiclass, 
        # "num_class"  : 3, 
        "boosting_type": "gbdt", 
        "metrics"      : "binary_logloss", # multi_logloss
        "random_state" : random_state, 
        "num_leaves"    : 20,
        "min_data_in_leaf": 20, 
        "learning_rate": 0.01, 
        "lambda_l2": 10.,
        # "is_unbalance": True,
        "verbose"      : -1
    }
    _params.update(params)
    
    scores = {
        "roc_auc" : [], 
        "accuracy": []
    }
    models = []
    for _, (train_id, test_id) in tqdm(enumerate(cv.split(X, y))):
        X_train, X_valid, y_train, y_valid = train_test_split(X.iloc[train_id], 
                                                              y.iloc[train_id], 
                                                              test_size = 0.1,
                                                              random_state = random_state)
        dtrain = lgb.Dataset(X_train, label = y_train)
        dvalid = lgb.Dataset(X_valid, label = y_valid)
        dtest = lgb.Dataset(X.iloc[test_id], label = y.iloc[test_id], free_raw_data = False)
        
        # if there is no improvement in dtrain or dvalid, execute early stopping
        model = lgb.train(
                _params, dtrain, 
                valid_sets = [dvalid, dtest], 
                num_boost_round = 1000, 
                callbacks=[
                    lgb.early_stopping(stopping_rounds=10, first_metric_only = True, verbose=True), 
                    lgb.log_evaluation(period=100)
                ]
        )
        
        y_pred = model.predict(dtest.get_data())
        auc_score = roc_auc_score(dtest.get_label(), y_pred)
        accuracy = accuracy_score(dtest.get_label(), np.where(y_pred < 0.5, 0, 1))
        
        scores["roc_auc"].append(auc_score)
        scores["accuracy"].append(accuracy)
        models.append(model)
    
    print("validation result (auc):", np.mean(scores["roc_auc"]))
    print("validation result (acc):", np.mean(scores["accuracy"]))
    return scores, models



# get feature importance from multiple models
def models_feature_importance(models, columns, importance_type = "gain"):
    fi_array = np.zeros(len(columns))
    for model in models:
        fi_array += model.feature_importance(importance_type=importance_type)
    fi_df = pd.DataFrame(fi_array / len(models), 
                         index = columns, columns=['importance'])
    return fi_df.sort_values(by = "importance", ascending = False)
    
# hyperparameter tuning in lightGBM
def tuning_lightgbm(X, y, data_transformer, random_state = 42):
    cv = KFold(n_splits = 10, shuffle = True, random_state = 42)
    
    def objective(trial, X, y):
        params = {
            "num_boost_round" : trial.suggest_categorical("num_boost_round", [2000]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 1000, step=20), 
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 1000, step=100), 
            "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5), 
            "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5), 
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15), 
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0, step = 0.1),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1)
        }
        scores, models = manual_cv_lgbcls(X, y, cv, data_transformer, params = params)
        return np.mean(scores)
    
    study = optuna.create_study(
            direction = "maximize", # or minimize
            sampler   = optuna.samplers.TPESampler(seed = random_state), 
            study_name = "lgb classifier")
    func  = lambda trial: objective(trial, X, y)
    study.optimize(func, n_trials=500)
    
    return study

# %%
def pred_models(models, X, aggregator = None):
    if aggregator is None:
        y_pred = np.array([0.] * len(X))
        for model in models:
            y_pred += model.predict(X)
        return y_pred / len(models)
    else:
        return aggregator.predict(models, X)