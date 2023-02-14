from scipy.special import kl_div
from scipy.stats import binned_statistic

class RidgeAggregation:
    
    def __init__(self):
        self.aggregator = Ridge(alpha = 10.)
    
    def fit(self, models, X, y, params = 1.):
        y_preds = np.zeros((len(X), len(models)))
        for model_i in range(len(models)):
            y_preds[:, model_i] = models[model_i].predict(X)
        self.aggregator = Ridge(alpha = params)
        self.aggregator.fit(y_preds, y)
        
    def tuning_kl_div(self, models, X_train, y_train, X_test, random_state = 42):
        def pred_dist_kl_div(y_train, y_test):
            thresholds = np.arange(0, 1, 0.05)
            train_dist = binned_statistic(y_train, y_train, bins = thresholds)[0]
            test_dist = binned_statistic(y_test, y_test, bins = thresholds)[0]
            print(train_dist)
            return np.sum(kl_div(train_dist, test_dist) + kl_div(test_dist, train_dist))
            
        def objective(trial, X_train, y_train):
            alpha = trial.suggest_loguniform("alpha", 0.01, 100)
            self.fit(models, X_train, y_train, params = alpha)
            y_train = self.predict(models, X_train)
            y_test  = self.predict(models, X_test)
            return pred_dist_kl_div(y_train, y_test)
        
        study = optuna.create_study(direction = "minimize",
                                    sampler   = optuna.samplers.TPESampler(seed = random_state), 
                                    study_name = "ridge_aggregator")
        study.optimize(lambda x: objective(x, X_train, y_train), n_trials=50)
        return study.best_params["alpha"]
    
    def predict(self, models, X):
        y_preds = np.zeros((len(X), len(models)))
        for model_i in range(len(models)):
            y_preds[:, model_i] = models[model_i].predict(X)
        return self.aggregator.predict(y_preds)