import numpy as np
import xgboost as xgb

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

X, y = load_breast_cancer(return_X_y=True)
X_train = X[:50]
y_train = y[:50]
X_test = X[50:]
y_test = y[50:]

class FocalLoss:

    def __init__(self, gamma, alpha=None):
        self.alpha = alpha
        self.gamma = gamma

    def at(self, y):
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)

    def pt(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    def __call__(self, y_true, y_pred):
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

def floss(y_true, y_pred):
    F = FocalLoss(2, 0.00000000001)
    return F.grad(y_true, y_pred), F.hess(y_true, y_pred)

def logloss(y_true, y_pred):
    return y_pred - y_true, y_pred * (1. - y_pred)

params = {
    "learning_rate": 0.1, 
    "n_estimators": 200, 
    "max_depth": 5, 
    "obj": logloss
}

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("############################### Log Loss ##########################")
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

params = {
    "learning_rate": 0.1, 
    "n_estimators": 200, 
    "max_depth": 5, 
    "obj": floss
}

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("############################### Focal Loss ##########################")
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))