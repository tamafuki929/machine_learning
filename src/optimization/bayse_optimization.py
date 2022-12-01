import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel, RBF, ConstantKernel

from scipy.stats import norm

from sklearn.datasets import fetch_california_housing

def gpr(X_train, y_train):
    return GaussianProcessRegressor(kernel = Matern()).fit(X_train, y_train)

def baysian_optimization(X_train, y_train, X, y):
    
    print("############# model train #################")
    eps = np.std(y_train)
    
    i = 0
    print("############### obt ############")
    while len(X) != 0:
        model = gpr(X_train, y_train)
        y_pred, std_y = model.predict(X, return_std = True)
        y_max = max(y_train)
        
        if i % 10 == 0:
            print("#################### iteration: ", i)
            print("max y", y_max)

        z = (y_pred - y_max - eps) / std_y
        EI = (y_pred - y_max - eps) * norm.cdf(z) + std_y * norm.pdf(z)
        
        opt_ind = np.argmax(EI)
        if i % 10 == 0:
            print("opt x:", X[opt_ind])
            print("y for opt x:", y[opt_ind], "y_pred for opt x:", y_pred[opt_ind])
            print("score:", EI[opt_ind])
        
        X_train = np.vstack((X_train, X[opt_ind]))
        y_train = np.append(y_train, y[opt_ind])
        X = np.delete(X, opt_ind, 0)
        y = np.delete(y, opt_ind)
        
        i += 1
        
def main():
    X, y = fetch_california_housing(return_X_y = True)
    print(max(y), len(y[y == max(y)]))
    X_train = np.array(X[y < 2])
    y_train = np.array(y[y < 2])
    X = np.array(X[y >= 4])
    y = np.array(y[y >= 4])
    
    baysian_optimization(X_train[:100], y_train[:100], X[:200], y[:200])

if __name__ == "__main__":
    main()
