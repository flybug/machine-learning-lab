import numpy as np
from numpy.typing import NDArray

from regression import Regression

class LogisticRegression(Regression):
    
    def __init__(self, X_train: NDArray, y_train: NDArray):
        super().__init__(X_train, y_train)
    
    def sigmoid(self, z: NDArray):
        g = 1 / (1 + np.exp(-z))
        return g

    def compute_f(self, X, w, b):
        z = np.dot(w, X) + b
        return self.sigmoid(z)


    def compute_cost(self, w: float, b: float, lambda_: float):
        x = self.X_train
        y = self.y_train
        m = x.shape[0]
        cost_sum = 0
        for i in range(m):
            f_wb = self.compute_f(x[i], w, b)
            loss = (-y[i] * np.log(f_wb)) - (1 - y[i]) * np.log(1 - f_wb)
            cost_sum += loss
        
        cost_without_reg = cost_sum / m
        reg_cost = sum(np.square(w))
        cost = cost_without_reg + (lambda_ / (2 * m)) * reg_cost
        return cost

    def compute_gradient(self, w: NDArray, b: float):
        x = self.X_train
        y = self.y_train
        m, n = x.shape
        dj_dw = np.zeros(w.shape)
        dj_db = 0.
        for i in range(m):
            f_wb_i = self.compute_f(x[i], w, b)
            err_i = f_wb_i - y[i]
            dj_db += err_i
            for j in range(n):
                dj_dw[j] += err_i * x[i, j]
        
        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return dj_dw, dj_db
    