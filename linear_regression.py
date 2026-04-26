import math
import numpy as np
from numpy.typing import NDArray

from regression import Regression

class LinearRegression(Regression):

    def __init__(self, X_train: NDArray, y_train: NDArray):
        super().__init__(X_train, y_train)
    

    def compute_f(self, X, w, b):
        return np.dot(w, X) + b

    def compute_cost(self, w: NDArray, b: float, lambda_: float):
        x = self.X_train
        y = self.y_train
        m = x.shape[0]
        cost_sum = 0
        for i in range(m):
            f_wb = self.compute_f(x[i], w, b)
            loss = (f_wb - y[i]) ** 2
            cost_sum += loss
        
        cost = cost_sum / (2 * m)
        return cost

    def compute_gradient(self, w: NDArray, b: float):
        x = self.X_train
        y = self.y_train
        m = x.shape[0]
        dj_dw = 0
        dj_db = 0
        for i in range(m):
            f_wb = self.compute_f(x[i], w, b)
            dj_dw_i = (f_wb - y[i]) * x[i]
            dj_db_i = f_wb - y[i]
            dj_dw += dj_dw_i
            dj_db += dj_db_i
        
        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return dj_dw, dj_db
