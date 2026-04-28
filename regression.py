import math
import numpy as np
from abc import abstractmethod
from numpy.typing import NDArray

class Regression:
    def __init__(self, X_train: NDArray, y_train: NDArray):
        self.X_train = X_train
        self.y_train = y_train
    
    @abstractmethod
    def compute_f(self, X, w, b):
        pass

    @abstractmethod
    def compute_cost(self, w_in: NDArray, b_in: float, lambda_: float):
        pass

    @abstractmethod
    def compute_gradient(self, w_in: NDArray, b_in: float):
        pass

    def gradient_descent(self, w_in: NDArray, b_in: float, lr: float, epochs: int, lambda_: float): 
        j_history = []
        p_history = []
        
        w = w_in
        b = b_in

        print_step = max(1, math.ceil(epochs / 10))
        for i in range(epochs):
            dj_dw, dj_db = self.compute_gradient(w, b)
            w -= lr * dj_dw
            b -= lr * dj_db

            cost = self.compute_cost(w, b, lambda_)
            j_history.append(cost)
            p_history.append([w, b])
            if i % print_step == 0:
                print(f'{i:10} - Cost: {cost:10.8f}')
        
        return w, b, j_history, p_history
    
    def fit(self, w_in: NDArray, b_in: float, lr: float, epochs: int, lambda_: float = 0):
        w, b, J_history, p_history = self.gradient_descent(w_in, b_in, lr, epochs, lambda_)
        self.w = w
        self.b = b
        return w, b, J_history, p_history

    def predict(self, X: NDArray):
        m, n = X.shape
        p = np.zeros(m)
        for i in range(m):
            p[i] = self.compute_f(X[i], self.w, self.b)

        return p
