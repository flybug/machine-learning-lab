from abc import abstractmethod, ABC
import numpy as np

class Regression(ABC):
    def __init__(self, x: np.array, y: np.array, w: float, b: float):
        self.x = x
        self.y = y
        self.m = x.shape[0]
        self.a = w
        self.b = b

    @abstractmethod
    def fit(self, lr: float, epochs: int):
        pass

    @abstractmethod
    def predict(self, x):
        pass