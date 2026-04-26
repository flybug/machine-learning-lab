import numpy as np

def load_data(file_name):
    data = np.loadtxt(file_name, delimiter=',')
    X = data[:,:-1]
    y = data[:,-1]
    return X, y

def load_data_npy():
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    X = X[0:1000]
    y = y[0:1000]
    return X, y

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
