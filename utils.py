import numpy as np

def load_data(file_name):
    data = np.loadtxt(file_name, delimiter=',')
    X = data[:,:-1]
    y = data[:,-1]
    return X, y