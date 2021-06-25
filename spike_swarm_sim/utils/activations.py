import numpy as np

def heaviside(x):
    return x * float(x >= 0.0)

def relu(x):
    x[x < 0] = 0.0
    return x

def sigmoid(x):
    return np.divide(1, 1 + np.exp(-x))

def tanh(x):
    return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)

def softmax(x, tau=1):
    return np.exp(x/tau) / np.sum(np.exp(x/tau))