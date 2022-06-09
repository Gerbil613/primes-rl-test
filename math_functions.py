import numpy as np

def entropy(X):
    '''entropy(arr) -> float
    outputs shannon entropy of prob distribution x, which is represented by the 1d arr'''
    sum = 0
    for x in X:
        if x != 0:
            sum += x * np.log2(x)

    return -sum

def klr_div(p, q):
    '''klr_div(arr, arr) -> float
    outputs Kullback-Leibler divergence rate between two distributions'''
    pass