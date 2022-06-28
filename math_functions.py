import numpy as np

def entropy(X):
    '''entropy(arr) -> float
    outputs shannon entropy of prob distribution x, which is represented by the 1d arr'''
    sum = 0
    for x in X:
        if x != 0:
            sum += x * np.log2(x)

    return -sum

def lin_div(p, q):
    '''lin_div(arr, arr) -> float
    outputs average distance for two distributions'''
    div = 0.0
    for i in range(len(p)):
        div += abs(p[i] - q[i])

    return div / len(p)