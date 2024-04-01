import numpy as np

def compute_layer(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray):

    Z = np.dot(W, A_prev) + b

    cache = [A_prev, W, b]

    return Z, cache
