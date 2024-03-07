import numpy as np

def initialize_params(layer_dims: list) -> dict:

    params = {}
    L = len(1, layer_dims)

    for l in range(L):

        params['W'+ str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) *0.01
        params['b'+ str(l)] = np.zeros((layer_dims[l], 1))


    return params
