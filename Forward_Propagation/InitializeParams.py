import numpy as np

def initialize_params(layer_dims: list) -> dict:

    params = {}
    L = len(layer_dims)

    for l in range(1, L):

        # Implemented the HE system for initialization
        # Which is sqrt of 2/the dimension of previous layer
        # Taken from his research paper.
        # Guy is a genius. The accracy went from 0.445 to 0.845.
        params['W'+ str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * (2/layer_dims[l-1])**(1/2)
        params['b'+ str(l)] = np.zeros((layer_dims[l], 1))

    return params


