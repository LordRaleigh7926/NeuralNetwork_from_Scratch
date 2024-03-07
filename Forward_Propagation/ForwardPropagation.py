from LayerActivation import activate_layer
import numpy as np


def forward_propagation(X: np.ndarray, params: dict, activations:list):

    caches = []
    A = X
    L = len(params) // 2

    for l in range(1,L):

        A_prev = A
        A, cache = activate_layer(A_prev, params['W'+str(l)], params['b'+str(l)], activations[l])
        caches.append(cache)


    Y_hat, cache = activate_layer(A, params['W'+str(L)], params['b'+str(L)], activations[l])
    caches.append(cache)

    return Y_hat, caches










