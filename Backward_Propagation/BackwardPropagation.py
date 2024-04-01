import numpy as np
from .LayerBackwardActivation import activated_layers_backwards

def backward_propagation(Y_hat: np.ndarray, Y: np.ndarray, caches, activation: list, metric:str):

    grads = {}
    L = len(caches)
    m = Y_hat.shape[1]
    Y = Y.reshape(Y_hat.shape)

    if metric == "cross_entropy":
        dY_hat = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

    current_cache = caches
    dA_prev_temp, dW_temp, db_temp = activated_layers_backwards(dY_hat, current_cache[-1], activation[-1])
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L-1)):

        dA_prev_temp, dW_temp, db_temp = activated_layers_backwards(dA_prev_temp, current_cache[l], activation[l])
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp


    return grads
