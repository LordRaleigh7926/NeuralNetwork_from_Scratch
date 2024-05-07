import numpy as np 

def computed_layers_backwards(dZ, cache, lambd):


    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m)*(np.dot(dZ, A_prev.T)) + (lambd/m)*W
    db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    
    return dA_prev, dW, db

