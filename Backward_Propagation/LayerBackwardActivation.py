from .LayerBackwardComputation import computed_layers_backwards
from .derivatedActivation_funcs import *

def activated_layers_backwards(dA, cache, activation, lambd):

    linear_cache, activation_cache = cache

    Z = activation_cache[1]

    if activation == "relu":
        dZ = Differentiated_RELU(Z, dA)
        
    elif activation == "sigmoid":
        dZ = Differentiated_Sigmoid(Z, dA)

    else:

        raise ValueError("Unsupported activation function") 

    dA_prev, dW, db = computed_layers_backwards(dZ, linear_cache, lambd)
    
    return dA_prev, dW, db



