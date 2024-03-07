from LayerBackwardComputation import computed_layers_backwards
from derivatedActivation_funcs import *

def activated_layers_backwards(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = Differentiated_RELU(activation_cache)
        
    elif activation == "sigmoid":
        dZ = Differentiated_Sigmoid(activation_cache)

    dA_prev, dW, db = computed_layers_backwards(dZ, linear_cache)
    
    return dA_prev, dW, db



# EDITING NEED TO BE DONE FOR CACHE