import numpy as np

from .LayerComputation import compute_layer
from .activation_funcs import *


def activate_layer(prev_A, W, b, activation):

    Z, linear_cache = compute_layer(prev_A, W,  b)

    if activation.lower() == "relu":
        
        A = relu(Z)

    elif activation.lower() == "sigmoid":
        
        A = sigmoid(Z)

    elif activation.lower() == "elu":
        
        A = elu(Z)

    elif activation.lower() == "tanh":
        
        A = tanh(Z)

    elif activation.lower() == "linear":

        A = linear(Z)

    elif activation.lower() == "swish":

        A = swish(Z)

    elif activation.lower() == "leakyrelu":

        A = leakyrelu(Z)

    else:

        raise ValueError("Unsupported Activation Function")

    activation_cache = [A, Z]

    cache = [linear_cache, activation_cache]

    return A, cache

