import numpy as np
import copy



def update_parameters(params, grads, learning_rate):
    
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2 
    for l in range(L):
        
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]


    return parameters