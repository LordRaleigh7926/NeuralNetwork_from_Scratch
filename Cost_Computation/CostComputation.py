import numpy as np
from .cost_funcs import *

def compute_cost(Y_hat, Y, cost_metric, params, lambd):

    
    m = Y.shape[1]
    

    if cost_metric.lower() == "logloss":

        cost = log_loss(m, Y_hat, Y)


    # Implementing the L2 regularization
    if lambd != 0:

        i = 1

        L2_regularization_cost = (1/m)*(lambd/2)
        rest_of_reg_cost = 0
        
        while True:

            try:
                rest_of_reg_cost = rest_of_reg_cost + np.sum(np.square(params["W"+str(i)]))
                i += 1

            except:
                L2_regularization_cost = L2_regularization_cost*rest_of_reg_cost
                cost = np.squeeze(cost+L2_regularization_cost)
                return cost


    cost = np.squeeze(cost)    

    return cost