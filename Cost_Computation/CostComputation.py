import numpy as np
from cost_funcs import *

def compute_cost(Y_hat, Y, cost_metric):

    
    m = Y.shape[1]
    
    if cost_metric.lower() == "logloss":

        cost = log_loss(m, Y_hat, Y)
    
    cost = np.squeeze(cost)   

    return cost