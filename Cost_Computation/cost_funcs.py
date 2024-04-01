import numpy as np 

def log_loss(m, Y_hat, Y):
    
    cost = -(1/m)*np.sum(np.dot(Y, np.log(Y_hat).T) + np.dot((1-Y), np.log(1-Y_hat).T), axis=1, keepdims=True)


    return cost