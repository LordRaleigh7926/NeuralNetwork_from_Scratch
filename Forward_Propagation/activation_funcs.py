import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    y = np.maximum(0,x)
    
    assert(y.shape == x.shape)
    
    return y


def sigmoid(x: np.ndarray) -> np.ndarray:
    y = 1/(1+(np.exp(-x)))
    return y


def tanh(x: np.ndarray)-> np.ndarray:
    y = ((np.exp(x))-(np.exp(x)))/((np.exp(x))+(np.exp(-x)))
    return y


def elu(x: np.ndarray, alpha: float = 0.1)-> np.ndarray:

    if x>0:
        return x
    else:
        y = alpha*((np.exp(x))-1)
        return y


def swish(x: np.ndarray)-> np.ndarray:
    y = x*sigmoid(x)
    return y


def leakyrelu(x: np.ndarray)-> np.ndarray:
    y = np.maximum(0.1*x,x)
    return y 


def linear(x: np.ndarray)-> np.ndarray:
    return x
