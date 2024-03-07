import numpy as np 


def sigmoid(x: np.ndarray) -> np.ndarray:
    y = 1/(1+(np.exp(-x)))
    return y


def Differentiated_RELU(x:np.ndarray):

    return np.where(x >= 0, 1, 0)

def Differentiated_Sigmoid(x:np.ndarray):

    sigmoid_x = 1 / (1 + np.exp(-x))
    return sigmoid_x * (1 - sigmoid_x)



# IMPLEMENT MORE FUNCS

