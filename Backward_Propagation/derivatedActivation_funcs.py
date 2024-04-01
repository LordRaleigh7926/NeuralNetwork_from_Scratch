import numpy as np 


def Differentiated_RELU(Z:np.ndarray, dA):

    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0

    return dZ

def Differentiated_Sigmoid(Z:np.ndarray, dA):

    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ



# IMPLEMENT MORE FUNCS

