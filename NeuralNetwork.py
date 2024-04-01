import numpy as np

from Backward_Propagation.BackwardPropagation import backward_propagation
from Backward_Propagation.UpdateParams import update_parameters
from Forward_Propagation.ForwardPropagation import forward_propagation
from Forward_Propagation.InitializeParams import initialize_params
from Cost_Computation.CostComputation import compute_cost

class NeuralNetwork:

    def __init__(self, layer_dims: list, activation: list, metric:str, cost_metric: str = "logloss") -> None:
        
        np.random.seed(1)
        self.costs = [] 

        self.layer_dims = layer_dims

        self.params = initialize_params(layer_dims)

        self.activation = activation

        self.cost_metric = cost_metric


        # --------- Change this when adding compile func -------

        self.metric = metric

        # --------- ------------------------------------ -------

    def train(self, X:np.ndarray, Y:np.ndarray, epoch: int, alpha: float, verbose: bool=False):

        if self.layer_dims[0] != X.shape[1]:

            raise ValueError(f"Dims of input_layer != X.shape[1] || {self.layer_dims[0]} != {X.shape[1]}")
        
        for i in range(0, epoch):

        
            Y_Hat, caches = forward_propagation(X.T, self.params, self.activation)
            
            cost = compute_cost(Y_Hat, Y.reshape(-1, 1).T, self.cost_metric)
            
            grads = backward_propagation(Y_Hat, Y, caches, self.activation, self.metric)

    
            self.params = update_parameters(self.params, grads, alpha)
            
            if verbose and i % 10 == 0 or i == epoch - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
                
            self.costs.append(cost)

    def predict(self):
        pass