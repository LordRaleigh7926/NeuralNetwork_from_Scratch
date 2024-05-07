import numpy as np

from Backward_Propagation.BackwardPropagation import backward_propagation
from Backward_Propagation.UpdateParams import update_parameters
from Forward_Propagation.ForwardPropagation import forward_propagation
from Forward_Propagation.InitializeParams import initialize_params
from Cost_Computation.CostComputation import compute_cost

class NeuralNetwork:

    """
    A class representing a simple neural network from scratch, designed for educational purposes.

    This neural network implementation provides a hands-on approach to understanding the inner workings of neural networks, including forward and backward propagation, parameter updates, and cost computation.

    Attributes:
        layer_dims (list): A list containing the dimensions of each layer in the network. The first element represents the number of input features, and the last element represents the number of output features.
        activation (list): A list of activation functions for each layer. The activation function is applied to the output of each layer, introducing non-linearity into the model.
        cost_metric (str): The metric used to compute the cost function. This determines how the model's predictions are compared to the true labels to calculate the loss.
        metric (str): The metric used for evaluating the model's performance. This is used to assess the model's accuracy, precision, recall, etc., depending on the problem at hand.
        params (dict): A dictionary containing the weights and biases for each layer. These parameters are learned during the training process.
        costs (list): A list of costs computed during training. This can be used to monitor the model's learning progress over epochs.
    """

    def __init__(self, layer_dims: list, activation: list, metric:str, cost_metric: str = "logloss", random_seed: int = 1) -> None:
        
        """
        Initializes the NeuralNetwork with the given parameters.

        Parameters:
            layer_dims (list): A list containing the dimensions of each layer in the network. The first element represents the number of input features, and the last element represents the number of output features.
            activation (list): A list of activation functions for each layer. The activation function is applied to the output of each layer, introducing non-linearity into the model.
            metric (str): The metric used for evaluating the model's performance. This is used to assess the model's accuracy, precision, recall, etc., depending on the problem at hand.
            cost_metric (str, optional): The metric used to compute the cost function. This determines how the model's predictions are compared to the true labels to calculate the loss. Defaults to "logloss".
            random_seed (int, optional): The seed for the random number generator. This ensures reproducibility of the model's training process. Defaults to 1.
        """

        # Check if layer_dims is a list
        if not isinstance(layer_dims, list):
            raise TypeError(f"layer_dims must be a list. Whereas here a {type(layer_dims)} is provided")
        
        # Check if activation is a list
        if not isinstance(activation, list):
            raise TypeError(f"activation must be a list. Whereas here a {type(activation)} is provided")
        
        # Check if metric is a string
        if not isinstance(metric, str):
            raise TypeError(f"metric must be a string. Whereas here a {type(metric)} is provided")
        
        # Check if cost_metric is a string
        if not isinstance(cost_metric, str):
            raise TypeError(f"cost_metric must be a string. Whereas here a {type(cost_metric)} is provided")
        
        # Check if random_seed is an integer
        if not isinstance(random_seed, int):
            raise TypeError(f"random_seed must be an integer. Whereas here a {type(random_seed)} is provided")

        np.random.seed(random_seed)

        self.costs = [] 

        self.layer_dims = layer_dims

        self.params = initialize_params(layer_dims)

        self.activation = activation

        self.cost_metric = cost_metric


        # --------- Change this when adding compile func -------

        self.metric = metric

        # ------------------------------------------------------

    def train(self, X:np.ndarray, Y:np.ndarray, epoch: int, alpha: float, verbose: bool=False, L2_Regularization_lambd:float = 0.0):

        """
        Trains the neural network using the given data for a specified number of epochs.

        This method implements the forward and backward propagation steps, cost computation, and parameter updates
        for each epoch. The training process aims to minimize the cost function, improving the model's predictions.

        Parameters:
            X (np.ndarray): The input data, represented as a 2D array where each row is an instance and each column is a feature.
            Y (np.ndarray): The true labels corresponding to the input data, represented as a 2D array where each row is an instance and each column is a label.
            epoch (int): The number of epochs to train the network. An epoch is a complete pass through the entire training dataset.
            alpha (float): The learning rate, which determines the step size at each iteration while updating the model's parameters.
            verbose (bool, optional): If True, prints the cost after each epoch. This can be useful for monitoring the training process. Defaults to False.
            L2_Regularization_lambd (float, optional): The L2 regularizer. Used for regularization. Default value is 0 that means no regularization.
        """

        if self.layer_dims[0] != X.shape[1]:

            raise ValueError(f"Dims of input_layer != X.shape[1] || {self.layer_dims[0]} != {X.shape[1]}")
        
        if not isinstance(L2_Regularization_lambd, float):

            raise TypeError(f"L2_Regularization_lambd must be a float. Whereas here a {type(L2_Regularization_lambd)} is provided")

        if not isinstance(epoch, int):

            raise TypeError(f"epoch must be an integer. Whereas here a {type(epoch)} is provided")

        if not isinstance(alpha, float):

            raise TypeError(f"alpha must be a float. Whereas here a {type(alpha)} is provided")

        
        for i in range(0, epoch):

        
            Y_Hat, caches = forward_propagation(X.T, self.params, self.activation)
            
            cost = compute_cost(Y_Hat, Y.reshape(-1, 1).T, self.cost_metric, self.params, L2_Regularization_lambd)
            
            grads = backward_propagation(Y_Hat, Y, caches, self.activation, self.metric, lambd=L2_Regularization_lambd)

    
            self.params = update_parameters(self.params, grads, alpha)
            
            if verbose and i % 10 == 0 or i == epoch - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
                
            self.costs.append(cost)

    def predict(self, X):
            """
            This function is used to predict the results of a  L-layer neural network.
            
            Arguments:
            X -- data set of examples you would like to label
            
            Returns:
            pred -- predictions for the given dataset X
            """
            
            m = X.shape[1]
            n = len(self.params) // 2 # number of layers in the neural network
            p = np.zeros((1,m))
            
            # Forward propagation
            probas, caches = forward_propagation(X.T, self.params, self.activation)

            pred = probas.squeeze()
            
                
            return pred
