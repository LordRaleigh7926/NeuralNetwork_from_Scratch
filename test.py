import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from NeuralNetwork import NeuralNetwork


# Generate a binary classification dataset
X, Y = make_classification(n_samples=1000, n_features=20, n_informative=8, n_redundant=12, random_state=42)

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the architecture of your neural network
layer_dims = [X_train.shape[1], 100, 100, 1] # Input layer, 2 hidden layer, and output layer
activation = ['relu', 'relu', 'sigmoid'] # Activation functions for each layer
metric = 'cross_entropy' # Loss function for binary classification

# Initialize and train the neural network
nn = NeuralNetwork(layer_dims, activation, metric)
nn.train(X_train, Y_train.T, epoch=3000, alpha=0.01, verbose=True, L2_Regularization_lambd=0.001)


# Make predictions on the test set
Y_pred_proba = nn.predict(X_test)

# Convert probabilities to class labels
Y_pred = np.where(Y_pred_proba >= 0.5, 1, 0)

# Calculate the accuracy of the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Test Accuracy: {accuracy}")
