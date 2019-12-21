
# Dependencies

import numpy as np
import matplotlib.pyplot as plt


# Activation function and its derivative

sigmoid = lambda z : 1 / (1 + np.exp(-z))
d_sigmoid = lambda z : np.cosh(z/2)**(-2) / 4


# This function initializes the network with it's structure and resets any training already done
# You can tune the reset_network function's parameters n1, n2 to vary the amount of neurons
# in the hidden layer

def reset_network (n1 = 6, n2 = 7, random=np.random) :
    global W1, W2, W3, b1, b2, b3
    W1 = random.randn(n1, 1) / 2
    W2 = random.randn(n2, n1) / 2
    W3 = random.randn(2, n2) / 2
    b1 = random.randn(n1, 1) / 2
    b2 = random.randn(n2, 1) / 2
    b3 = random.randn(2, 1) / 2


# This function feeds forward each activation to the next layer and returns all weighted sums and activations

def network_function(a0) :
    z1 = W1 @ a0 + b1
    a1 = sigmoid(z1)
    z2 = W2 @ a1 + b2
    a2 = sigmoid(z2)
    z3 = W3 @ a2 + b3
    a3 = sigmoid(z3)
    return a0, z1, a1, z2, a2, z3, a3


# This is the loss function of the neural network with respect to the training set
def loss(x, y) :
    return np.linalg.norm(network_function(x)[-1] - y)**2 / x.size


# Jacobian for the third layer weights

def J_W3 (x, y) :
    # Collects all the activations and weighted sums at each layer of the network
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    # The variable J will store parts of the result as the network trains, updating it in each line
    # First, dC/da3 is calculated, using the expressions above
    J = 2 * (a3 - y)
    # Second, the result is multiplied and calculated by the derivative of sigmoid, evaluated at z3
    J = J * d_sigmoid(z3)
    # Third, the dot product is taken (along the axis that holds the training examples) with the final partial derivative,
    # i.e. dz3/dW3 = a2
    # and divide by the number of training examples, for the average over all training examples
    J = J @ a2.T / x.size
    # Finally return the result of the function
    return J


# In this function, the Jacobian is implemented for the bias
# Only the last partial derivative is different
# The first two partial derivatives are the same as previously

def J_b3 (x, y) :
    # As last time, we'll first set up the activations
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    # Next, the first two partial derivatives of the Jacobian terms are calculated
    J = 2 * (a3 - y)
    J = J * d_sigmoid(z3)
    # For the final line, we don't need to multiply by dz3/db3, because that is multiplying by 1
    # But we still need to sum over all training examples 
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J


# Compare this function to J_W3 to see how it changes
def J_W2 (x, y) :
    #The first two lines are identical to in J_W3
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)    
    J = 2 * (a3 - y)
    # The next two lines implement da3/da2, first σ' and then W3.
    J = J * d_sigmoid(z3)
    J = (J.T @ W3).T
    # Then the final lines are the same as in J_W3 but with the layer number bumped down
    J = J * d_sigmoid(z2)
    J = J @ a1.T / x.size
    return J


def J_b2 (x, y) :
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigmoid(z3)
    J = (J.T @ W3).T
    J = J * d_sigmoid(z2)
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J


def J_W1 (x, y) :
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigmoid(z3)
    J = (J.T @ W3).T
    J = J * d_sigmoid(z2)
    J = (J.T @ W2).T
    J = J * d_sigmoid(z1)
    J = J @ a0.T / x.size
    return J


def J_b1 (x, y) :
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigmoid(z3)
    J = (J.T @ W3).T
    J = J * d_sigmoid(z2)
    J = (J.T @ W2).T
    J = J * d_sigmoid(z1)
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J


# Fit the training data

reset_network(n1, n2)
x, y = training_data()
reset_network()


# Plots the neural network fitting to the outline of a heart
# You can tune the number of epochs, the steps (aggression) of the Jacobian descent and how much noise to add

plot_training(x, y, epochs=50000, aggression=7, noise=1)    
