# Dependencies

import numpy as np
import matplotlib.pyplot as plt


# Activation function and its derivative

sigmoid = lambda z : 1 / (1 + np.exp(-z))
sigmoid_derivative = lambda z : np.cosh(z/2)**(-2) / 4


# This function initializes the network with it's structure and resets any training already done
# You can tune the reset_network function's parameters neuron_1, neuron_2 to vary the amount of neurons
# in each hidden layer

def reset_network (neuron_1 = 6, neuron_2 = 7, random=np.random) :
    global Weight_1, weight_2, weight_3, bias_1, bias_2, bias_3
    weight_1 = random.randn(neuron_1, 1) / 2
    weight_2 = random.randn(neuron_2, neuron_1) / 2
    weight_3 = random.randn(2, neuron_2) / 2
    bias_1 = random.randn(neuron_1, 1) / 2
    bias_2 = random.randn(neuron_2, 1) / 2
    bias_3 = random.randn(2, 1) / 2


# This function feeds forward each activation to the next layer and returns all weighted sums and activations

def network_function(activation_0) :
    z1 = weight_1 @ activation_0 + bias_1
    activation_1 = sigmoid(z1)
    z2 = weight_2 @ activation_1 + bias_2
    activation_2 = sigmoid(z2)
    z3 = weight_3 @ activation_2 + bias_3
    activation_3 = sigmoid(z3)
    return activation_0, z1, activation_1, z2, activation_2, z3, activation_3


# This is the loss function of the neural network with respect to the training set
def loss(x, y) :
    return np.linalg.norm(network_function(x)[-1] - y)**2 / x.size


# Jacobian for the third layer weights

def Jacobian_Weight_3 (x, y) :
    # Collects all the activations and weighted sums at each layer of the network
    activation_0, z1, activation_1, z2, activation_2, z3, activation_3 = network_function(x)
    
    # The variable J will store parts of the result as the network trains, updating it in each line
    # First, dC/da3 is calculated, using the expressions above
    
    jacobian = 2 * (activation_3 - y)
    
    # Second, the result is multiplied and calculated by the derivative of sigmoid, evaluated at z3
    jacobian = jacobian * sigmoid_derivative(z3)
    
    # Third, the dot product is taken (along the axis that holds the training examples) with the final partial derivative,
    # i.e. dz3/dW3 = a2
    # and divide by the number of training examples, for the average over all training examples
    jacobian = jacobian @ activation_2.T / x.size
    
    # Finally return the result of the function
    return jacobian


# In this function, the Jacobian is implemented for the bias
# Only the last partial derivative is different
# The first two partial derivatives are the same as previously

def Jacobian_bias_3 (x, y) :
    # As last time, we'll first set up the activations
    activation_0, z1, activation_1, z2, activation_2, z3, activation_3 = network_function(x)
    
    # Next, the first two partial derivatives of the Jacobian terms are calculated
    jacobian = 2 * (activation_3 - y)
    jacobian = jacobian * sigmoid_derivative(z3)
    
    # For the final line, we don't need to multiply by dz3/db3, because that is multiplying by 1
    # But we still need to sum over all training examples 
    jacobian = np.sum(jacobian, axis=1, keepdims=True) / x.size
    return jacobian


# Compare this function to J_W3 to see how it changes
# Weight
def Jacobian_Weight_2 (x, y) :
    #The first two lines are identical to in J_W3
    activation_0, z1, activation_1, z2, activation_2, z3, activation_3 = network_function(x)    
    jacobian = 2 * (activation_3 - y)
    
    # The next two lines implement da3/da2, first σ' and then W3.
    jacobian = jacobian * sigmoid_derivative(z3)
    jacobian = (jacobian.T @ weight_3).T
    
    # Then the final lines are the same as in J_W3 but with the layer number bumped down
    jacobian = jacobian * sigmoid_derivative(z2)
    jacobian = jacobian @ activation_1.T / x.size
    return jacobian

# Bias
def Jacobian_bias_2 (x, y) :
    activation_0, z1, activation_1, z2, activation_2, z3, activation_3 = network_function(x)
    jacobian = 2 * (activation_3 - y)
    jacobian = jacobian * sigmoid_derivative(z3)
    jacboian = (jacobian.T @ weight_3).T
    jacobian = jacobian * sigmoid_derivative(z2)
    jacobian = np.sum(jacobian, axis=1, keepdims=True) / x.size
    return jacobian

# Weight
def Jacobian_Weight_1 (x, y) :
    activation_0, z1, activation_1, z2, activation_2, z3, activation_3 = network_function(x)
    jacobian = 2 * (activation_3 - y)
    jacobian = jacobian * sigmoid_derivative(z3)
    jacobian = (jacobian.T @ weight_3).T
    jacobian = jacobian * sigmoid_derivative(z2)
    jacobian = (jacobian.T @ weight_2).T
    jacobian = jacobian * sigmoid_derivative(z1)
    jacobian = jacobian @ activation_0.T / x.size
    return jacobian

# Bias
def Jacobian_bias_1 (x, y) :
    activation_0, z1, activation_1, z2, activation_2, z3, activation_3 = network_function(x)
    jacobian = 2 * (activation_3 - y)
    jacobian = jacobian * sigmoid_derivative(z3)
    jacobian = (jacobian.T @ weight_3).T
    jacobian = jacobian * sigmoid_derivative(z2)
    jacobian = (jacobian.T @ weight_2).T
    jacobian = jacobian * sigmoid_derivative(z1)
    jacobian = np.sum(J, axis=1, keepdims=True) / x.size
    return jacobian


# Fit the training data

reset_network(neuron_1, neuron_2)
x, y = training_data()
reset_network()


# Plots the neural network fitting to the outline of a heart
# You can tune the number of epochs, the steps (aggression) of the Jacobian descent and how much noise to add

plot_training(x, y, epochs=50000, aggression=7, noise=1)    
