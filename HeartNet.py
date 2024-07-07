# Dependencies
import numpy as np
import matplotlib.pyplot as plt


def leaky_relu(z, alpha=0.01):
    """Applies the leaky ReLU activation function."""
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    """Computes the derivative of the leaky ReLU activation function."""
    return np.where(z > 0, 1, alpha)


def initialize_network(neuron_1=None, neuron_2=None, neuron_3=None, random=np.random):
    """Network initialization. Resets the network with new random weights and biases."""
    global weight_1, weight_2, weight_3, bias_1, bias_2, bias_3
    weight_1 = random.randn(neuron_1, 2) * np.sqrt(2 / 2)
    weight_2 = random.randn(neuron_2, neuron_1) * np.sqrt(2 / neuron_1)
    weight_3 = random.randn(2, neuron_2) * np.sqrt(2 / neuron_2)
    bias_1 = random.randn(neuron_1, 1) * np.sqrt(2 / 2)
    bias_2 = random.randn(neuron_2, 1) * np.sqrt(2 / neuron_1)
    bias_3 = random.randn(2, 1) * np.sqrt(2 / neuron_2)


def forward_propagation(activation_0):
    """Performs forward propagation through the network."""
    global weight_1, weight_2, weight_3, bias_1, bias_2, bias_3
    z1 = weight_1 @ activation_0 + bias_1
    activation_1 = leaky_relu(z1)
    z2 = weight_2 @ activation_1 + bias_2
    activation_2 = leaky_relu(z2)
    z3 = weight_3 @ activation_2 + bias_3
    activation_3 = leaky_relu(z3)
    return activation_0, z1, activation_1, z2, activation_2, z3, activation_3


def loss(x, y):
    """Calculates the loss using mean squared error."""
    return np.linalg.norm(forward_propagation(x)[-1] - y) ** 2 / x.size


def Jacobian_Weight_3(x, y):
    """Calculates the Jacobian for the third layer weights.
       
       1. Collect all activations and weighted sums at each layer of the network.
       2. Variable J will store parts of the result as the network trains, updating it 
          in each line.
       3. dC/da3 is calculated.
       4. Result is multiplied and calculated by the derivative of sigmoid, evaluated at z3.
       5. Dot product is taken (along the axis that holds the training examples) with the 
          final partial derivative, i.e. dz3/dW3 = a2, and divide by the number of training 
          examples, for the average over all training examples.
       6. Return the result of the function.
    """
    activation_0, z1, activation_1, z2, activation_2, z3, activation_3 = forward_propagation(x)
    jacobian = 2 * (activation_3 - y)
    jacobian = jacobian * leaky_relu_derivative(z3)
    jacobian = jacobian @ activation_2.T / x.size
    return jacobian


def Jacobian_bias_3(x, y):
    """
    Calculate the Jacobian for the third layer biases.

    This function computes the gradient of the loss function with respect to 
    the biases in the third layer of the network. The process involves:

    1. Performing forward propagation to obtain activations and pre-activations 
       (z-values) at each layer.
    2. Calculating the gradient of the loss with respect to the activations 
       at the output layer.
    3. Propagating the gradient backward through the third layer using the chain 
       rule, incorporating the derivative of the leaky ReLU activation function.
    4. Summing the gradients over all training examples to get the final Jacobian.
    """
    activation_0, z1, activation_1, z2, activation_2, z3, activation_3 = forward_propagation(x)
    jacobian = 2 * (activation_3 - y)
    jacobian = jacobian * leaky_relu_derivative(z3)
    jacobian = np.sum(jacobian, axis=1, keepdims=True) / x.size
    return jacobian


def Jacobian_Weight_2(x, y):
    """
    Calculates the Jacobian for the second layer weights.

    Computes the gradient of the loss function wrt weights in the second layer 
    of the network. Steps:

    1. Forward propagation to obtain activations and pre-activations (z-values) 
       at each layer.
    2. Calculate gradient of the loss wrt the activations at the output layer.
    3. Propagate gradient backward through the third layer to the second 
       layer using the chain rule, incorporate the derivative of leaky 
       ReLU.
    4. Compute the final Jacobian by multiplying the gradient with the 
       activations from the first layer and averaging over the number of training examples.
    """
    activation_0, z1, activation_1, z2, activation_2, z3, activation_3 = forward_propagation(x)
    jacobian = 2 * (activation_3 - y)
    jacobian = jacobian * leaky_relu_derivative(z3)
    jacobian = (jacobian.T @ weight_3).T
    jacobian = jacobian * leaky_relu_derivative(z2)
    jacobian = jacobian @ activation_1.T / x.size
    return jacobian


def Jacobian_bias_2(x, y):
    """Calculate Jacobian for the second layer biases."""
    activation_0, z1, activation_1, z2, activation_2, z3, activation_3 = forward_propagation(x)
    jacobian = 2 * (activation_3 - y)
    jacobian = jacobian * leaky_relu_derivative(z3)
    jacobian = (jacobian.T @ weight_3).T
    jacobian = jacobian * leaky_relu_derivative(z2)
    jacobian = np.sum(jacobian, axis=1, keepdims=True) / x.size
    return jacobian    


def Jacobian_Weight_1(x, y):
    """Calculates the Jacobian for the first layer weights."""
    activation_0, z1, activation_1, z2, activation_2, z3, activation_3 = forward_propagation(x)
    jacobian = 2 * (activation_3 - y)
    jacobian = jacobian * leaky_relu_derivative(z3)
    jacobian = (jacobian.T @ weight_3).T
    jacobian = jacobian * leaky_relu_derivative(z2)
    jacobian = (jacobian.T @ weight_2).T
    jacobian = jacobian * leaky_relu_derivative(z1)
    jacobian = jacobian @ activation_0.T / x.size
    return jacobian


def Jacobian_bias_1(x, y):
    """Calculate Jacobian for the first layer biases."""
    activation_0, z1, activation_1, z2, activation_2, z3, activation_3 = forward_propagation(x)
    jacobian = 2 * (activation_3 - y)
    jacobian = jacobian * leaky_relu_derivative(z3)
    jacobian = (jacobian.T @ weight_3).T
    jacobian = jacobian * leaky_relu_derivative(z2)
    jacobian = (jacobian.T @ weight_2).T
    jacobian = jacobian * leaky_relu_derivative(z1)
    jacobian = np.sum(jacobian, axis=1, keepdims=True) / x.size
    return jacobian

def training_data():
  """Generate training data."""
  t = np.linspace(0, 2 * np.pi, 1000)
  x = (16 * np.sin(t)**3).reshape(1, -1)
  y = (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)).reshape(1, -1)
  data = np.vstack((x, y))
  return data, data  


def plot_training(x, y, epochs=None, learning_rate=None, noise=None):
    """Train network and plot results."""
    global weight_1, weight_2, weight_3, bias_1, bias_2, bias_3
    losses = []
    
    for epoch in range(epochs):
        # Gaussian noise to improve generalization capabilities
        x_noise = x + noise * np.random.randn(*x.shape)
        y_noise = y + noise * np.random.randn(*y.shape)

        # Compute Jacobians with Gaussian noise
        J_w3 = Jacobian_Weight_3(x_noise, y_noise) # Layer 3
        J_b3 = Jacobian_bias_3(x_noise, y_noise)
        J_w2 = Jacobian_Weight_2(x_noise, y_noise) # Layer 2
        J_b2 = Jacobian_bias_2(x_noise, y_noise)
        J_w1 = Jacobian_Weight_1(x_noise, y_noise) # Layer 1
        J_b1 = Jacobian_bias_1(x_noise, y_noise)
        
        # Update weights and biases using computed Jacobians (gradient descent, chain rule, backpropagation)
        weight_3 -= learning_rate * J_w3
        bias_3 -= learning_rate * J_b3
        weight_2 -= learning_rate * J_w2
        bias_2 -= learning_rate * J_b2
        weight_1 -= learning_rate * J_w1
        bias_1 -= learning_rate * J_b1
        
        # Calculate and record current loss
        current_loss = loss(x, y)
        losses.append(current_loss)
        
        # Plot training progress every 1000 epochs
        if epoch % 1000 == 0:
            plt.clf()
            plt.scatter(x[0, :], x[1, :], color='red', marker='.', label='Training Data')
            prediction = forward_propagation(x)[-1]
            plt.scatter(prediction[0, :], prediction[1, :], color='blue', marker='.', label='Prediction')
            plt.legend()
            plt.title(f'Epoch: {epoch}, Loss: {current_loss:.4f}')
            plt.pause(0.1)

    # Plot training loss over epochs
    plt.figure()
    plt.plot(losses)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


# Fit the training data and parameters
neuron_1, neuron_2, neuron_3 = 40, 40, 40
initialize_network(neuron_1, neuron_2, neuron_3)
x, y = training_data()

# Plots the neural network fitting to the outline of a heart
plot_training(x, y, epochs=170000, learning_rate=1e-4, noise=0.5) # Tune epochs, learning rate of Jacobians, Gaussian noise 
