import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Input
x = np.array([0.1, 0.3])

# Target
y = 0.2

# Input to output weights
weights = np.array([-0.8, 0.5])

# The learning rate, eta in teh weight step equation
learnrate = 0.5

# The neural network output (y-hat)

nn_output = sigmoid(np.dot(x, weights))

# Output error
error = y - nn_output

# Error term (lowercase delta)
error_term = error * sigmoid_prime(np.dot(x, weights))

# Gradient descent step

del_w = learnrate * error_term * x

