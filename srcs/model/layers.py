import numpy as np
import pandas as pd

def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

ACTIVATIONS = {
    "sigmoid": lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500))),
    "softmax": softmax,
    "relu": lambda z: np.maximum(0, z),
    "leaky_relu": lambda z, alpha=0.01: np.where(z > 0, z, alpha * z),
    "selu": lambda z, alpha=1.67326, scale=1.0507: scale * np.where(z > 0, z, alpha * (np.exp(z) - 1))
}


def heUniform(input_dim, output_dim):
    limit = np.sqrt(6 / input_dim)
    return np.random.uniform(-limit, limit, (input_dim, output_dim))

def xavierUniform(input_dim, output_dim):
    limit = np.sqrt(6 / (input_dim + output_dim))
    return np.random.uniform(-limit, limit, (input_dim, output_dim))

def lecunNormal(input_dim, output_dim):
    stddev = np.sqrt(1 / input_dim)
    return np.random.normal(0, stddev, (input_dim, output_dim))

INITIALIZERS = {
    "heUniform": heUniform,
    "xavierUniform": xavierUniform,
    "lecunNormal": lecunNormal, 
    "random": lambda z,x : np.random.randn(z, x) * 0.01
}


class DenseLayer:
    def __init__(self, num_of_neuron, activation: str, weights_initializer: str = None):
        self._num_of_neuron = num_of_neuron
        self._activation = activation
        self._weights_initializer = weights_initializer
        self.weights = None
        self.biases = None

        self.in_data = None

        self.d_weights = None
        self.d_biases = None


    def initialize(self, input_dim):
        if self._weights_initializer not in INITIALIZERS:
            self._weights_initializer = "random"
        self.weights = INITIALIZERS[self._weights_initializer](input_dim, self._num_of_neuron)
        self.biases = np.zeros((1, self._num_of_neuron))
    

    def forward(self, input_data):
        # Linear Function
        self.in_data = input_data
        z = np.dot(input_data, self.weights) + self.biases
        if self._activation in ACTIVATIONS:
            return ACTIVATIONS[self._activation](z)
        else:
            raise ValueError(f"Unsupported activation function: {self._activation}")


    def backward(self, delta, learning_rate):
        self.d_weights = np.dot(self.in_data.T, delta)
        self.d_biases = np.sum(delta, axis=0, keepdims=True)

        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases

        return np.dot(delta, self.weights.T)