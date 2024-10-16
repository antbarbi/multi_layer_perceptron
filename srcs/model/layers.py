import numpy as np

def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

ACTIVATIONS = {
    "sigmoid": lambda z: 1 / (1 + np.exp(-z)),
    "softmax": lambda z: softmax(z),
}

class DenseLayer:
    def __init__(self, num_of_neuron, activation: str, weights_initializer: str= None):
        self._num_of_neuron = num_of_neuron
        self._activation = activation
        self._weights_initializer = weights_initializer
        self.weights = None
        self.biases = None


    def initialize(self, input_dim):
        if self._weights_initializer == 'heUniform':
            limit = np.sqrt(6 / input_dim)
            self.weights = np.random.uniform(-limit, limit, (input_dim, self._num_of_neuron))
        else:
            self.weights = np.random.randn(input_dim, self._num_of_neuron) * 0.01
        self.biases = np.zeros((1, self._num_of_neuron))

        # print(self.weights.shape)


    # def sigmoid(self, z):
    #     return 1 / (1 + np.exp(-z))
    

    # def softmax(self, z):
    #     exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    #     return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    

    def forward(self, input_data):
        z = np.dot(input_data, self.weights) + self.biases
        if self._activation in ACTIVATIONS:
            return ACTIVATIONS[self._activation](z)
        else:
            raise ValueError(f"Unsupported activation function: {self._activation}")
