from .layers import DenseLayer
import pandas as pd
import numpy as np

class MultiLayerPerceptron:
    def __init__(self):
        self._network = None

    def predict(self) -> float:
        return None

    def createNetwork(self, layers: list[DenseLayer], input_shape: int) -> list[DenseLayer]:
        network = layers
        for i, layer in enumerate(network):
            if i == 0:
                # Initialize the first layer with the input_shape and set weights to (31, 24)
                layer.initialize(input_shape)
                print(f"layer({i}): {layer.weights.shape}")  # Weights should be (31, 24)
            else:
                # Initialize subsequent layers with the number of neurons from the previous layer
                layer.initialize(network[i-1]._num_of_neuron)
                print(f"layer({i}): {layer.weights.shape}")
        return network

    
    def fit(
            self,
            network: list[DenseLayer],
            data_train: pd,
            data_valid: pd,
            loss: str,
            learning_rate,
            batch_size,
            epochs
        ) -> None:

        assert len(network) > 3

        output = None
        for layer in network:
            if output is not None:
                layer.forward(output)
            else:
                output = layer.forward(data_train)


        # for epoch in range(epochs):
        #     pass