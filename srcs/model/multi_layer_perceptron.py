from .layers import DenseLayer
import pandas as pd
import numpy as np

class MultiLayerPerceptron:
    def __init__(self):
        pass
        # self.layers = []

    def predict(self) -> float:
        return None

    def createNetwork(self, layers: list[DenseLayer]) -> list[DenseLayer]:
        network = layers
        input_dim = None
        for layer in network:
            if input_dim:
                layer.initialize(input_dim)
            input_dim = layer._num_of_neuron
        return network

    
    def fit(
            self,
            network,
            data_train: pd,
            data_valid: pd,
            loss: str,
            learning_rate,
            batch_size,
            epochs
        ) -> None:

        assert len(network) > 3

        for epoch in range(epochs):
            pass