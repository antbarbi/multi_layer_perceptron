from .layers import DenseLayer
import pandas as pd
import numpy as np

class MultiLayerPerceptron:
    def __init__(self):
        self._network = None

    def predict(self) -> float:
        return None

    def createNetwork(self, layers: list[DenseLayer]) -> list[DenseLayer]:
        network = layers
        for i, layer in enumerate(network):
            if i == 0:
                layer.initialize(layer._num_of_neuron)
                print(f"layer({i}):", layer._num_of_neuron)
            else:
                layer.initialize(network[i-1]._num_of_neuron)
                print(f"layer({i}):", network[i-1]._num_of_neuron)
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
            if output:
                layer.forward(output)
            else:
                layer.forward(data_train)


        # for epoch in range(epochs):
        #     pass