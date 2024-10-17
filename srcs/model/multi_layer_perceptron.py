from .layers import DenseLayer
import pandas as pd
import numpy as np


def binaryCrossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    # Compute binary cross-entropy loss
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


L = {
    "binaryCrossentropy": binaryCrossentropy,
}


class MultiLayerPerceptron:
    def __init__(self):
        self._network = None

    def predict(self) -> float:
        return None

    def createNetwork(self, layers: list[DenseLayer], input_shape: int) -> list[DenseLayer]:
        network = layers
        for i, layer in enumerate(network):
            if i == 0:
                layer.initialize(input_shape)
            else:
                layer.initialize(network[i-1]._num_of_neuron)
        return network

    
    def fit(
            self,
            network: list[DenseLayer],
            data_train: pd.DataFrame,
            data_valid: pd.DataFrame,
            loss_func: str,
            learning_rate: float,
            batch_size: int,
            epochs: int
        ) -> None:

        assert len(network) > 3

        for epoch in range(epochs):
            forward_output: pd.DataFrame = data_train
            for layer in network:
                forward_output = layer.forward(forward_output)

            assert loss_func in L
            # Calculate Loss with Loss Function
            loss = L[loss_func](data_valid, forward_output)

            delta = forward_output - data_valid

            for layer in reversed(network):
                delta = layer.backward(delta, learning_rate)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

        
            

