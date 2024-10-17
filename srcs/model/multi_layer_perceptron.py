from .layers import DenseLayer
import pandas as pd
import numpy as np
import json


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
        self._network = []

    def createNetwork(self, layers: list[DenseLayer], input_shape: int) -> list[DenseLayer]:
        network = layers
        for i, layer in enumerate(network):
            if i == 0:
                layer.initialize(input_shape)
            else:
                layer.initialize(network[i-1]._num_of_neuron)
        return network


    def predict(self, input_data: np.ndarray) -> np.ndarray:
        if self._network is None:
            raise ValueError("The network has not been initialized. Please train the model first.")
        
        output = input_data
        for layer in self._network:
            output = layer.forward(output)
        
        return output

    
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

        self._network = network

    def save_model(self):

        model = {}
        model["layers"] = [
            {
                "units": net._num_of_neuron,
                "activation": net._activation,
                "weights": net.weights.tolist(),
                "biases": net.biases.tolist()
            } for net in self._network
        ]
        with open("model.json", "w") as file:  # Open file in text mode
            json.dump(model, file)

    def load_model(self, file_name: str):
        with open(file_name, "r") as file:  # Open file in text mode
            model = json.load(file)

        for layer in model["layers"]:
            init_layer = DenseLayer(layer["units"], layer["activation"])
            init_layer.weights = np.array(layer["weights"])
            init_layer.biases = np.array(layer["biases"])
            self._network.append(init_layer)

        

