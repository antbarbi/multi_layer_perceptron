from .layers import DenseLayer
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from IPython.display import clear_output
from dataclasses import dataclass, field


def binaryCrossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    # Compute binary cross-entropy loss
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    predictions = (y_pred > 0.5).astype(int)
    return np.mean(predictions == y_true)


L = {
    "binaryCrossentropy": binaryCrossentropy,
}

@dataclass
class InteractivePlot:
    filename: str = field(default_factory=str)
    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    train_accuracies: list = field(default_factory=list)
    val_accuracies: list = field(default_factory=list)


class EarlyStoping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss, network):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = [layer.weights.copy() for layer in network]
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping after {self.counter} epochs with patience {self.patience}")
                return True
        return False


early_stopping = EarlyStoping(patience=30, min_delta=0)


class MultiLayerPerceptron:
    def __init__(self):
        self._network = []
        self.int_plot = InteractivePlot()
        self.loss = L

        self.multi_plots: list[InteractivePlot] = []

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
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_valid: np.ndarray,
            y_valid: np.ndarray,
            loss_func: str,
            learning_rate: float,
            batch_size: int,
            epochs: int
        ) -> None:

        assert len(network) > 2 # At least 2 hidden layers

        fig_params = self.setup_figure()

        for epoch in range(epochs):
            forward_output: pd.DataFrame = X_train
            for layer in network:
                forward_output = layer.forward(forward_output)

            assert loss_func in L
            
            # Calculate Loss with Loss Function
            loss = L[loss_func](y_train, forward_output)
            self.int_plot.train_losses.append(loss)

            # Calculate training accuracy
            train_acc = accuracy(y_train, forward_output)
            self.int_plot.train_accuracies.append(train_acc)

            delta = forward_output - y_train

            for layer in reversed(network):
                delta = layer.backward(delta, learning_rate)

            val_output = X_valid
            for layer in network:
                val_output = layer.forward(val_output)
            
            # Calculate validation loss
            val_loss = L[loss_func](y_valid, val_output)
            self.int_plot.val_losses.append(val_loss)

            # Calculate validation accuracy
            val_acc = accuracy(y_valid, val_output)
            self.int_plot.val_accuracies.append(val_acc)
            
            print(
                f"Epoch {epoch + 1}/{epochs:<3} | Loss: {loss:<10.4f} | Val-Loss: {val_loss:<10.4f} | "
                f"Train-Acc: {train_acc:<10.4f} | Val-Acc: {val_acc:<10.4f}"
                )

            self.update_figure(*fig_params)

            # Early Stopping
            if early_stopping and early_stopping(val_loss, network):
                for i, layer in enumerate(network):
                    layer.weights = early_stopping.best_weights[i]
                break

        self._network = network
        plt.ioff()
        plt.show()


    def setup_figure(self):
        plt.ion()
        fix, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Subplot for loss
        line1, = ax.plot(self.int_plot.train_losses, label="Training Loss")
        line2, = ax.plot(self.int_plot.val_losses, label="Validation Loss")
        ax.legend()
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Loss by epochs")

        # Subplot for accuracy
        line3, = ax2.plot(self.int_plot.train_accuracies, label="Training Accuracy")
        line4, = ax2.plot(self.int_plot.val_accuracies, label="Validation Accuracy")
        ax2.legend()
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy by epochs")
        
        return ax, ax2, line1, line2, line3, line4


    def update_figure(self, ax, ax2, line1, line2, line3, line4):
        # Update interactive plot for loss
        line1.set_ydata(self.int_plot.train_losses)
        line1.set_xdata(range(len(self.int_plot.train_losses)))
        line2.set_ydata(self.int_plot.val_losses)
        line2.set_xdata(range(len(self.int_plot.val_losses)))
        ax.relim()
        ax.autoscale_view()
        
        # Update interactive plot for accuracy
        line3.set_ydata(self.int_plot.train_accuracies)
        line3.set_xdata(range(len(self.int_plot.train_accuracies)))
        line4.set_ydata(self.int_plot.val_accuracies)
        line4.set_xdata(range(len(self.int_plot.val_accuracies)))
        ax2.relim()
        ax2.autoscale_view()
        clear_output(wait=True)
        plt.draw()
        plt.pause(0.01)


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
        with open("model.json", "w") as file:
            json.dump(model, file)


    def save_metrics(self, file_name: str = "metrics.json"):
        metrics = {
            "train_losses": self.int_plot.train_losses,
            "val_losses": self.int_plot.val_losses,
            "train_accuracies": self.int_plot.train_accuracies,
            "val_accuracies": self.int_plot.val_accuracies
        }
        with open(file_name, "w") as file:
            json.dump(metrics, file, indent=4)
        print(f"Training metrics saved to {file_name}")


    def load_metrics(self, *file_names: tuple[str]):
        
        for i, file_name in enumerate(file_names):
            with open(file_name, "r") as file:
                metrics = json.load(file)
            self.multi_plots.append(InteractivePlot())
            self.multi_plots[i].filename = file_name
            self.multi_plots[i].train_losses = metrics.get("train_losses", [])
            self.multi_plots[i].val_losses = metrics.get("val_losses", [])
            self.multi_plots[i].val_accuracies = metrics.get("train_accuracies", [])
            self.multi_plots[i].val_accuracies = metrics.get("val_accuracies", [])

            print(f"Training metrics loaded from {file_name}")


    def plot_metrics(self):
        plt.figure(figsize=(12, 5))
        rows = len(self.multi_plots)

        for i, plot in enumerate(self.multi_plots):
            loss_subplot = 2 * i + 1      # Odd indices for Loss
            acc_subplot = 2 * i + 2 
            plt.subplot(rows, 2, loss_subplot)
            plt.plot(plot.train_losses, label="Training Loss")
            plt.plot(plot.val_losses, label="Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title(f"Loss by Epochs - {plot.filename}")
            plt.legend()
            plt.grid(True)

            # Plot Accuracy
            plt.subplot(rows, 2, acc_subplot)
            plt.plot(plot.train_accuracies, label="Training Accuracy")
            plt.plot(plot.val_accuracies, label="Validation Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title(f"Accuracy by Epochs - {plot.filename}")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()


    def load_model(self, file_name: str):
        with open(file_name, "r") as file:  # Open file in text mode
            model = json.load(file)

        for layer in model["layers"]:
            init_layer = DenseLayer(layer["units"], layer["activation"])
            init_layer.weights = np.array(layer["weights"])
            init_layer.biases = np.array(layer["biases"])
            self._network.append(init_layer)

