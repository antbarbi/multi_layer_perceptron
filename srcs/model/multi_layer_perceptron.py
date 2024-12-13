from .layers import DenseLayer
import pandas as pd
import numpy as np
import json
import sys
import matplotlib.pyplot as plt
from IPython.display import clear_output
from dataclasses import dataclass, field
from sklearn.metrics import precision_score, recall_score, f1_score


def binaryCrossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    # Compute binary cross-entropy loss
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


L = {
    "binaryCrossentropy": binaryCrossentropy,
}


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    predictions = (y_pred > 0.5).astype(int)
    return np.mean(predictions == y_true)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred_binary = (y_pred > 0.5).astype(int)
    true_positives = np.sum((y_pred_binary == 1) & (y_true == 1))
    predicted_positives = np.sum(y_pred_binary == 1)
    if predicted_positives == 0:
        return 0.0
    return true_positives / predicted_positives


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred_binary = (y_pred > 0.5).astype(int)
    true_positives = np.sum((y_pred_binary == 1) & (y_true == 1))
    actual_positives = np.sum(y_true == 1)
    if actual_positives == 0:
        return 0.0
    return true_positives / actual_positives


# def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     prec = precision(y_true, y_pred)
#     rec = recall(y_true, y_pred)
#     if (prec + rec) == 0:
#         return 0.0
#     return 2 * (prec * rec) / (prec + rec)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    # acc = accuracy(y_true, y_pred)
    predictions = (y_pred > 0.5).astype(int)
    prec = precision_score(y_true, predictions, average="macro", zero_division=0)
    rec = recall_score(y_true, predictions, average="macro", zero_division=0)
    score = f1_score(y_true, predictions, average="macro", zero_division=0)

    print(f"Train-precision: {prec:<10.4f} | train_recall: {rec:<10.4f} | train_f1_score: {score:<10.4f} ")
    

@dataclass
class InteractivePlot:
    filename: str = field(default_factory=str)
    
    train_losses: list = field(default_factory=list)
    train_accuracies: list = field(default_factory=list)
    train_precision: list = field(default_factory=list)
    train_recall: list = field(default_factory=list)
    train_f1_score: list = field(default_factory=list)
    
    val_losses: list = field(default_factory=list)
    val_accuracies: list = field(default_factory=list)
    val_precision: list = field(default_factory=list)
    val_recall: list = field(default_factory=list)
    val_f1_score: list = field(default_factory=list)


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

    def create_network(self, layers: list[DenseLayer], input_shape: int) -> list[DenseLayer]:
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
            assert loss_func in L
            
            ## Training Set   
            # Feed Forward         
            forward_output: pd.DataFrame = X_train
            for layer in network:
                forward_output = layer.forward(forward_output)

            # Calculate metrics
            loss = L[loss_func](y_train, forward_output)
            train_acc = accuracy(y_train, forward_output)
            
            x_train_pred = (forward_output > 0.5).astype(int)
            
            train_precision = precision_score(y_train, x_train_pred, average="macro", zero_division=0)
            train_recall = recall_score(y_train, x_train_pred, average="macro", zero_division=0)
            train_f1_score = f1_score(y_train, x_train_pred, average="macro", zero_division=0)
            
            self.int_plot.train_losses.append(loss)
            self.int_plot.train_accuracies.append(train_acc)
            self.int_plot.train_precision.append(train_precision)
            self.int_plot.train_recall.append(train_recall)
            self.int_plot.train_f1_score.append(train_f1_score)

            # Backpropagation
            delta = forward_output - y_train
            for layer in reversed(network):
                delta = layer.backward(delta, learning_rate)


            ## Validation Set
            # Inference
            val_output: pd.DataFrame = X_valid
            for layer in network:
                val_output = layer.forward(val_output)

            # Calculate metrics
            val_loss = L[loss_func](y_valid, val_output)
            val_acc = accuracy(y_valid, val_output)

            y_val_pred = (val_output > 0.5).astype(int)

            val_precision = precision_score(y_valid, y_val_pred, average="macro", zero_division=0)
            val_recall = recall_score(y_valid, y_val_pred, average="macro", zero_division=0)
            val_f1_score = f1_score(y_valid, y_val_pred, average="macro", zero_division=0)

            
            self.int_plot.val_accuracies.append(val_acc)
            self.int_plot.val_losses.append(val_loss)
            self.int_plot.val_precision.append(val_precision)
            self.int_plot.val_recall.append(val_recall)
            self.int_plot.val_f1_score.append(val_f1_score)

            train_metrics = \
                f"{'Epoch ' + str(epoch + 1) + '/' + str(epochs):<10} | " \
                f"loss: {loss:<8.4f} | " \
                f"acc: {train_acc:<8.4f} | " \
                f"prec: {train_precision:<8.4f} | " \
                f"recall: {train_recall:<8.4f} | " \
                f"f1: {train_f1_score:<8.4f}" \

            validation_metrics = \
                f"{'Validation':<{10+len(train_metrics)-94}} | " \
                f"loss: {val_loss:<8.4f} | " \
                f"acc: {val_acc:<8.4f} | " \
                f"prec: {val_precision:<8.4f} | " \
                f"recall: {val_recall:<8.4f} | " \
                f"f1: {val_f1_score:<8.4f}" \

             # If not the first epoch, move the cursor up two lines to overwrite
            if epoch > 0:
                sys.stdout.write("\033[F\033[K")  # Move cursor up one line and clear it
                sys.stdout.write("\033[F\033[K")  # Repeat for second line

            # Print the metrics
            print(train_metrics)
            print(validation_metrics)

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
        fix, axes = plt.subplots(5, 1, figsize=(10, 8))
        
        ax, ax2, ax3, ax4, ax5 = axes

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

        # Subplot for precision
        line5, = ax3.plot(self.int_plot.train_precision, label="Training Precision")
        line6, = ax3.plot(self.int_plot.val_precision, label="Validation Precision")
        ax3.legend()
        ax3.set_xlabel("Epochs")
        ax3.set_ylabel("Precision")
        ax3.set_title("Precision by epochs")        

        # Subplot for recall
        line7, = ax4.plot(self.int_plot.train_recall, label="Training Recall")
        line8, = ax4.plot(self.int_plot.val_recall, label="Validation Recall")
        ax4.legend()
        ax4.set_xlabel("Epochs")
        ax4.set_ylabel("Recall")
        ax4.set_title("Recall by epochs")  

        # Subplot for f1 score
        line9, = ax5.plot(self.int_plot.train_f1_score, label="Training F1 Score")
        line10, = ax5.plot(self.int_plot.val_f1_score, label="Validation F1 Score")
        ax5.legend()
        ax5.set_xlabel("Epochs")
        ax5.set_ylabel("F1 Score")
        ax5.set_title("F1 Score by epochs")  
        
        return ax, ax2, ax3, ax4, ax5, line1, line2, line3, line4, line5, line6, line7, line8, line9, line10


    def update_figure(self, ax, ax2, ax3, ax4, ax5, line1, line2, line3, line4, line5, line6, line7, line8, line9, line10):
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

        # Update interactive plot for precision
        line5.set_ydata(self.int_plot.train_precision)
        line5.set_xdata(range(len(self.int_plot.train_precision)))
        line6.set_ydata(self.int_plot.val_precision)
        line6.set_xdata(range(len(self.int_plot.val_precision)))
        ax3.relim()
        ax3.autoscale_view()

        # Update interactive plot for recall
        line7.set_ydata(self.int_plot.train_recall)
        line7.set_xdata(range(len(self.int_plot.train_recall)))
        line8.set_ydata(self.int_plot.val_recall)
        line8.set_xdata(range(len(self.int_plot.val_recall)))
        ax4.relim()
        ax4.autoscale_view()

        # Update interactive plot for f1 score
        line9.set_ydata(self.int_plot.train_f1_score)
        line9.set_xdata(range(len(self.int_plot.train_f1_score)))
        line10.set_ydata(self.int_plot.val_f1_score)
        line10.set_xdata(range(len(self.int_plot.val_f1_score)))
        ax5.relim()
        ax5.autoscale_view()

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

