import pandas as pd
import numpy as np
from .model.multi_layer_perceptron import MultiLayerPerceptron
from .model.metrics_score import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import argparse
import os


def parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_file", type=str, required=True,)
    parser.add_argument("-d", "--data_test", type=str, required=True,)

    return args.input_filenames


def check_args(model_file: str, data_test: str):
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"File {model_file} not found")
    if not model_file.endswith(".json"):
        raise ValueError("Only .json files are accepted for model")
    
    if not os.path.exists(data_test):
        raise FileNotFoundError(f"File {data_test} not found")
    if not data_test.endswith(".csv"):
        raise ValueError("Only .csv files are accepted for data_test")


def main(*metric_files, model_file: str = None, data_test: str = None) -> None:

    check_args(*metric_files, model_file=model_file, data_test=data_test)
    
    model = MultiLayerPerceptron()

    model.load_model(model_file)

    data = pd.read_csv(data_test)
    input_shape = data.shape[1]

    # Preprocess
    column_names = [f'col{i+1}' if i != 1 else "type" for i in range(input_shape)]
    data.columns = column_names
    X = data.drop(labels="type", axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    input_shape = X.shape[1]
    y = data[["type"]]
    ohe = OneHotEncoder(sparse_output=False)
    y = ohe.fit_transform(y)

    # Predict
    pred = model.predict(X)

    # Convert predicted probabilities to class labels
    pred_labels = np.argmax(pred, axis=1)

    # Convert one-hot encoded true labels to class labels
    true_labels = np.argmax(y, axis=1)

    # Calculate accuracy
    res_accuracy = accuracy_score(true_labels, pred_labels)
    loss = model.loss["binaryCrossentropy"](y, pred)
    print(f"Accuracy: {res_accuracy}, Loss: {loss}")


if __name__ == "__main__":
    args = parser()
    main(*args)
