from .model.multi_layer_perceptron import MultiLayerPerceptron
from .model.layers import DenseLayer
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import json
import argparse
import os


def parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", "--config_filename", type=str, help="The configuration file for the training phase")
    parser.add_argument("--t", "--training_file", type=str, help="The input file for the training phase")
    parser.add_argument("--v", "--validation_file", type=str, help="The input file for the validation phase")
    parser.add_argument("--o", "--output_filename", type=str, help="The output file for the training phase")

    args = parser.parse_args()

    return args.config_filename, args.training_file, args.validation_file, args.output_filename


def check_args(config_file: str, training_dataset: str, validation_dataset: str, output_file: str) -> None:
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"File {config_file} not found")
    if not config_file.endswith(".json"):
        raise ValueError("Only .json files are accepted for config")

    if not os.path.exists(training_dataset):
        raise FileNotFoundError(f"File {training_dataset} not found")
    if not training_dataset.endswith(".csv"):
        raise ValueError("Only .csv files are accepted for training_dataset")

    if not os.path.exists(validation_dataset):
        raise FileNotFoundError(f"File {validation_dataset} not found")
    if not validation_dataset.endswith(".csv"):
        raise ValueError("Only .csv files are accepted for validation_dataset")
    
    if ".json" not in output_file:
        raise ValueError("Output file should be a .json")


def create_layer(config: dict):
    layer = DenseLayer(
        config["units"],
        config["activation"],
        config.get("initializer", None),
        config.get("use_momentum", False),
        config.get("momentum", 0.9)
    )
    return layer


def main(training_dataset, validation_dataset, config_file, output_file):
    check_args(config_file, training_dataset, validation_dataset, output_file)
    
    data = pd.read_csv(training_dataset)
    valid = pd.read_csv(validation_dataset)
    input_shape = data.shape[1]
    
    #Preprocess
    column_names = [f'col{i+1}' if i != 1 else "type" for i in range(input_shape)]
    data.columns = column_names
    valid.columns = column_names

    X_train = data.drop(labels="type", axis=1)
    X_valid = valid.drop(labels="type", axis=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.fit_transform(X_valid)
    
    input_shape = X_train.shape[1]
    y_train = data[["type"]]
    y_valid = valid[["type"]]

    ohe = OneHotEncoder(sparse_output=False)
    y_train = ohe.fit_transform(y_train)
    y_valid = ohe.fit_transform(y_valid)


    model =  MultiLayerPerceptron()

    with open(config_file, "r") as file:
        config = json.load(file)

    network = model.create_network([create_layer(layer) for layer in config["layers"]], 31)

    model.fit(
        network,
        X_train,
        y_train,
        X_valid,
        y_valid,
        loss_func=config["loss_function"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        epochs=config["epochs"]
    )

    model.save_model()
    model.save_metrics(output_file)
    



if __name__ == "__main__":
    args = parser()
    main(*args)
