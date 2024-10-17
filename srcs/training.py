from model import MultiLayerPerceptron, DenseLayer
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import json


def create_layer(config: dict):
    layer = DenseLayer(
        config["units"],
        config["activation"],
        config.get("initializer", None)
    )
    return layer


def main():
    data = pd.read_csv("../Training_data.csv")
    valid = pd.read_csv("../Validation_data.csv")
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

    with open("config.json", "r") as file:
        config = json.load(file)

    network = model.createNetwork([create_layer(layer) for layer in config["layers"]], 31)

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
    



if __name__ == "__main__":
    main()
