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
    input_shape = data.shape[1]
    output_shape = 2
    
    #Preprocess
    column_names = [f'col{i+1}' if i != 1 else "type" for i in range(input_shape)]
    data.columns = column_names

    data_train = data.drop(labels="type", axis=1)
    scaler = StandardScaler()
    data_train = scaler.fit_transform(data_train)
    
    input_shape = data_train.shape[1]
    data_valid = data[["type"]]

    ohe = OneHotEncoder(sparse_output=False)
    data_valid = ohe.fit_transform(data_valid)


    model =  MultiLayerPerceptron()

    with open("config.json", "r") as file:
        config = json.load(file)

    network = model.createNetwork([create_layer(layer) for layer in config["layers"]], 31)

    model.fit(
        network,
        data_train,
        data_valid,
        loss_func="binaryCrossentropy",
        learning_rate=0.0001,
        batch_size=8,
        epochs=1300
    )

    



if __name__ == "__main__":
    main()
