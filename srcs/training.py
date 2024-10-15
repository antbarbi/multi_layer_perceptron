from model import MultiLayerPerceptron, DenseLayer
import pandas as pd

def main():
    data = pd.read_csv("../Training_data.csv")

    input_shape = data.shape[1]
    output_shape = 2



    model =  MultiLayerPerceptron()

    network = model.createNetwork([
        DenseLayer(input_shape, activation="sigmoid"),
        DenseLayer(24, activation="sigmoid", weights_initializer="heUniform"),
        DenseLayer(24, activation="sigmoid", weights_initializer="heUniform"),
        DenseLayer(24, activation="sigmoid", weights_initializer="heUniform"),
        DenseLayer(output_shape, activation="softmax", weights_initializer="heUniform"),
    ])


if __name__ == "__main__":
    main()
