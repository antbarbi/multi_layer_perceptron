import pandas as pd
import numpy as np
from .model.multi_layer_perceptron import MultiLayerPerceptron
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import argparse


def parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_filenames", "-i",
        type=str,
        nargs='+',  # Accept one or more filenames
        help="Configuration files for the prediction phase"
    )
    
    ##TODO add check for file format

    args = parser.parse_args()

    return args.input_filenames


#TODO add check for file format
#TODO add args for model file
#TODO maybe separate plotting metrics

def main(*metric_files, model_file: str = None, data_test: str = None) -> None:
    if not model_file or not data_test:
        print("Exiting: Missing model file or test data file")
        exit(1)
    
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
    accuracy = accuracy_score(true_labels, pred_labels)
    loss = model.loss["binaryCrossentropy"](y, pred)
    print(f"Accuracy: {accuracy}, Loss: {loss}")

    model.load_metrics(*metric_files)
    model.plot_metrics()


if __name__ == "__main__":
    args = parser()
    main(*args)
