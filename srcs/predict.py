import pandas as pd
import numpy as np
from model import MultiLayerPerceptron
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score


def main():
    model = MultiLayerPerceptron()

    model.load_model("model.json")

    data = pd.read_csv("../Validation_data (3).csv")
    input_shape = data.shape[1]
    output_shape = 2

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
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()