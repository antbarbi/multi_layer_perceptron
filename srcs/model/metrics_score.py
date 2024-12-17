import numpy as np
from sklearn.metrics import precision_score as sk_precision_score, \
    recall_score as sk_recall_score, f1_score as sk_f1_score, accuracy_score as sk_accuracy_score


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    predictions = (y_pred > 0.5).astype(int)
    
    res = np.mean(predictions == y_true)
    ret = sk_accuracy_score(y_true, predictions)
    
    if not np.isclose(res, ret):
        raise ValueError("Accuracy: Custom implementation is different from sklearn.")
    return res


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_labels = np.argmax(y_true, axis=1) # Return index of maximum value (454, 2) -> (454,)
    y_pred_labels = np.argmax(y_pred, axis=1)
    classes = np.unique(y_true_labels)
    precisions = []
    
    for cls in classes:
        true_positives = np.sum((y_pred_labels == cls) & (y_true_labels == cls))
        predicted_positives = np.sum(y_pred_labels == cls)
    
        if predicted_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / predicted_positives
        precisions.append(precision)
    
    res = np.mean(precisions)
    ret = sk_precision_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
    
    if not np.isclose(res, ret):
        raise ValueError(f"Precision: Custom implementation is different from sklearn.\n{res} != {ret}")
    return res


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_labels = np.argmax(y_true, axis=1) # Return index of maximum value (454, 2) -> (454,)
    y_pred_labels = np.argmax(y_pred, axis=1)
    classes = np.unique(y_true_labels)
    recalls = []

    for cls in classes:
        true_positives = np.sum((y_pred_labels == cls) & (y_true_labels == cls))
        actual_positives = np.sum(y_true_labels == cls)
    
        if actual_positives == 0:
            recall = 0.0
        else:
            recall = true_positives / actual_positives
        recalls.append(recall)

    res = np.mean(recalls)
    ret = sk_recall_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)

    if not np.isclose(res, ret):
        raise ValueError(f"Recall: Custom implementation is different from sklearn.\n{res} != {ret}")
    return res


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_labels = np.argmax(y_true, axis=1) # Return index of maximum value (454, 2) -> (454,)
    y_pred_labels = np.argmax(y_pred, axis=1)
    classes = np.unique(y_true_labels)
    f1s = []

    for cls in classes:
        true_positives = np.sum((y_pred_labels == cls) & (y_true_labels == cls))
        false_positives = np.sum((y_pred_labels == cls) & (y_true_labels != cls))
        false_negatives = np.sum((y_pred_labels != cls) & (y_true_labels == cls))

        if true_positives == 0 and false_positives == 0 and false_negatives == 0:
            f1 = 0.0
        else:
            f1 = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
        f1s.append(f1)
    
    res = np.mean(f1s)
    ret = sk_f1_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
    
    if not np.isclose(res, ret) :
        raise ValueError(f"F1: Custom implementation is different from sklearn.\n{res} != {ret}")
    return res
