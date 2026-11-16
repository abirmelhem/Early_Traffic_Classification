"""Fixed-order shared metrics, calculated only from supplied predictions."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .features import CLASSES


def calculate_metrics(y_true, y_pred):
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    if len(true) == 0 or len(true) != len(pred) or not set(true).issubset(CLASSES) or not set(pred).issubset(CLASSES):
        raise ValueError("Invalid prediction lengths or labels")
    if set(true) != set(CLASSES):
        raise ValueError("Every evaluated class must have true support")
    precision, recall, f1, support = precision_recall_fscore_support(true, pred, labels=CLASSES, zero_division=0)
    per_class = pd.DataFrame(dict(candidate_class=CLASSES, precision=precision, recall=recall, f1=f1,
                                  support=support, no_predicted_positives=[not np.any(pred == c) for c in CLASSES]))
    summary = dict(accuracy=float(accuracy_score(true, pred)),
                   macro_f1=float(f1_score(true, pred, labels=CLASSES, average="macro", zero_division=0)),
                   weighted_f1=float(f1_score(true, pred, labels=CLASSES, average="weighted", zero_division=0)),
                   support=len(true))
    return summary, per_class, confusion_matrix(true, pred, labels=CLASSES)


def aggregate_predictions(predictions):
    summaries, classes, matrices = [], [], []
    for n, group in predictions.groupby("n_packets", sort=True):
        if group.flow_id.duplicated().any():
            raise ValueError("Duplicate outer-test prediction for a flow and N")
        summary, per_class, matrix = calculate_metrics(group.y_true, group.y_pred)
        summaries.append(dict(n_packets=int(n), **summary))
        classes.append(per_class.assign(n_packets=int(n)))
        for i, actual in enumerate(CLASSES):
            for j, predicted in enumerate(CLASSES):
                matrices.append(dict(n_packets=int(n), actual_class=actual, predicted_class=predicted,
                                     count=int(matrix[i, j]), row_fraction=float(matrix[i, j]/matrix[i].sum())))
    return pd.DataFrame(summaries), pd.concat(classes, ignore_index=True), pd.DataFrame(matrices)
