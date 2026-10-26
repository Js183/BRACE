"""Sentiment metrics on the [-3, 3] regression scale."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def sentiment_metrics(pred, true):
    pred = np.clip(np.nan_to_num(pred.astype(np.float64), nan=0.0, posinf=3.0, neginf=-3.0), -3.0, 3.0)
    true = np.clip(true.astype(np.float64), -3.0, 3.0)
    nonzero = true != 0
    return {
        "acc2": float(accuracy_score(true[nonzero] > 0, pred[nonzero] > 0)),
        "f1": float(f1_score(true[nonzero] > 0, pred[nonzero] > 0, average="weighted", zero_division=0)),
        "acc7": float(np.mean(np.round(pred) == np.round(true))),
        "mae": float(np.mean(np.abs(pred - true))),
    }
