from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    f1: float


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def compute_binary_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (probs >= threshold).astype(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "average_precision": float(average_precision_score(labels, probs)),
    }
    return metrics


def sweep_thresholds(labels: np.ndarray, probs: np.ndarray, steps: int) -> ThresholdSelection:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.0, 1.0, steps):
        predictions = (probs >= threshold).astype(np.int64)
        score = float(f1_score(labels, predictions, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return ThresholdSelection(threshold=best_threshold, f1=best_f1)
