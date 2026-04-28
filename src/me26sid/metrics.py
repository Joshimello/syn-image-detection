from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
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


def sweep_thresholds(
    labels: np.ndarray,
    probs: np.ndarray,
    steps: int | None = None,
) -> ThresholdSelection:
    """Select the best F1 threshold from observed validation scores.

    The old fixed grid missed useful thresholds near zero because most model
    scores are saturated. Exact observed-score selection is deterministic and
    avoids making `threshold_grid_size` a hidden performance limit.
    """
    del steps
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = (2.0 * precision * recall) / np.maximum(precision + recall, 1e-12)
    best_index = int(np.nanargmax(f1_scores))
    if best_index >= len(thresholds):
        best_threshold = float(np.nextafter(np.max(probs), np.inf))
    else:
        best_threshold = float(thresholds[best_index])
    return ThresholdSelection(threshold=best_threshold, f1=float(f1_scores[best_index]))
