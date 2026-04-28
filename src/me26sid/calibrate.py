from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from me26sid.config import load_settings
from me26sid.metrics import sweep_thresholds
from me26sid.utils import write_json

console = Console()


def calibrate_main(
    config_path: Path,
    predictions_path: Path | None = None,
    run_name_override: str | None = None,
) -> None:
    settings = load_settings(config_path, run_name_override=run_name_override)
    path = predictions_path or settings.val_predictions_path()
    predictions = pd.read_parquet(path)
    labels = predictions["label"].to_numpy(dtype=int)
    probs = predictions["prob"].to_numpy(dtype=float)
    selected = sweep_thresholds(labels=labels, probs=probs, steps=settings.eval.threshold_grid_size)
    payload = {
        "threshold": selected.threshold,
        "validation_f1": selected.f1,
        "selection_method": "exact_observed_scores",
    }
    write_json(settings.threshold_path(), payload)
    fixed_thresholds = [
        0.000001,
        0.000005,
        0.00001,
        0.00005,
        0.0001,
        0.0005,
        0.001,
        0.002,
        0.003,
        0.005,
        0.01,
        0.05,
        0.1,
        0.2,
        0.5,
    ]
    threshold_metrics: dict[str, dict[str, float | int]] = {}
    for threshold in fixed_thresholds:
        predicted = (probs >= threshold).astype(int)
        tp = int(((predicted == 1) & (labels == 1)).sum())
        fp = int(((predicted == 1) & (labels == 0)).sum())
        fn = int(((predicted == 0) & (labels == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
        threshold_metrics[str(threshold)] = {
            "f1": round(f1, 6),
            "positives": int(predicted.sum()),
        }

    calibration = {
        "best_threshold": payload["threshold"],
        "best_validation_f1": payload["validation_f1"],
        "selection_method": payload["selection_method"],
        "fixed_threshold_metrics": threshold_metrics,
        "real_quantiles": {
            str(q): float(predictions.loc[predictions["label"] == 0, "prob"].quantile(q))
            for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        },
        "fake_quantiles": {
            str(q): float(predictions.loc[predictions["label"] == 1, "prob"].quantile(q))
            for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        },
        "threshold_sweep_summary": build_threshold_sweep_summary(
            labels=labels,
            probs=probs,
            steps=min(settings.eval.threshold_grid_size, 101),
        ),
    }
    write_json(settings.calibration_path(), calibration)
    console.print(payload)


def build_threshold_sweep_summary(
    labels: np.ndarray,
    probs: np.ndarray,
    steps: int,
) -> list[dict[str, float]]:
    summary: list[dict[str, float]] = []
    for threshold in np.linspace(0.0, 1.0, steps):
        predicted = (probs >= threshold).astype(int)
        tp = int(((predicted == 1) & (labels == 1)).sum())
        fp = int(((predicted == 1) & (labels == 0)).sum())
        fn = int(((predicted == 0) & (labels == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
        summary.append({"threshold": float(threshold), "f1": round(float(f1), 6)})
    return summary
