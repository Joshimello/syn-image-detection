from __future__ import annotations

from pathlib import Path

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
    payload = {"threshold": selected.threshold, "validation_f1": selected.f1}
    write_json(settings.threshold_path(), payload)
    console.print(payload)
