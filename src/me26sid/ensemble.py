from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rich.console import Console

from me26sid.config import Settings, load_settings
from me26sid.data import EvalTransform, load_metadata_index, make_loader, split_frame
from me26sid.eval import (
    bucket_by_size,
    evaluate_predictions,
    jpeg_recompress,
    laundering_transform,
    load_model_for_inference,
    predict_loader,
    resolve_device,
)
from me26sid.metrics import sigmoid, sweep_thresholds
from me26sid.train import snapshot_run_config
from me26sid.utils import ensure_dir, write_json

console = Console()


@dataclass(frozen=True)
class EnsembleSource:
    name: str
    run_dir: Path
    settings: Settings
    checkpoint_path: Path
    val_predictions_path: Path


def ensemble_main(
    config_path: Path,
    source_runs: list[str],
    run_name_override: str | None = None,
) -> None:
    settings = load_settings(config_path, run_name_override=run_name_override)
    ensure_dir(settings.run_dir())
    snapshot_run_config(settings, config_path)
    sources = [resolve_source(settings, source_run) for source_run in source_runs]
    if not sources:
        raise ValueError("At least one source run is required")

    write_json(
        settings.run_dir() / "ensemble_sources.json",
        [
            {
                "name": source.name,
                "run_dir": str(source.run_dir),
                "checkpoint_path": str(source.checkpoint_path),
                "val_predictions_path": str(source.val_predictions_path),
            }
            for source in sources
        ],
    )

    val_predictions = average_saved_predictions(sources)
    val_predictions.to_parquet(settings.val_predictions_path(), index=False)
    labels = val_predictions["label"].to_numpy(dtype=int)
    probs = val_predictions["prob"].to_numpy(dtype=float)
    selected = sweep_thresholds(
        labels=labels,
        probs=probs,
        steps=settings.eval.threshold_grid_size,
    )
    write_json(
        settings.threshold_path(),
        {
            "threshold": selected.threshold,
            "validation_f1": selected.f1,
            "selection_method": "exact_observed_scores",
        },
    )
    metrics = evaluate_predictions(val_predictions, threshold=selected.threshold)
    write_json(settings.metrics_path(), metrics)

    device = resolve_device(settings.train.device)
    models = load_source_models(sources, device=device)
    frame = load_metadata_index(settings)
    val_frame = split_frame(frame, "val")
    robustness = run_ensemble_robustness(
        settings=settings,
        models=models,
        val_frame=val_frame,
        device=device,
        threshold=selected.threshold,
        baseline_predictions=val_predictions,
    )
    write_json(settings.robustness_path(), robustness)

    test_predictions = run_ensemble_prediction(
        settings=settings,
        models=models,
        frame=split_frame(frame, "test"),
        device=device,
    )
    submission = pd.DataFrame(
        {
            "image_id": test_predictions["image_id"],
            "prob": test_predictions["prob"],
            "label": (test_predictions["prob"] >= selected.threshold).astype(int),
            "threshold": selected.threshold,
        }
    ).sort_values("image_id")
    submission.to_csv(settings.submission_path(), index=False)
    console.print(
        {
            "run_dir": str(settings.run_dir()),
            "source_runs": [source.name for source in sources],
            "metrics": metrics,
            "threshold": selected.threshold,
            "submission_path": str(settings.submission_path()),
        }
    )


def resolve_source(settings: Settings, source_run: str) -> EnsembleSource:
    raw_path = Path(source_run)
    run_dir = raw_path if raw_path.exists() else settings.paths.runs_root / source_run
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing source run directory: {run_dir}")

    config_snapshot = run_dir / "config.snapshot.yaml"
    checkpoint_path = run_dir / "best.ckpt"
    val_predictions_path = run_dir / "val_predictions.parquet"
    for path in (config_snapshot, checkpoint_path, val_predictions_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing source artifact: {path}")

    source_settings = load_settings(config_snapshot, run_name_override=run_dir.name)
    return EnsembleSource(
        name=run_dir.name,
        run_dir=run_dir,
        settings=source_settings,
        checkpoint_path=checkpoint_path,
        val_predictions_path=val_predictions_path,
    )


def average_saved_predictions(sources: list[EnsembleSource]) -> pd.DataFrame:
    frames = [
        pd.read_parquet(source.val_predictions_path).sort_values("image_id").reset_index(drop=True)
        for source in sources
    ]
    return average_prediction_frames(frames)


def average_prediction_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    reference = frames[0][["image_id", "label", "width", "height"]].copy()
    logits = []
    for frame in frames:
        aligned = frame.sort_values("image_id").reset_index(drop=True)
        if not aligned["image_id"].equals(reference["image_id"]):
            raise ValueError("Source predictions do not contain matching image_ids")
        logits.append(aligned["logit"].to_numpy(dtype=float))

    mean_logits = np.mean(logits, axis=0)
    output = reference.copy()
    output["logit"] = mean_logits
    output["prob"] = sigmoid(mean_logits)
    return output


def load_source_models(
    sources: list[EnsembleSource],
    device: torch.device,
) -> list[tuple[EnsembleSource, torch.nn.Module]]:
    return [
        (
            source,
            load_model_for_inference(
                source.settings,
                checkpoint_path=source.checkpoint_path,
                device=device,
            ),
        )
        for source in sources
    ]


def run_ensemble_prediction(
    settings: Settings,
    models: list[tuple[EnsembleSource, torch.nn.Module]],
    frame: pd.DataFrame,
    device: torch.device,
    pil_transform: Any | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source, model in models:
        loader = make_loader(
            frame=frame,
            transform=EvalTransform(settings),
            settings=settings,
            shuffle=False,
            drop_last=False,
            pil_transform=pil_transform,
        )
        frames.append(
            predict_loader(
                model=model,
                loader=loader,
                device=device,
                amp=source.settings.train.amp,
            )
        )
    return average_prediction_frames(frames)


def run_ensemble_robustness(
    settings: Settings,
    models: list[tuple[EnsembleSource, torch.nn.Module]],
    val_frame: pd.DataFrame,
    device: torch.device,
    threshold: float,
    baseline_predictions: pd.DataFrame,
) -> dict[str, Any]:
    robustness: dict[str, Any] = {
        "size_buckets": bucket_by_size(
            predictions=baseline_predictions,
            edges=settings.eval.size_bucket_edges,
            threshold=threshold,
        ),
        "jpeg": {},
    }

    for quality in settings.eval.jpeg_qualities:
        predictions = run_ensemble_prediction(
            settings=settings,
            models=models,
            frame=val_frame,
            device=device,
            pil_transform=jpeg_recompress(quality),
        )
        robustness["jpeg"][str(quality)] = evaluate_predictions(predictions, threshold=threshold)

    laundering_predictions = run_ensemble_prediction(
        settings=settings,
        models=models,
        frame=val_frame,
        device=device,
        pil_transform=laundering_transform,
    )
    robustness["laundering"] = evaluate_predictions(
        laundering_predictions,
        threshold=threshold,
    )
    return robustness
