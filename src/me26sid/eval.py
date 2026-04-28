from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from torch.utils.data import DataLoader

from me26sid.config import Settings, load_settings
from me26sid.data import (
    EvalTransform,
    jpeg_recompress,
    laundering_transform,
    load_metadata_index,
    make_loader,
    split_frame,
)
from me26sid.metrics import compute_binary_metrics, sigmoid
from me26sid.model import SyntheticImageDetector
from me26sid.utils import read_json, write_json

console = Console()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_for_inference(
    settings: Settings,
    checkpoint_path: Path,
    device: torch.device,
) -> SyntheticImageDetector:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SyntheticImageDetector(settings.model)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def predict_loader(
    model: SyntheticImageDetector,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor, torch.Tensor]],
    device: torch.device,
    amp: bool,
) -> pd.DataFrame:
    image_ids: list[str] = []
    labels: list[float] = []
    widths: list[int] = []
    heights: list[int] = []
    logits_list: list[np.ndarray] = []
    for images, batch_labels, batch_image_ids, batch_widths, batch_heights in loader:
        crop_count = 1
        if images.ndim == 5:
            batch_size, crop_count, channels, height, width = images.shape
            images = images.view(batch_size * crop_count, channels, height, width)
        else:
            batch_size = images.shape[0]
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(images)
        if crop_count > 1:
            logits = logits.view(batch_size, crop_count).mean(dim=1)
        logits_np = logits.detach().cpu().float().numpy()
        logits_list.append(logits_np)
        image_ids.extend(list(batch_image_ids))
        labels.extend(batch_labels.numpy().tolist())
        widths.extend(batch_widths.numpy().tolist())
        heights.extend(batch_heights.numpy().tolist())

    logits_concat = (
        np.concatenate(logits_list, axis=0)
        if logits_list
        else np.empty((0,), dtype=np.float32)
    )
    probs = sigmoid(logits_concat)
    return pd.DataFrame(
        {
            "image_id": image_ids,
            "label": labels,
            "width": widths,
            "height": heights,
            "logit": logits_concat,
            "prob": probs,
        }
    )


def evaluate_predictions(predictions: pd.DataFrame, threshold: float) -> dict[str, float]:
    labels = predictions["label"].to_numpy(dtype=np.int64)
    probs = predictions["prob"].to_numpy(dtype=np.float64)
    return compute_binary_metrics(labels=labels, probs=probs, threshold=threshold)


def bucket_by_size(
    predictions: pd.DataFrame,
    edges: list[int],
    threshold: float,
) -> dict[str, dict[str, float]]:
    short_side = np.minimum(predictions["width"].to_numpy(), predictions["height"].to_numpy())
    buckets = {
        f"lt_{edges[0]}": short_side < edges[0],
        f"{edges[0]}_{edges[1]}": (short_side >= edges[0]) & (short_side <= edges[1]),
        f"gt_{edges[1]}": short_side > edges[1],
    }
    output: dict[str, dict[str, float]] = {}
    for name, mask in buckets.items():
        subset = predictions.loc[mask]
        if len(subset) == 0:
            continue
        output[name] = evaluate_predictions(subset, threshold=threshold)
    return output


def run_validation_pass(
    settings: Settings,
    model: SyntheticImageDetector,
    frame: pd.DataFrame,
    device: torch.device,
    pil_transform: Any | None = None,
) -> pd.DataFrame:
    loader = make_loader(
        frame=frame,
        transform=EvalTransform(settings),
        settings=settings,
        shuffle=False,
        drop_last=False,
        pil_transform=pil_transform,
    )
    return predict_loader(model=model, loader=loader, device=device, amp=settings.train.amp)


def run_robustness_suite(
    settings: Settings,
    model: SyntheticImageDetector,
    val_frame: pd.DataFrame,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    robustness: dict[str, Any] = {
        "size_buckets": {},
        "jpeg": {},
    }

    baseline_predictions = run_validation_pass(settings, model, val_frame, device=device)
    robustness["size_buckets"] = bucket_by_size(
        predictions=baseline_predictions,
        edges=settings.eval.size_bucket_edges,
        threshold=threshold,
    )

    for quality in settings.eval.jpeg_qualities:
        predictions = run_validation_pass(
            settings=settings,
            model=model,
            frame=val_frame,
            device=device,
            pil_transform=jpeg_recompress(quality),
        )
        robustness["jpeg"][str(quality)] = evaluate_predictions(predictions, threshold=threshold)

    laundering_predictions = run_validation_pass(
        settings=settings,
        model=model,
        frame=val_frame,
        device=device,
        pil_transform=laundering_transform,
    )
    robustness["laundering"] = evaluate_predictions(laundering_predictions, threshold=threshold)
    return robustness


def eval_main(
    config_path: Path,
    checkpoint_override: Path | None = None,
    run_name_override: str | None = None,
) -> None:
    settings = load_settings(config_path, run_name_override=run_name_override)
    settings.run_dir().mkdir(parents=True, exist_ok=True)
    device = resolve_device(settings.train.device)
    frame = load_metadata_index(settings)
    val_frame = split_frame(frame, "val")
    checkpoint_path = checkpoint_override or settings.checkpoint_path()
    threshold_payload = (
        read_json(settings.threshold_path())
        if settings.threshold_path().exists()
        else {"threshold": 0.5}
    )
    threshold = float(threshold_payload["threshold"])
    model = load_model_for_inference(settings, checkpoint_path=checkpoint_path, device=device)

    predictions = run_validation_pass(settings, model, val_frame, device=device)
    predictions.to_parquet(settings.val_predictions_path(), index=False)
    metrics = evaluate_predictions(predictions, threshold=threshold)
    write_json(settings.metrics_path(), metrics)

    robustness = run_robustness_suite(
        settings=settings,
        model=model,
        val_frame=val_frame,
        device=device,
        threshold=threshold,
    )
    write_json(settings.robustness_path(), robustness)
    console.print({"metrics": metrics, "robustness_path": str(settings.robustness_path())})
