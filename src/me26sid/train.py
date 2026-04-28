from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from me26sid.config import Settings, load_settings
from me26sid.data import (
    EvalTransform,
    TrainTransform,
    build_and_save_index,
    build_train_sampler,
    inspect_counts,
    load_metadata_index,
    make_loader,
    split_frame,
)
from me26sid.eval import predict_loader, resolve_device
from me26sid.metrics import compute_binary_metrics
from me26sid.model import SyntheticImageDetector
from me26sid.utils import ensure_dir, serializable, set_seed, write_json, write_text

console = Console()


def snapshot_run_config(settings: Settings, config_path: Path) -> None:
    ensure_dir(settings.run_dir())
    write_text(
        settings.run_dir() / "config.snapshot.yaml",
        config_path.read_text(encoding="utf-8"),
    )
    write_json(
        settings.run_dir() / "run_info.json",
        {
            "run_name": settings.train.run_name,
            "run_dir": str(settings.run_dir()),
            "source_config": str(config_path.resolve()),
        },
    )


def inspect_data_main(config_path: Path, run_name_override: str | None = None) -> None:
    settings = load_settings(config_path, run_name_override=run_name_override)
    frame = build_and_save_index(settings)
    console.print(inspect_counts(frame))


def load_or_build_index(settings: Settings):
    if settings.paths.metadata_path.exists():
        return load_metadata_index(settings)
    return build_and_save_index(settings)


def limit_training_frame(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    real_cap = settings.data.max_train_real
    fake_cap = settings.data.max_train_fake
    if real_cap is None and fake_cap is None:
        return frame

    sampled_frames = []
    seed = settings.train.seed
    for label, cap in ((0.0, real_cap), (1.0, fake_cap)):
        subset = frame.loc[frame["label"] == label]
        if cap is None or len(subset) <= cap:
            sampled_frames.append(subset)
            continue
        sampled_frames.append(
            subset.sample(n=cap, random_state=seed, replace=False).sort_values("image_id")
        )

    return pd.concat(sampled_frames, ignore_index=True).sample(
        frac=1.0,
        random_state=seed,
        replace=False,
    ).reset_index(drop=True)


def train_one_epoch(
    model: SyntheticImageDetector,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor, torch.Tensor]],
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    settings: Settings,
    epoch_index: int,
    total_epochs: int,
) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    total = 0
    epoch_start = time.monotonic()
    total_steps = len(loader)
    optimizer.zero_grad(set_to_none=True)
    for step, (images, labels, _image_ids, _widths, _heights) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            enabled=settings.train.amp and device.type == "cuda",
        ):
            logits = model(images)
            loss = criterion(logits, labels) / settings.train.grad_accum_steps
        scaler.scale(loss).backward()
        if step % settings.train.grad_accum_steps == 0:
            if settings.train.gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), settings.train.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        batch_size = images.shape[0]
        running_loss += float(loss.item()) * settings.train.grad_accum_steps * batch_size
        total += batch_size
        if step % settings.train.log_every_steps == 0:
            elapsed_epoch = time.monotonic() - epoch_start
            avg_step_seconds = elapsed_epoch / step
            remaining_epoch_steps = max(total_steps - step, 0)
            eta_epoch_seconds = avg_step_seconds * remaining_epoch_steps
            completed_epoch_fraction = (epoch_index - 1) + (step / max(total_steps, 1))
            elapsed_run = elapsed_epoch
            avg_epoch_seconds = elapsed_run / max(completed_epoch_fraction, 1e-6)
            remaining_epochs = max(total_epochs - completed_epoch_fraction, 0.0)
            eta_run_seconds = avg_epoch_seconds * remaining_epochs
            console.print(
                {
                    "epoch": epoch_index,
                    "step": f"{step}/{total_steps}",
                    "avg_loss": running_loss / max(total, 1),
                    "elapsed_epoch_s": round(elapsed_epoch, 1),
                    "avg_step_s": round(avg_step_seconds, 3),
                    "eta_epoch_s": round(eta_epoch_seconds, 1),
                    "eta_run_max_s": round(eta_run_seconds, 1),
                }
            )

    if total == 0:
        return 0.0
    return running_loss / total


def save_checkpoint(
    settings: Settings,
    model: SyntheticImageDetector,
    optimizer: AdamW,
    epoch: int,
    best_metric: float,
) -> None:
    payload: dict[str, Any] = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": serializable(settings.model_dump()),
    }
    checkpoint_path = settings.checkpoint_path()
    ensure_dir(checkpoint_path.parent)
    torch.save(payload, checkpoint_path)


def train_main(config_path: Path, run_name_override: str | None = None) -> None:
    settings = load_settings(config_path, run_name_override=run_name_override)
    set_seed(settings.train.seed)
    ensure_dir(settings.run_dir())
    snapshot_run_config(settings, config_path)
    frame = load_or_build_index(settings)
    train_frame = limit_training_frame(split_frame(frame, "train"), settings)
    val_frame = split_frame(frame, "val")
    device = resolve_device(settings.train.device)

    console.print(
        {
            "train_real_cap": settings.data.max_train_real,
            "train_fake_cap": settings.data.max_train_fake,
            "train_rows": len(train_frame),
            "val_rows": len(val_frame),
        }
    )

    train_loader = make_loader(
        frame=train_frame,
        transform=TrainTransform(settings),
        settings=settings,
        shuffle=True,
        drop_last=False,
        sampler=build_train_sampler(train_frame, settings),
    )
    val_loader = make_loader(
        frame=val_frame,
        transform=EvalTransform(settings),
        settings=settings,
        shuffle=False,
        drop_last=False,
    )

    model = SyntheticImageDetector(
        settings.model,
        unfreeze_last_n_blocks=settings.train.unfreeze_last_n_blocks,
    ).to(device)
    head_lr = settings.train.head_learning_rate or settings.train.learning_rate
    parameter_groups: list[dict[str, Any]] = [
        {
            "params": list(model.head_parameters()),
            "lr": head_lr,
            "weight_decay": settings.train.weight_decay,
        }
    ]
    backbone_params = list(model.backbone_parameters())
    if backbone_params:
        parameter_groups.append(
            {
                "params": backbone_params,
                "lr": settings.train.backbone_learning_rate or settings.train.learning_rate,
                "weight_decay": settings.train.weight_decay,
            }
        )
    optimizer = AdamW(parameter_groups)
    scaler = torch.amp.GradScaler(enabled=settings.train.amp and device.type == "cuda")

    best_auc = float("-inf")
    best_ap = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(1, settings.train.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            settings=settings,
            epoch_index=epoch,
            total_epochs=settings.train.epochs,
        )
        model.eval()
        predictions = predict_loader(
            model=model,
            loader=val_loader,
            device=device,
            amp=settings.train.amp,
        )
        predictions.to_parquet(
            settings.run_dir() / f"val_predictions_epoch_{epoch}.parquet",
            index=False,
        )
        metrics = compute_binary_metrics(
            labels=predictions["label"].to_numpy(dtype=np.int64),
            probs=predictions["prob"].to_numpy(dtype=np.float64),
            threshold=0.5,
        )
        metrics["train_loss"] = train_loss
        metrics["epoch"] = float(epoch)
        write_json(settings.run_dir() / f"metrics_epoch_{epoch}.json", metrics)
        improved = (metrics["roc_auc"] > best_auc) or (
            metrics["roc_auc"] == best_auc and metrics["average_precision"] > best_ap
        )
        if improved:
            best_auc = metrics["roc_auc"]
            best_ap = metrics["average_precision"]
            epochs_without_improvement = 0
            predictions.to_parquet(settings.val_predictions_path(), index=False)
            write_json(settings.metrics_path(), metrics)
            save_checkpoint(settings, model, optimizer, epoch=epoch, best_metric=best_auc)
        else:
            epochs_without_improvement += 1

        console.print({"epoch": epoch, **metrics})
        if epochs_without_improvement >= settings.train.early_stopping_patience:
            console.print("Early stopping triggered")
            break
