from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.console import Console

from me26sid.config import load_settings
from me26sid.data import EvalTransform, load_metadata_index, make_loader, split_frame
from me26sid.eval import load_model_for_inference, predict_loader, resolve_device
from me26sid.utils import read_json

console = Console()


def export_submission_main(
    config_path: Path,
    checkpoint_override: Path | None = None,
    threshold_override: Path | None = None,
    output_override: Path | None = None,
    run_name_override: str | None = None,
) -> None:
    settings = load_settings(config_path, run_name_override=run_name_override)
    device = resolve_device(settings.train.device)
    frame = load_metadata_index(settings)
    test_frame = split_frame(frame, "test")
    checkpoint_path = checkpoint_override or settings.checkpoint_path()
    threshold_path = threshold_override or settings.threshold_path()
    threshold = float(read_json(threshold_path)["threshold"])
    model = load_model_for_inference(settings, checkpoint_path=checkpoint_path, device=device)
    loader = make_loader(
        frame=test_frame,
        transform=EvalTransform(settings),
        settings=settings,
        shuffle=False,
        drop_last=False,
    )
    predictions = predict_loader(model=model, loader=loader, device=device, amp=settings.train.amp)
    submission = pd.DataFrame(
        {
            "image_id": predictions["image_id"],
            "prob": predictions["prob"],
            "label": (predictions["prob"] >= threshold).astype(int),
            "threshold": threshold,
        }
    ).sort_values("image_id")
    output_path = output_override or settings.submission_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    console.print(
        {
            "submission_path": str(output_path),
            "rows": len(submission),
            "threshold": threshold,
        }
    )
