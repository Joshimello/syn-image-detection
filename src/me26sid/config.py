from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from me26sid.utils import find_project_root, resolve_input_path, resolve_output_path


class PathsConfig(BaseModel):
    corvi_dir: Path = Path("data/raw/corvi")
    coco_train_dir: Path = Path("data/raw/coco_train2017")
    val_real_dir: Path = Path("data/raw/itw_val/ITW-SM/0_real")
    val_fake_dir: Path = Path("data/raw/itw_val/ITW-SM/1_fake")
    test_dir: Path = Path("data/raw/test/taska_test")
    metadata_path: Path = Path("data/metadata/index.parquet")
    metadata_csv_path: Path = Path("data/metadata/index.csv")
    runs_root: Path = Path("artifacts/runs")


class DataConfig(BaseModel):
    input_size: int = 224
    resize_min_short_side: int = 256
    max_train_real: int | None = 50_000
    max_train_fake: int | None = 50_000
    small_image_oversample_threshold: int | None = None
    small_image_oversample_factor: float = 1.0
    patch_mask_probability: float = 0.0
    patch_mask_mode: str = "random_erasing"
    patch_mask_scale_min: float = 0.02
    patch_mask_scale_max: float = 0.10
    patch_mask_ratio_min: float = 0.3
    patch_mask_ratio_max: float = 3.3
    train_jpeg_probability: float = 0.5
    train_jpeg_quality_min: int = 60
    train_jpeg_quality_max: int = 100
    train_resize_degrade_probability: float = 0.3
    train_resize_degrade_min: int = 112
    train_resize_degrade_max: int = 196
    color_jitter_probability: float = 0.3
    rotation_probability: float = 0.2
    rotation_degrees: float = 5.0
    horizontal_flip_probability: float = 0.5
    batch_size: int = 16
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True


class ModelConfig(BaseModel):
    model_name: str = "ViT-L-14"
    pretrained: str = "openai"
    projection_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.4


class TrainConfig(BaseModel):
    run_name: str = "constrained_clip_l14"
    seed: int = 1337
    device: str = "cuda"
    epochs: int = 5
    learning_rate: float = 1e-3
    head_learning_rate: float | None = None
    backbone_learning_rate: float | None = None
    weight_decay: float = 1e-4
    amp: bool = True
    grad_accum_steps: int = 1
    unfreeze_last_n_blocks: int = 0
    gradient_clip_norm: float | None = None
    early_stopping_patience: int = 2
    log_every_steps: int = 50


class EvalConfig(BaseModel):
    threshold_grid_size: int = 1001
    num_crops: int = 1
    jpeg_qualities: list[int] = Field(default_factory=lambda: [95, 85, 75])
    size_bucket_edges: list[int] = Field(default_factory=lambda: [512, 1024])


class OutputConfig(BaseModel):
    team_name: str = "teamname"


class Settings(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    outputs: OutputConfig = Field(default_factory=OutputConfig)

    def run_dir(self) -> Path:
        return self.paths.runs_root / self.train.run_name

    def checkpoint_path(self) -> Path:
        return self.run_dir() / "best.ckpt"

    def val_predictions_path(self) -> Path:
        return self.run_dir() / "val_predictions.parquet"

    def metrics_path(self) -> Path:
        return self.run_dir() / "metrics.json"

    def threshold_path(self) -> Path:
        return self.run_dir() / "threshold.json"

    def calibration_path(self) -> Path:
        return self.run_dir() / "calibration.json"

    def robustness_path(self) -> Path:
        return self.run_dir() / "robustness.json"

    def submission_path(self) -> Path:
        filename = f"{self.outputs.team_name}_constrained.csv"
        return self.run_dir() / filename


def timestamped_run_name(base_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{stamp}"


def load_settings(config_path: Path, run_name_override: str | None = None) -> Settings:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    base_dir = config_path.parent.resolve()
    project_root = find_project_root()
    paths = raw.setdefault("paths", {})
    input_keys = {"corvi_dir", "coco_train_dir", "val_real_dir", "val_fake_dir", "test_dir"}
    output_keys = {"metadata_path", "metadata_csv_path", "runs_root"}
    for key, value in list(paths.items()):
        if value is None:
            continue
        path_value = Path(value)
        if key in input_keys:
            paths[key] = str(resolve_input_path(base_dir, path_value))
        elif key in output_keys:
            paths[key] = str(resolve_output_path(project_root, path_value))
        else:
            paths[key] = str(resolve_input_path(base_dir, path_value))

    train = raw.setdefault("train", {})
    train["run_name"] = run_name_override or timestamped_run_name(
        str(train.get("run_name", TrainConfig().run_name))
    )

    settings = Settings.model_validate(raw)
    return settings
