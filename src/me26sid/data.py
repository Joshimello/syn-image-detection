from __future__ import annotations

import io
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from rich.console import Console
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torchvision.transforms import ColorJitter, RandomCrop
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from me26sid.config import Settings
from me26sid.utils import ensure_dir

console = Console()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class SampleRecord:
    image_id: str
    image_path: str
    label: float | None
    split: str
    source_dataset: str
    width: int
    height: int
    file_format: str


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Missing dataset directory: {root}")
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def read_image_metadata(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        width, height = image.size
        image_format = (image.format or path.suffix.lstrip(".")).lower()
    return width, height, image_format


def build_metadata_index(settings: Settings) -> pd.DataFrame:
    records: list[SampleRecord] = []
    specs = [
        ("train", "corvi_latent_diffusion", 1.0, settings.paths.corvi_dir),
        ("train", "coco_train2017", 0.0, settings.paths.coco_train_dir),
        ("val", "itw_real", 0.0, settings.paths.val_real_dir),
        ("val", "itw_fake", 1.0, settings.paths.val_fake_dir),
        ("test", "taska_test", None, settings.paths.test_dir),
    ]
    for split, source, label, root in specs:
        paths = iter_images(root)
        console.print(f"Indexing {len(paths)} images from {root}")
        for path in paths:
            width, height, image_format = read_image_metadata(path)
            records.append(
                SampleRecord(
                    image_id=path.name,
                    image_path=str(path.resolve()),
                    label=label,
                    split=split,
                    source_dataset=source,
                    width=width,
                    height=height,
                    file_format=image_format,
                )
            )

    frame = pd.DataFrame([record.__dict__ for record in records])
    return frame


def save_metadata_index(settings: Settings, frame: pd.DataFrame) -> None:
    ensure_dir(settings.paths.metadata_path.parent)
    frame.to_parquet(settings.paths.metadata_path, index=False)
    frame.to_csv(settings.paths.metadata_csv_path, index=False)


def load_metadata_index(settings: Settings) -> pd.DataFrame:
    return pd.read_parquet(settings.paths.metadata_path)


def build_and_save_index(settings: Settings) -> pd.DataFrame:
    frame = build_metadata_index(settings)
    save_metadata_index(settings, frame)
    return frame


class RandomJPEGCompression:
    def __init__(self, probability: float, quality_min: int, quality_max: int) -> None:
        self.probability = probability
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        quality = random.randint(self.quality_min, self.quality_max)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomResizeDegradation:
    def __init__(self, probability: float, min_side: int, max_side: int, final_size: int) -> None:
        self.probability = probability
        self.min_side = min_side
        self.max_side = max_side
        self.final_size = final_size

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        side = random.randint(self.min_side, self.max_side)
        degraded = TF.resize(image, side, interpolation=InterpolationMode.BICUBIC, antialias=True)
        return TF.resize(
            degraded,
            [self.final_size, self.final_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )


class TrainTransform:
    def __init__(self, settings: Settings) -> None:
        data = settings.data
        self.resize_min_short_side = data.resize_min_short_side
        self.input_size = data.input_size
        self.horizontal_flip_probability = data.horizontal_flip_probability
        self.color_jitter_probability = data.color_jitter_probability
        self.rotation_probability = data.rotation_probability
        self.rotation_degrees = data.rotation_degrees
        self.patch_mask_probability = data.patch_mask_probability
        self.patch_mask_scale_min = data.patch_mask_scale_min
        self.patch_mask_scale_max = data.patch_mask_scale_max
        self.patch_mask_ratio_min = data.patch_mask_ratio_min
        self.patch_mask_ratio_max = data.patch_mask_ratio_max
        self.jpeg = RandomJPEGCompression(
            probability=data.train_jpeg_probability,
            quality_min=data.train_jpeg_quality_min,
            quality_max=data.train_jpeg_quality_max,
        )
        self.resize_degrade = RandomResizeDegradation(
            probability=data.train_resize_degrade_probability,
            min_side=data.train_resize_degrade_min,
            max_side=data.train_resize_degrade_max,
            final_size=data.input_size,
        )
        self.color_jitter = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = ensure_min_short_side(image, self.resize_min_short_side)
        image = TF.crop(image, *RandomCrop.get_params(image, (self.input_size, self.input_size)))
        if random.random() < self.horizontal_flip_probability:
            image = TF.hflip(image)
        if random.random() < self.color_jitter_probability:
            image = self.color_jitter(image)
        if random.random() < self.rotation_probability:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
        image = self.jpeg(image)
        image = self.resize_degrade(image)
        tensor = normalize_clip(image)
        if random.random() < self.patch_mask_probability:
            tensor = apply_random_erasing(
                tensor,
                scale_min=self.patch_mask_scale_min,
                scale_max=self.patch_mask_scale_max,
                ratio_min=self.patch_mask_ratio_min,
                ratio_max=self.patch_mask_ratio_max,
            )
        return tensor


class EvalTransform:
    def __init__(self, settings: Settings) -> None:
        self.resize_min_short_side = settings.data.resize_min_short_side
        self.input_size = settings.data.input_size
        self.num_crops = settings.eval.num_crops

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = ensure_min_short_side(image, self.resize_min_short_side)
        if self.num_crops <= 1:
            image = TF.center_crop(image, [self.input_size, self.input_size])
            return normalize_clip(image)
        crops = generate_eval_crops(
            image=image,
            input_size=self.input_size,
            num_crops=self.num_crops,
        )
        return torch.stack([normalize_clip(crop) for crop in crops], dim=0)


def ensure_min_short_side(image: Image.Image, min_short_side: int) -> Image.Image:
    width, height = image.size
    if min(width, height) >= min_short_side:
        return image
    return TF.resize(image, min_short_side, interpolation=InterpolationMode.BICUBIC, antialias=True)


def normalize_clip(image: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, mean=CLIP_MEAN, std=CLIP_STD)


def apply_random_erasing(
    tensor: torch.Tensor,
    scale_min: float,
    scale_max: float,
    ratio_min: float,
    ratio_max: float,
) -> torch.Tensor:
    channels, height, width = tensor.shape
    area = height * width
    target_area = random.uniform(scale_min, scale_max) * area
    aspect_ratio = random.uniform(ratio_min, ratio_max)
    erase_h = min(height, max(1, int(round(math.sqrt(target_area * aspect_ratio)))))
    erase_w = min(width, max(1, int(round(math.sqrt(target_area / aspect_ratio)))))
    top = random.randint(0, max(height - erase_h, 0))
    left = random.randint(0, max(width - erase_w, 0))
    tensor = tensor.clone()
    tensor[:, top : top + erase_h, left : left + erase_w] = 0.0
    return tensor


def generate_eval_crops(image: Image.Image, input_size: int, num_crops: int) -> list[Image.Image]:
    width, height = image.size
    crop_w = min(input_size, width)
    crop_h = min(input_size, height)
    if num_crops >= 5:
        return [crop.copy() for crop in TF.five_crop(image, [crop_h, crop_w])]
    if width >= height:
        offsets = [0, max((width - crop_w) // 2, 0), max(width - crop_w, 0)]
        top = max((height - crop_h) // 2, 0)
        return [TF.crop(image, top, left, crop_h, crop_w) for left in offsets]
    offsets = [0, max((height - crop_h) // 2, 0), max(height - crop_h, 0)]
    left = max((width - crop_w) // 2, 0)
    return [TF.crop(image, top, left, crop_h, crop_w) for top in offsets]


class SyntheticImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str, int, int]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        transform: TrainTransform | EvalTransform,
        pil_transform: Callable[[Image.Image], Image.Image] | None = None,
    ) -> None:
        self.records = frame.to_dict(orient="records")
        self.transform = transform
        self.pil_transform = pil_transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, int, int]:
        record = self.records[index]
        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")
            if self.pil_transform is not None:
                image = self.pil_transform(image)
            tensor = self.transform(image)
        label_value = -1.0 if pd.isna(record["label"]) else float(record["label"])
        label = torch.tensor(label_value, dtype=torch.float32)
        return tensor, label, record["image_id"], int(record["width"]), int(record["height"])


def make_loader(
    frame: pd.DataFrame,
    transform: TrainTransform | EvalTransform,
    settings: Settings,
    shuffle: bool,
    drop_last: bool,
    pil_transform: Callable[[Image.Image], Image.Image] | None = None,
    sampler: Sampler[int] | None = None,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor, torch.Tensor]]:
    dataset = SyntheticImageDataset(frame=frame, transform=transform, pil_transform=pil_transform)
    workers = settings.data.num_workers
    return DataLoader(
        dataset,
        batch_size=settings.data.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=settings.data.pin_memory,
        persistent_workers=settings.data.persistent_workers and workers > 0,
    )


def split_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    return frame.loc[frame["split"] == split].reset_index(drop=True)


def inspect_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split, split_df in frame.groupby("split"):
        summary[str(split)] = {
            "count": int(len(split_df)),
            "real": int((split_df["label"] == 0).sum()),
            "fake": int((split_df["label"] == 1).sum()),
        }
    return summary


def jpeg_recompress(quality: int):
    def inner(image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    return inner


def laundering_transform(image: Image.Image) -> Image.Image:
    width, height = image.size
    down_w = max(32, int(width * 0.5))
    down_h = max(32, int(height * 0.5))
    image = image.resize((down_w, down_h), Image.Resampling.BICUBIC)
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    crop_w = max(32, int(width * 0.9))
    crop_h = max(32, int(height * 0.9))
    image = TF.center_crop(image, [crop_h, crop_w])
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    return image


def build_train_sampler(frame: pd.DataFrame, settings: Settings) -> Sampler[int] | None:
    threshold = settings.data.small_image_oversample_threshold
    factor = settings.data.small_image_oversample_factor
    if threshold is None or factor <= 1.0:
        return None

    weights = pd.Series(1.0, index=frame.index, dtype=float)
    short_side = frame[["width", "height"]].min(axis=1)
    for _label, subset in frame.groupby("label"):
        label_weights = pd.Series(1.0, index=subset.index, dtype=float)
        small_mask = short_side.loc[subset.index] < threshold
        label_weights.loc[small_mask] *= factor
        label_weights *= len(subset) / label_weights.sum()
        weights.loc[subset.index] = label_weights

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights.to_numpy(), dtype=torch.double),
        num_samples=len(frame),
        replacement=True,
    )
