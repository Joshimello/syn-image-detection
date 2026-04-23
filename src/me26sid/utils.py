from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_text(path: Path, payload: str) -> None:
    ensure_dir(path.parent)
    path.write_text(payload, encoding="utf-8")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return serializable(asdict(value))
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [serializable(item) for item in value]
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_project_root(start_dir: Path | None = None) -> Path:
    current = (start_dir or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current


def resolve_input_path(base_dir: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    config_relative = (base_dir / value).resolve()
    if config_relative.exists():
        return config_relative
    cwd_relative = (Path.cwd() / value).resolve()
    if cwd_relative.exists():
        return cwd_relative
    return config_relative


def resolve_output_path(project_root: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    return (project_root / value).resolve()
