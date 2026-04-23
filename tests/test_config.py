from pathlib import Path

from me26sid.config import load_settings
from me26sid.utils import find_project_root


def test_load_settings_resolves_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
paths:
  corvi_dir: data/raw/corvi
  metadata_path: data/metadata/index.parquet
train:
  run_name: smoke
""".strip(),
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.paths.corvi_dir == (Path.cwd() / "data/raw/corvi").resolve()
    assert settings.paths.metadata_path == (
        find_project_root() / "data/metadata/index.parquet"
    ).resolve()
    assert settings.run_dir().parent == settings.paths.runs_root
    assert settings.train.run_name.startswith("smoke_")


def test_load_settings_prefers_existing_config_relative_path(tmp_path: Path) -> None:
    (tmp_path / "data/raw/corvi").mkdir(parents=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("paths:\n  corvi_dir: data/raw/corvi\n", encoding="utf-8")

    settings = load_settings(config_path)

    assert settings.paths.corvi_dir == (tmp_path / "data/raw/corvi").resolve()
