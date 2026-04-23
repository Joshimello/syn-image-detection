from pathlib import Path

from PIL import Image

from me26sid.config import load_settings
from me26sid.data import build_metadata_index, inspect_counts


def write_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size=size, color=(128, 64, 32)).save(path)


def test_build_metadata_index(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
paths:
  corvi_dir: corvi
  coco_train_dir: coco/train2017
  val_real_dir: itw/ITW-SM/0_real
  val_fake_dir: itw/ITW-SM/1_fake
  test_dir: test/taska_test
  metadata_path: data/metadata/index.parquet
  metadata_csv_path: data/metadata/index.csv
""".strip(),
        encoding="utf-8",
    )
    write_image(tmp_path / "corvi/fake.png", size=(70, 80))
    write_image(tmp_path / "coco/train2017/real.jpg", size=(81, 71))
    write_image(tmp_path / "itw/ITW-SM/0_real/val_real.jpg")
    write_image(tmp_path / "itw/ITW-SM/1_fake/val_fake.jpg")
    write_image(tmp_path / "test/taska_test/test.jpg")

    settings = load_settings(config_path)
    frame = build_metadata_index(settings)
    counts = inspect_counts(frame)

    assert len(frame) == 5
    assert counts["train"]["count"] == 2
    assert counts["val"]["count"] == 2
    assert counts["test"]["count"] == 1
    assert set(frame["image_id"]) == {
        "fake.png",
        "real.jpg",
        "val_real.jpg",
        "val_fake.jpg",
        "test.jpg",
    }
