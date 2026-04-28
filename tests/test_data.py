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


def test_build_metadata_index_adds_stratified_truefake_social(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
paths:
  corvi_dir: corvi
  coco_train_dir: coco/train2017
  truefake_social_dir: truefake_social
  val_real_dir: itw/ITW-SM/0_real
  val_fake_dir: itw/ITW-SM/1_fake
  test_dir: test/taska_test
  metadata_path: data/metadata/index.parquet
  metadata_csv_path: data/metadata/index.csv
data:
  official_max_train_real: 2
  official_max_train_fake: 2
  truefake_social_max_real: 6
  truefake_social_fake_per_bucket: 1
train:
  seed: 123
""".strip(),
        encoding="utf-8",
    )
    for index in range(4):
        write_image(tmp_path / f"corvi/fake_{index}.png")
        write_image(tmp_path / f"coco/train2017/real_{index}.jpg")
    for platform in ("Facebook", "Telegram", "Twitter"):
        for source in ("FFHQ", "FORLAB"):
            write_image(tmp_path / f"truefake_social/{platform}/Real/{source}/real.jpg")
        for family in ("FLUX.1", "StyleGAN"):
            for index in range(3):
                write_image(
                    tmp_path
                    / f"truefake_social/{platform}/Fake/{family}/fake_{index}.jpg"
                )
    write_image(tmp_path / "itw/ITW-SM/0_real/val_real.jpg")
    write_image(tmp_path / "itw/ITW-SM/1_fake/val_fake.jpg")
    write_image(tmp_path / "test/taska_test/test.jpg")

    settings = load_settings(config_path, run_name_override="smoke")
    frame = build_metadata_index(settings)
    train = frame.loc[frame["split"] == "train"]
    truefake_fake = train.loc[train["source_dataset"] == "truefake_social_fake"]

    assert len(train) == 16
    assert int((train["label"] == 0.0).sum()) == 8
    assert int((train["label"] == 1.0).sum()) == 8
    assert len(truefake_fake) == 6
    assert truefake_fake["image_id"].str.startswith(("Facebook/", "Telegram/", "Twitter/")).all()
