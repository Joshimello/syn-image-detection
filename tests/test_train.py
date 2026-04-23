from pathlib import Path

import pandas as pd

from me26sid.config import load_settings
from me26sid.train import limit_training_frame


def test_limit_training_frame_applies_balanced_caps(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
data:
  max_train_real: 2
  max_train_fake: 2
train:
  seed: 123
""".strip(),
        encoding="utf-8",
    )
    settings = load_settings(config_path, run_name_override="smoke")
    frame = pd.DataFrame(
        {
            "image_id": [f"r{i}" for i in range(4)] + [f"f{i}" for i in range(5)],
            "label": [0.0] * 4 + [1.0] * 5,
            "split": ["train"] * 9,
        }
    )

    limited = limit_training_frame(frame, settings)

    assert len(limited) == 4
    assert int((limited["label"] == 0.0).sum()) == 2
    assert int((limited["label"] == 1.0).sum()) == 2
