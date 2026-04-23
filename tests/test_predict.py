from pathlib import Path

import pandas as pd

from me26sid.utils import write_json


def test_submission_csv_shape(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        {
            "image_id": ["b.jpg", "a.jpg"],
            "prob": [0.9, 0.1],
        }
    )
    threshold_path = tmp_path / "threshold.json"
    output_path = tmp_path / "submission.csv"
    write_json(threshold_path, {"threshold": 0.5})

    submission = pd.DataFrame(
        {
            "image_id": predictions["image_id"],
            "prob": predictions["prob"],
            "label": (predictions["prob"] >= 0.5).astype(int),
            "threshold": 0.5,
        }
    ).sort_values("image_id")
    submission.to_csv(output_path, index=False)

    exported = pd.read_csv(output_path)
    assert list(exported.columns) == ["image_id", "prob", "label", "threshold"]
    assert exported["image_id"].tolist() == ["a.jpg", "b.jpg"]
    assert exported["threshold"].nunique() == 1
