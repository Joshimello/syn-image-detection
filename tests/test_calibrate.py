from pathlib import Path

import pandas as pd

from me26sid.calibrate import calibrate_main


def test_calibrate_writes_calibration_artifact(tmp_path: Path) -> None:
    runs_root = tmp_path / "artifacts/runs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  runs_root: {runs_root}
train:
  run_name: smoke
eval:
  threshold_grid_size: 11
""".strip(),
        encoding="utf-8",
    )
    predictions = pd.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "prob": [0.1, 0.2, 0.8, 0.9],
        }
    )
    predictions_path = tmp_path / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)

    calibrate_main(config_path, predictions_path=predictions_path, run_name_override="smoke")

    calibration_path = runs_root / "smoke/calibration.json"
    assert calibration_path.exists()
