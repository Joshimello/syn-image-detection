from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from me26sid.utils import find_project_root

console = Console()


def compare_runs_main() -> None:
    runs_root = find_project_root() / "artifacts/runs"
    table = Table(title="Run Summary")
    columns = [
        "run",
        "roc_auc",
        "ap",
        "f1",
        "acc",
        "lt512_f1",
        "laundering_auc",
        "jpeg85_auc",
        "threshold",
    ]
    for column in columns:
        table.add_column(column)

    for run in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        metrics = _read_json(run / "metrics.json")
        robustness = _read_json(run / "robustness.json")
        threshold = _read_json(run / "threshold.json")
        if not metrics:
            continue
        lt512 = robustness.get("size_buckets", {}).get("lt_512", {})
        laundering = robustness.get("laundering", {})
        jpeg85 = robustness.get("jpeg", {}).get("85", {})
        table.add_row(
            run.name,
            _fmt(metrics.get("roc_auc")),
            _fmt(metrics.get("average_precision")),
            _fmt(metrics.get("f1")),
            _fmt(metrics.get("accuracy")),
            _fmt(lt512.get("f1")),
            _fmt(laundering.get("roc_auc")),
            _fmt(jpeg85.get("roc_auc")),
            _fmt(threshold.get("threshold")),
        )

    console.print(table)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"
