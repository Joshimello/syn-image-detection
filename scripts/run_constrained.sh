#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/constrained.yaml}"

if [[ "${CONFIG_PATH}" == "-h" || "${CONFIG_PATH}" == "--help" ]]; then
  echo "Usage: ./scripts/run_constrained.sh [config_path]"
  echo "Default config: configs/constrained.yaml"
  exit 0
fi

BASE_RUN_NAME="$(
  uv run python - <<'PY' "$CONFIG_PATH"
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as handle:
    raw = yaml.safe_load(handle) or {}
print(raw.get("train", {}).get("run_name", "constrained_clip_l14"))
PY
)"

METADATA_PATH="$(
  uv run python - <<'PY' "$CONFIG_PATH"
from pathlib import Path
import sys

from me26sid.config import load_settings

settings = load_settings(Path(sys.argv[1]), run_name_override="preflight")
print(settings.paths.metadata_path)
PY
)"

RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_NAME="${BASE_RUN_NAME}_${RUN_STAMP}"

echo "Running constrained pipeline with config: ${CONFIG_PATH}"
echo "Run name: ${RUN_NAME}"

if [[ -f "${METADATA_PATH}" ]]; then
  echo "Skipping indexing; found existing metadata: ${METADATA_PATH}"
else
  uv run me26sid-inspect-data --config "${CONFIG_PATH}" --run-name "${RUN_NAME}"
fi
uv run me26sid-train --config "${CONFIG_PATH}" --run-name "${RUN_NAME}"
uv run me26sid-calibrate --config "${CONFIG_PATH}" --run-name "${RUN_NAME}"
uv run me26sid-eval --config "${CONFIG_PATH}" --run-name "${RUN_NAME}"
uv run me26sid-export-submission --config "${CONFIG_PATH}" --run-name "${RUN_NAME}"

echo "Finished run: ${RUN_NAME}"
echo "Artifacts: artifacts/runs/${RUN_NAME}"
