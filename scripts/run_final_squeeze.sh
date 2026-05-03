#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/archive/constrained_unfreeze_last2_smallimg_moredata_epochs3_patchmask025.yaml"
  "configs/archive/constrained_unfreeze_last2_smallimg_moredata_epochs3_patchmask05.yaml"
  "configs/archive/constrained_unfreeze_last2_smallimg_moredata_epochs3_backbone5e6.yaml"
  "configs/archive/constrained_unfreeze_last2_smallimg_moredata_epochs3_seed2025.yaml"
  "configs/archive/constrained_unfreeze_last2_smallimg_moredata_epochs3_seed7.yaml"
)

echo "Running final squeeze batch"
for config in "${CONFIGS[@]}"; do
  echo "Starting ${config}"
  ./scripts/run_constrained.sh "${config}"
done

uv run me26sid-compare-runs
