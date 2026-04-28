#!/usr/bin/env bash
set -euo pipefail

INCLUDE_OPTIONAL=0
if [[ "${1:-}" == "--include-optional" ]]; then
  INCLUDE_OPTIONAL=1
fi

CONFIGS=(
  "configs/constrained_unfreeze_last2_smallimg_moredata_seed2024.yaml"
  "configs/constrained_unfreeze_last2_smallimg_moredata_epochs3.yaml"
  "configs/constrained_unfreeze_last2_smallimg_moredata_100k.yaml"
  "configs/constrained_unfreeze_last2_smallimg_moredata_backbone5e6.yaml"
  "configs/constrained_unfreeze_last2_smallimg_moredata_patchmask05.yaml"
)

if [[ "${INCLUDE_OPTIONAL}" == "1" ]]; then
  CONFIGS+=("configs/constrained_unfreeze_last2_smallimg_moredata_head2e4.yaml")
fi

echo "Running next optimization batch"
for config in "${CONFIGS[@]}"; do
  echo "Starting ${config}"
  ./scripts/run_constrained.sh "${config}"
done

uv run me26sid-compare-runs
