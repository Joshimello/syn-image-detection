#!/usr/bin/env bash
set -euo pipefail

WITH_TRAINING=0
if [[ "${1:-}" == "--with-training" ]]; then
  WITH_TRAINING=1
fi
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

BEST_RUN="constrained_clip_l14_smallimg_trainonly_20260423_231236"
MOREDATA_RUN="constrained_clip_l14_smallimg_moredata_20260424_115152"
UNFREEZE_LAST2_RUN="constrained_clip_l14_unfreeze_last2_20260424_110242"

echo "Running exact recalibration and logit ensemble candidates"

uv run me26sid-ensemble \
  --config configs/archive/constrained_smallimg_trainonly.yaml \
  --run-name "recal_smallimg_trainonly_exact_${RUN_STAMP}" \
  --source-run "${BEST_RUN}"

uv run me26sid-ensemble \
  --config configs/archive/constrained_unfreeze_last2.yaml \
  --run-name "recal_unfreeze_last2_exact_${RUN_STAMP}" \
  --source-run "${UNFREEZE_LAST2_RUN}"

uv run me26sid-ensemble \
  --config configs/archive/constrained_smallimg_moredata.yaml \
  --run-name "ensemble_moredata_unfreeze_last2_logit_${RUN_STAMP}" \
  --source-run "${MOREDATA_RUN}" \
  --source-run "${UNFREEZE_LAST2_RUN}"

uv run me26sid-ensemble \
  --config configs/archive/constrained_smallimg_trainonly.yaml \
  --run-name "ensemble_best_moredata_unfreeze_last2_logit_${RUN_STAMP}" \
  --source-run "${BEST_RUN}" \
  --source-run "${MOREDATA_RUN}" \
  --source-run "${UNFREEZE_LAST2_RUN}"

if [[ "${WITH_TRAINING}" == "1" ]]; then
  echo "Running optional GPU training candidates"
  ./scripts/run_constrained.sh configs/archive/constrained_unfreeze_last2_smallimg_moredata.yaml
  ./scripts/run_constrained.sh configs/archive/constrained_unfreeze_last2_backbone5e6.yaml
else
  echo "Skipping optional GPU training. Re-run with --with-training to include it."
fi

uv run me26sid-compare-runs
