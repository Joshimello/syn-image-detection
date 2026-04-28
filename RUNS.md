# Run Log

This document records constrained and open-run experiments for the MediaEval
2026 synthetic image detection pipeline. The constrained runs use only the
official data available locally: Corvi latent-diffusion fakes plus COCO
train2017 reals. The open run adds TrueFake social-media images. All runs use
the ITW-SM validation set for model selection and thresholding.

## Current Recommendation

The pinned constrained submission candidate is:

`artifacts/runs/constrained_clip_l14_unfreeze_last2_smallimg_moredata_epochs3_20260424_204835`

The pinned open submission candidate is:

`artifacts/runs/open_clip_l14_unfreeze_last2_smallimg_truefake_social_135k_epochs3_20260428_121207`

`configs/constrained.yaml` points to the constrained recipe by default.
`configs/open_truefake_social.yaml` points to the current open recipe. All other
experiment configs are archived under `configs/archive`, and non-pinned run
artifacts are archived under `artifacts/archive/runs`.

Key validation metrics:

| Run | Accuracy | Precision | Recall | F1 | ROC AUC | AP | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Constrained pinned | 0.7318 | 0.6941 | 0.8288 | 0.7555 | 0.8114 | 0.7963 | 0.0001303297 |
| Open TrueFake social | 0.7640 | 0.7087 | 0.8964 | 0.7916 | 0.8643 | 0.8541 | 0.0012207952 |

Challenge-test submission artifacts:

`artifacts/runs/constrained_clip_l14_unfreeze_last2_smallimg_moredata_epochs3_20260424_204835/teamname_constrained.csv`

`artifacts/runs/open_clip_l14_unfreeze_last2_smallimg_truefake_social_135k_epochs3_20260428_121207/teamname_open.csv`

Both CSVs have 10,000 rows and the required columns: `image_id`, `prob`,
`label`, and `threshold`.

## Top Runs

| Run | Status | Key change | Acc | F1 | ROC AUC | AP |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `open_clip_l14_unfreeze_last2_smallimg_truefake_social_135k_epochs3_20260428_121207` | Pinned open | 75k+75k official plus 60k+60k TrueFake social | 0.7640 | 0.7916 | 0.8643 | 0.8541 |
| `constrained_clip_l14_unfreeze_last2_smallimg_moredata_epochs3_20260424_204835` | Pinned | 75k+75k, small-image training, unfreeze last 2, 3 epochs | 0.7318 | 0.7555 | 0.8114 | 0.7963 |
| `constrained_clip_l14_unfreeze_last2_smallimg_moredata_patchmask05_20260424_233013` | Archived backup | 2 epochs plus mild patch masking | 0.7311 | 0.7553 | 0.8099 | 0.7955 |
| `constrained_clip_l14_unfreeze_last2_smallimg_moredata_epochs3_patchmask05_20260425_100647` | Archived | 3 epochs plus patch masking | 0.7205 | 0.7536 | 0.8046 | 0.7863 |
| `constrained_clip_l14_unfreeze_last2_smallimg_moredata_epochs3_seed2025_20260425_120621` | Archived | 3 epochs, seed 2025 | 0.7236 | 0.7530 | 0.8094 | 0.7972 |
| `constrained_clip_l14_unfreeze_last2_smallimg_moredata_20260424_155204` | Archived | 2-epoch predecessor | 0.7116 | 0.7533 | 0.8074 | 0.7928 |
| `ensemble_moredata_unfreeze_last2_logit_20260424_150646` | Archived | Logit ensemble | 0.6993 | 0.7425 | 0.7918 | 0.7797 |
| `constrained_clip_l14_smallimg_trainonly_20260423_231236` | Archived | Best frozen-backbone baseline | 0.6991 | 0.7335 | 0.7796 | 0.7667 |

## Robustness Snapshot

| Run | lt512 F1 | Laundering ROC AUC | JPEG85 ROC AUC |
| --- | ---: | ---: | ---: |
| `open_clip_l14_unfreeze_last2_smallimg_truefake_social_135k_epochs3_20260428_121207` | 0.7156 | 0.8212 | 0.8385 |
| `constrained_clip_l14_unfreeze_last2_smallimg_moredata_epochs3_20260424_204835` | 0.5833 | 0.7826 | 0.7716 |
| `constrained_clip_l14_unfreeze_last2_smallimg_moredata_patchmask05_20260424_233013` | 0.6126 | 0.7838 | 0.7670 |
| `constrained_clip_l14_unfreeze_last2_smallimg_moredata_epochs3_patchmask05_20260425_100647` | 0.6261 | 0.7735 | 0.7662 |
| `constrained_clip_l14_unfreeze_last2_smallimg_moredata_epochs3_seed2025_20260425_120621` | 0.5766 | 0.7818 | 0.7735 |
| `constrained_clip_l14_unfreeze_last2_smallimg_moredata_20260424_155204` | 0.5818 | 0.7773 | 0.7728 |

## Main Lessons

- Exact observed-score threshold selection was necessary. The best thresholds
  are very small, and linear `0..1` sweeps missed useful operating points.
- The strongest constrained recipe is CLIP ViT-L/14 with the last two visual
  transformer blocks unfrozen, moderate small-image oversampling, 75k real plus
  75k fake training caps, and 3 epochs.
- Increasing to `100k + 100k` hurt performance, so more constrained data was
  not automatically better.
- Patch masking helps the weak `<512` bucket, but the best overall F1 still
  came from the non-patchmask 3-epoch run.
- Older frozen-backbone and multi-crop experiments are no longer competitive.
- The TrueFake social open run is a clear improvement over the pinned
  constrained run: F1 improved by 0.0361, ROC AUC by 0.0530, and AP by 0.0577.
  The largest robustness gain is on the `<512` bucket, from 0.5833 to 0.7156 F1.

## Useful Commands

Run the default constrained pipeline:

```bash
./scripts/run_constrained.sh
```

Run the current open pipeline:

```bash
./scripts/run_open_truefake_social.sh
```

Export the pinned challenge-test submission:

```bash
uv run me26sid-export-submission \
  --config configs/constrained.yaml \
  --run-name constrained_clip_l14_unfreeze_last2_smallimg_moredata_epochs3_20260424_204835
```

Export the pinned open challenge-test submission:

```bash
uv run me26sid-export-submission \
  --config configs/open_truefake_social.yaml \
  --run-name open_clip_l14_unfreeze_last2_smallimg_truefake_social_135k_epochs3_20260428_121207
```

Compare currently active runs:

```bash
uv run me26sid-compare-runs
```
