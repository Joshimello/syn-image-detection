# MediaEval 2026 Synthetic Image Detection

This repository contains a CLIP-based detector for MediaEval 2026 Synthetic
Image Detection Task A.

The detector uses OpenCLIP ViT-L/14 features from every visual transformer block,
a learned block-fusion head, exact observed-score threshold calibration, and
robustness checks for image size, JPEG recompression, and resize/crop
laundering.

## Results

| Run         | Training data                                                          |     F1 | ROC AUC |     AP |
| ----------- | ---------------------------------------------------------------------- | -----: | ------: | -----: |
| Constrained | 75k COCO real + 75k Corvi fake                                         | 0.7555 |  0.8114 | 0.7963 |
| Open        | Constrained data + 60k TrueFake social real + 60k TrueFake social fake | 0.7916 |  0.8643 | 0.8541 |

Pinned run artifacts are generated locally under `artifacts/runs/` and are not
tracked by git. Dataset files and derived metadata are also local-only.

Pinned local artifact names:

```text
constrained_clip_l14_unfreeze_last2_smallimg_moredata_epochs3_20260424_204835/teamname_constrained.csv
open_clip_l14_unfreeze_last2_smallimg_truefake_social_135k_epochs3_20260428_121207/teamname_open.csv
```

## Repository Layout

```text
configs/                  Active run configs
configs/archive/          Historical experiment configs
scripts/                  End-to-end run scripts
src/me26sid/              Training, evaluation, calibration, and export code
tests/                    Unit tests for data, metrics, config, and exports
```

## Setup

This project uses Python 3.11 and `uv`.

```bash
uv sync
```

The active configs expect local datasets under:

```text
data/raw/corvi
data/raw/coco_train2017/train2017
data/raw/itw_val/ITW-SM/0_real
data/raw/itw_val/ITW-SM/1_fake
data/raw/test/taska_test
data/raw/truefake_social
```

These paths are ignored by git.

## Run Pipelines

Constrained run:

```bash
./scripts/run_constrained.sh
```

Open run:

```bash
./scripts/run_open_truefake_social.sh
```

Each script indexes data if needed, trains, calibrates the validation threshold,
runs robustness evaluation, and exports a submission CSV.

## Development Checks

```bash
uv run ruff check .
uv run pytest
```
