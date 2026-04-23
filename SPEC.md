# Constrained pipeline implementation guide

## Build target

The target is a **frozen CLIP ViT L 14 detector with intermediate block fusion**, trained only on the official benchmark data and hardened against online image transformations. This is the best balance of accuracy, training speed, and implementation risk for a single consumer GPU \[Coz23, Kou24\]. The core design choice comes from \[Kou24\]: use intermediate CLIP features rather than only the final embedding, because synthetic traces live more in low and mid level structure than in the most semantic layer. The training recipe borrows the crop first and local bias control ideas from \[Li24\], while the validation protocol explicitly checks for compression and size shortcuts highlighted by \[Gro24\].

## What to implement

| Part            | Final choice                                                              |
| :-------------- | :------------------------------------------------------------------------ |
| Backbone        | Frozen CLIP ViT L 14 image encoder                                        |
| Features        | CLS token from every transformer block                                    |
| Fusion head     | Per block projection, learned softmax block weights, small MLP classifier |
| Input size      | 224                                                                       |
| Training data   | Official training data only                                               |
| Validation      | Official 10k labeled validation set                                       |
| Loss            | Binary cross entropy with logits                                          |
| Model selection | Best validation ROC AUC or AP, then fix threshold by validation F1        |

## End to end pipeline

### 1. Environment

A simple stack is enough:

- Python 3.10 or newer
- PyTorch
- torchvision
- open clip or another CLIP implementation that exposes intermediate blocks
- pandas
- scikit learn for metrics

If ViT L 14 is too tight in memory on the chosen batch size, mixed precision should be enabled from the start. If memory is still tight, gradient accumulation is better than shrinking the crop too aggressively.

### 2. Dataset layout

Prepare three splits:

- official training real
- official training synthetic
- official validation real
- official validation synthetic

Create a csv or dataframe with these fields:

- image_path
- label
- split
- source_dataset
- width
- height
- file_format if easy to extract

That metadata is useful later for the bias checks from \[Gro24\].

### 3. Preprocessing

The main rule is **crop first, do not crush images into a fixed size too early** \[Li24\]. A practical implementation is:

#### Training transform

- decode image at native resolution
- if shortest side is smaller than 256, resize shortest side to 256
- random crop 224 by 224
- random horizontal flip with probability 0.5
- mild ColorJitter with probability 0.3
- small random rotation up to about 5 degrees with probability 0.2
- JPEG augmentation with probability 0.5
- random resize degradation with probability 0.3
- CLIP normalization

#### Validation transform

- decode image at native resolution
- if shortest side is smaller than 256, resize shortest side to 256
- center crop 224 by 224
- CLIP normalization

This follows the spirit of \[Li24\]. Their main point is not that resizing is forbidden, but that heavy resize driven preprocessing weakens the traces the detector needs.

## Transformation details

### JPEG augmentation

Use random JPEG quality in a moderate range, such as 60 to 100. This addresses the social media style laundering stress tested in \[Cor22\] and helps avoid the compression shortcut problem from \[Gro24\].

### Resize degradation

With some probability, downscale the crop to a random side length such as 112 to 196 and resize back to 224. This simulates platform resizing while keeping training cheap.

### Optional patch masking

If early runs overfit too much to scene content, add a light random patch masking step during training, inspired by \[Li24\]. Keep it weak. The goal is to reduce semantic dependence, not destroy the image.

## Model

### Backbone

Use the CLIP ViT L 14 image encoder and keep it frozen \[Coz23, Kou24\]. Do not fine tune the backbone in the first implementation.

### Intermediate feature extraction

Register hooks or modify the forward pass so that the CLS token from each transformer block is collected:

```math
z_1, z_2, \ldots, z_n \in \mathbb{R}^d
```

For ViT L 14, $`n`$ is the number of encoder blocks. \[Kou24\] uses intermediate block representations because they retain more forensic signal than the last layer.

### Fusion head

A simple head is enough:

1.  project each block feature from $`d`$ to a smaller dimension such as 256
2.  learn one scalar logit per block
3.  apply softmax over blocks to get weights
4.  compute weighted sum of projected features
5.  pass through a two layer MLP
6.  output one logit

In symbols:

```math
h_i = W z_i + b
```

```math
\alpha_i = \mathrm{softmax}(a_i)
```

```math
h = \sum_i \alpha_i h_i
```

```math
\hat{y} = \mathrm{MLP}(h)
```

This is a practical simplification of the weighting idea in \[Kou24\]. It preserves the paper’s core advantage without much engineering overhead.

### Head dimensions

A good default:

- projection dim 256
- hidden dim 256
- dropout 0.3 to 0.5 in the head

Because the backbone is frozen, almost all training time sits in feature extraction, not in the head.

## Training recipe

### Batch and optimization

Start here:

- optimizer: AdamW
- learning rate: 1e-3 for the head
- weight decay: 1e-4
- batch size: as large as fits with mixed precision
- epochs: 5 to 10
- early stopping patience: 2 or 3 epochs

\[Kou24\] shows that this design space trains very quickly, because only the lightweight head is learned.

### Loss

Use binary cross entropy with logits. If the official training set is balanced, class weighting is not needed. If it is not balanced, use a positive class weight or a balanced sampler.

### Training schedule

A good order is:

1.  run a short smoke test on a small subset
2.  train the full model for 5 epochs
3.  inspect validation F1 and ROC AUC
4.  if still improving, continue to 8 or 10 epochs
5.  stop once validation plateaus

Within an eight hour budget, this setup should leave room for at least one rerun with improved augmentation or thresholding.

## Thresholding and checkpoint choice

The benchmark uses F1 as the main metric, so threshold choice matters.

### Checkpoint selection

Choose the checkpoint with the best validation ROC AUC or AP. This is more stable than selecting directly by one thresholded F1 value.

### Final threshold

After the checkpoint is fixed, sweep thresholds on the validation set and choose the single global threshold that maximizes F1. Save that threshold and use it for every test row.

Record at least these metrics on validation:

- Accuracy
- Precision
- Recall
- F1
- ROC AUC
- AP

## Bias and robustness checks

This is where many synthetic image detectors fail in practice.

### Compression shortcut check

Following \[Gro24\], evaluate the chosen model on validation images after simulated JPEG at several qualities such as 95, 85, and 75. Watch especially for recall collapse on synthetic images. If recall falls sharply while precision stays high, the model is behaving like a JPEG detector rather than a synthetic image detector.

### Size shortcut check

Bucket validation images by size before preprocessing, such as:

- short side below 512
- 512 to 1024
- above 1024

Compare F1 across buckets. Large swings usually mean the model is exploiting resolution cues \[Gro24\].

### Laundering style stress test

Create a small stress set with:

- JPEG recompression
- downscale and upscale
- center crop and recrop

\[Cor22\] shows that diffusion traces can weaken under these operations, so this test is worth running before the final threshold is locked.

## Suggested project structure

| File        | Purpose                                           |
| :---------- | :------------------------------------------------ |
| dataset.py  | metadata loading and transforms                   |
| model.py    | CLIP wrapper, intermediate hooks, fusion head     |
| train.py    | training loop and checkpointing                   |
| eval.py     | validation metrics, threshold sweep, stress tests |
| predict.py  | test inference and csv export                     |
| config.yaml | paths and hyperparameters                         |

## Minimal training loop

### Per batch

- load image and label
- run CLIP image encoder with intermediate block capture
- pass intermediate CLS tokens to the fusion head
- compute binary cross entropy loss
- backward pass on the head only
- optimizer step

### Per epoch

- train over all batches
- run full validation
- save raw probabilities for threshold sweep
- save best checkpoint

## Export format

For the benchmark csv:

- `image_id`
- `prob`
- `label`
- `threshold`

Use the same global threshold value in every row. The `prob` field should be the sigmoid of the output logit.

## Recommended first run

Use this exact first run before any ablations:

- frozen CLIP ViT L 14
- all intermediate block CLS tokens
- 256 dim projection head with learned block weights
- crop first preprocessing
- JPEG augmentation at probability 0.5 with quality 60 to 100
- resize degradation at probability 0.3
- ColorJitter at probability 0.3
- rotation at probability 0.2
- 5 epochs with AdamW and mixed precision
- checkpoint by validation ROC AUC
- threshold by validation F1

This first run is simple enough to finish quickly and strong enough to be a serious constrained submission \[Coz23, Kou24\].

## If training is unstable or underwhelming

Use this order of fixes:

1.  reduce augmentation strength before changing the model
2.  check whether validation errors cluster by JPEG quality or size
3.  increase dropout in the head
4.  add light patch masking from \[Li24\]
5.  only then consider a more complex detector

Do not jump first to a heavier architecture. \[Gro24\] suggests that many apparent gains in this area come from shortcut learning rather than better forensic reasoning.

## Final recommendation

The most followable constrained pipeline is a frozen CLIP ViT L 14 detector with intermediate block fusion from \[Kou24\], trained with crop first preprocessing and moderate online style augmentation inspired by \[Li24\], and validated with explicit compression and size bias checks from \[Gro24\]. This gives a strong and practical detector without the engineering cost of more elaborate adaptation modules such as \[Liu23b\].

---

## References

\[Coz23\] D. Cozzolino, G. Poggi, R. Corvi, M. Nießner, and L. Verdoliva, “Raising the Bar of AI-generated Image Detection with CLIP,” _2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)_, pp. 4356–4366, Nov. 2023, doi: [10.1109/CVPRW63382.2024.00439](https://doi.org/10.1109/CVPRW63382.2024.00439).

\[Kou24\] C. Koutlis and S. Papadopoulos, “Leveraging Representations from Intermediate Encoder-blocks for Synthetic Image Detection,” _European Conference on Computer Vision_, pp. 394–411, Feb. 2024, doi: [10.48550/arXiv.2402.19091](https://doi.org/10.48550/arXiv.2402.19091).

\[Li24\] O. Li, J. Cai, Y. Hao, X. Jiang, Y. Hu, and F. Feng, “Improving Synthetic Image Detection Towards Generalization: An Image Transformation Perspective,” _Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1_, Aug. 2024, doi: [10.1145/3690624.3709392](https://doi.org/10.1145/3690624.3709392).

\[Gro24\] P. Grommelt, L. Weiss, F. Pfreundt, and J. Keuper, “Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets,” _ECCV Workshops_, pp. 80–95, Mar. 2024, doi: [10.48550/arXiv.2403.17608](https://doi.org/10.48550/arXiv.2403.17608).

\[Cor22\] R. Corvi, D. Cozzolino, G. Zingarini, G. Poggi, K. Nagano, and L. Verdoliva, “On The Detection of Synthetic Images Generated by Diffusion Models,” _ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_, pp. 1–5, Nov. 2022, doi: [10.1109/ICASSP49357.2023.10095167](https://doi.org/10.1109/ICASSP49357.2023.10095167).

\[Liu23b\] H. Liu, Z. Tan, C. Tan, Y. Wei, Y. Zhao, and J. Wang, “Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection,” _2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, pp. 10770–10780, Dec. 2023, doi: [10.1109/CVPR52733.2024.01024](https://doi.org/10.1109/CVPR52733.2024.01024).
