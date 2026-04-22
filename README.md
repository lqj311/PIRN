# PIRN

PIRN research codebase for unsupervised industrial anomaly detection with multimodal features.

This repository contains two tracks:

1. `pirn_pp`: PIRN++ modular blocks (`BPA++/APR++/MNC++`).
2. `pirn_paper`: paper-style PIRN pipeline (`BPA/APR/MNC`) with full training/evaluation scripts.

The `pirn_paper` track now also includes an optional dual-memory APR variant:

- base prototypes remain in `BPA` as the stable normality prior
- `APR` maintains a residual memory bank for safer online adaptation
- residual updates are filtered by assignment confidence and entropy
- training/eval logs expose `apr_reliable_ratio` and `apr_support_ratio`

## Structure

```text
.
|- src/
|  |- pirn_pp/               # PIRN++ modules
|  |- pirn_paper/            # PIRN paper pipeline
|     |- config.py
|     |- data.py
|     |- metrics.py
|     |- model.py
|     |- train.py
|     |- eval.py
|- examples/
|  |- pirn_paper_train_step.py
|  |- make_toy_feature_dataset.py
|- frontend/
   |- index.html
   |- app.js
   |- styles.css
   |- server.py
```

## Install

```bash
pip install -e .
```

## Dataset Format (feature-level)

```text
data_root/
  train/
    normal/
      *.pt
  test/
    normal/
      *.pt
    anomaly/
      *.pt
```

Each `*.pt` should be a dict and include:

- RGB feature key in: `rgb / f_rgb / feat_rgb / feature_rgb`
- SN (or 3D) feature key in: `sn / xyz / f_sn / f_xyz / feat_sn / feature_sn`

Supported tensor shapes:

- `[N, D]`
- `[H, W, D]`
- `[D, H, W]`

## Training (paper pipeline)

```bash
python -m pirn_paper.train \
  --data-root data/features \
  --output-dir runs/pirn_paper \
  --epochs 50 \
  --batch-size 8 \
  --device cuda

# dual-memory APR knobs (optional)
#   --apr-confidence-threshold 0.12
#   --apr-entropy-threshold 0.75
#   --apr-residual-scale 0.5
#   --apr-memory-momentum 0.9
#   --apr-residual-weight 0.01
```

## Evaluation

```bash
python -m pirn_paper.eval \
  --data-root data/features \
  --checkpoint runs/pirn_paper/best.pt \
  --batch-size 8 \
  --device cuda

# add --tta to enable online residual-memory adaptation during evaluation
```

## Frontend (experiment console)

```bash
python frontend/server.py
```

Open: `http://localhost:8080`

- configure model/training hyperparameters
- generate train command automatically
- save experiment history in browser storage
- import result JSON and track AUROC

## Quick toy data

```bash
python examples/make_toy_feature_dataset.py --out toy_features --dim 768 --tokens 196
python -m pirn_paper.train --data-root toy_features --output-dir runs/toy --epochs 5 --device cpu
```
