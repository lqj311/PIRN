# PIRN

PIRN research codebase for unsupervised industrial anomaly detection with multimodal features.

This repository contains two tracks:

1. `pirn_pp`: PIRN++ modular blocks (`BPA++/APR++/MNC++`).
2. `pirn_paper`: paper-style PIRN pipeline (`BPA/APR/MNC`) with full training/evaluation scripts.

## Structure

```text
.
├─ src/
│  ├─ pirn_pp/               # PIRN++ modules
│  └─ pirn_paper/            # PIRN paper pipeline
│     ├─ config.py
│     ├─ data.py
│     ├─ metrics.py
│     ├─ model.py
│     ├─ train.py
│     └─ eval.py
├─ examples/
│  ├─ pirn_paper_train_step.py
│  └─ make_toy_feature_dataset.py
└─ frontend/
   ├─ index.html
   ├─ app.js
   ├─ styles.css
   └─ server.py
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
```

## Evaluation

```bash
python -m pirn_paper.eval \
  --data-root data/features \
  --checkpoint runs/pirn_paper/best.pt \
  --batch-size 8 \
  --device cuda
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

