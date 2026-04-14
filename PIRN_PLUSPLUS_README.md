# PIRN++ (BPA++ / APR++ / MNC++) Integration Guide

This package upgrades PIRN with three state-of-the-art directions:

- `BPA++`: confidence-aware partial/unbalanced OT routing with dustbin mass.
- `APR++`: uncertainty-gated prototype refinement + EMA teacher consistency for TTA-friendliness.
- `MNC++`: sequential information-bottleneck communication for robust multimodal fusion.

Implemented modules:

- `src/pirn_pp/modules/bpa_pp.py`
- `src/pirn_pp/modules/apr_pp.py`
- `src/pirn_pp/modules/mnc_pp.py`
- `src/pirn_pp/model.py`

## Quick Start

```python
import torch
from pirn_pp import PIRNPlusPlus, PIRNPPConfig

cfg = PIRNPPConfig(dim=256, num_proto_rgb=32, num_proto_xyz=32)
model = PIRNPlusPlus(cfg).cuda()

f_rgb = torch.randn(4, 196, 256).cuda()
f_xyz = torch.randn(4, 196, 256).cuda()
target = torch.randint(0, 2, (4, 196)).float().cuda()

out = model(f_rgb, f_xyz, update_teacher=True)
losses = model.compute_loss(out, target)
losses["total"].backward()
```

## Mapping to Original PIRN Modules

- Original `BPA` -> `BPAPlusPlus`
- Original `APR` -> `APRPlusPlus`
- Original `MNC` -> `MNCPlusPlus`

## What Changed

### 1) BPA++ (Assignment)

- Keeps Sinkhorn-style balanced routing behavior but adds a dustbin channel.
- Dustbin supports partial/unbalanced assignment so uncertain tokens are not forced into normal prototypes.
- Exposes per-token confidence (`conf_rgb`, `conf_xyz`) for downstream filtering.
- Adds cross-modal harmonization signal to better align RGB/3D semantic partitions.

Main knobs:

- `sinkhorn_tau`
- `sinkhorn_iters`
- `assign_mass_floor`

### 2) APR++ (Refinement)

- Refines prototypes with GRU memory update.
- Uses uncertainty gating so only high-confidence + low-entropy tokens update prototypes.
- Adds EMA teacher prototypes and consistency penalty.
- Supports training-free adaptation behavior at test time through `update_teacher=True/False`.

Main knobs:

- `ema_momentum`
- `update_uncertainty_threshold`
- `consistency_weight`
- `entropy_reg_weight`

### 3) MNC++ (Communication)

- Replaces one-shot cross-attention with sequential communication blocks.
- Applies information-bottleneck regularization proxy (`ib_loss`) to prevent over-transfer noise.
- More robust under modality corruption or partial missing inputs.

Main knobs:

- `mnc_steps`
- `ib_beta`

## Recommended Training Recipe

1. Warm-up (5-10 epochs):
- Lower `consistency_weight` and `ib_beta`.
- Keep `assign_mass_floor` small (0.03-0.05).

2. Main phase:
- Increase `consistency_weight` to stabilize prototypes.
- Increase `mnc_steps` from 1 to 2 or 3.

3. TTA-like finetune or online phase:
- Freeze backbone encoders.
- Keep `update_teacher=True`.
- Adapt only prototypes and fusion blocks.

## Optimization Ideas (Practical + Academic)

- Backbone modernization: replace encoders with stronger frozen representations (for example DINOv2/SigLIP-family), while keeping projection heads trainable.
- Better OT routing: replace current simplified sinkhorn mass constraints with explicit unbalanced OT objective and add class-conditional or region-conditional transport costs.
- Better refinement: use confidence-calibrated uncertainty (temperature scaling + energy score) instead of raw entropy and add dual-memory banks (short-term + long-term) for non-stationary domains.
- Better multimodal communication: add modality dropout for missing-modality robustness and uncertainty-aware gating where low-confidence modalities contribute less.
- Compute optimization: use mixed precision (`torch.cuda.amp.autocast`) and gradient checkpointing, plus adaptive token pooling for high-resolution inputs.

## Suggested Ablations

- `assign_mass_floor`: `[0.00, 0.03, 0.05, 0.10]`
- `update_uncertainty_threshold`: `[0.40, 0.55, 0.70]`
- `mnc_steps`: `[1, 2, 3]`
- `ib_beta`: `[0.00, 0.01, 0.03, 0.05]`
- with/without EMA consistency

## Notes

- The implementation is framework-agnostic and can be inserted into existing PIRN training loops.
- If your current code already defines losses/heads, keep them and only swap module internals.
- `pirn_plusplus.py` is now a backward-compatible shim that re-exports the modular package API.
