from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


def _to_token_feature(x: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor to [N, D] token feature format.
    Supported:
    - [N, D]
    - [H, W, D]
    - [D, H, W]
    """
    if x.ndim == 2:
        return x.float()
    if x.ndim == 3:
        # Heuristic: if channel first [D,H,W], move D to last.
        if x.shape[0] < x.shape[-1] and x.shape[0] < x.shape[-2]:
            x = x.permute(1, 2, 0)
        h, w, d = x.shape
        return x.reshape(h * w, d).float()
    raise ValueError(f"Unsupported feature shape: {tuple(x.shape)}")


def _load_tensor_from_dict(obj: Dict[str, torch.Tensor], keys: Sequence[str]) -> Optional[torch.Tensor]:
    for k in keys:
        if k in obj and isinstance(obj[k], torch.Tensor):
            return obj[k]
    return None


@dataclass
class FeatureSample:
    rgb: torch.Tensor
    sn: torch.Tensor
    label: int
    path: str


class PIRNFeatureDataset(Dataset):
    """
    Dataset format (pt files):
      train/normal/*.pt
      test/normal/*.pt
      test/anomaly/*.pt

    Each pt is dict-like and should include RGB and SN features.
    Accepted keys:
      RGB: rgb, f_rgb, feat_rgb, feature_rgb
      SN : sn, xyz, f_sn, f_xyz, feat_sn, feature_sn
    """

    def __init__(self, data_root: str, split: str) -> None:
        super().__init__()
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        self.root = Path(data_root)
        self.split = split
        self.samples: List[Tuple[Path, int]] = []
        self._collect()
        if not self.samples:
            raise RuntimeError(f"No samples found under {self.root} for split={split}")

    def _collect(self) -> None:
        if self.split == "train":
            normal_dir = self.root / "train" / "normal"
            self.samples.extend((p, 0) for p in sorted(normal_dir.glob("*.pt")))
            return

        test_normal = self.root / "test" / "normal"
        test_anomaly = self.root / "test" / "anomaly"
        self.samples.extend((p, 0) for p in sorted(test_normal.glob("*.pt")))
        self.samples.extend((p, 1) for p in sorted(test_anomaly.glob("*.pt")))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> FeatureSample:
        path, label = self.samples[idx]
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict):
            raise ValueError(f"Expected dict in {path}, got {type(obj)}")

        rgb = _load_tensor_from_dict(obj, ["rgb", "f_rgb", "feat_rgb", "feature_rgb"])
        sn = _load_tensor_from_dict(obj, ["sn", "xyz", "f_sn", "f_xyz", "feat_sn", "feature_sn"])
        if rgb is None or sn is None:
            raise KeyError(f"Missing RGB/SN keys in {path}")

        rgb = _to_token_feature(rgb)
        sn = _to_token_feature(sn)
        if rgb.shape[0] != sn.shape[0]:
            raise ValueError(f"Token count mismatch in {path}: rgb={rgb.shape}, sn={sn.shape}")

        return FeatureSample(rgb=rgb, sn=sn, label=label, path=str(path))


def _pad_tokens(batch: List[FeatureSample]) -> Dict[str, torch.Tensor]:
    max_tokens = max(s.rgb.shape[0] for s in batch)
    dim = batch[0].rgb.shape[1]
    b = len(batch)
    rgb = torch.zeros((b, max_tokens, dim), dtype=torch.float32)
    sn = torch.zeros((b, max_tokens, dim), dtype=torch.float32)
    mask = torch.zeros((b, max_tokens), dtype=torch.bool)
    labels = torch.zeros((b,), dtype=torch.long)

    for i, sample in enumerate(batch):
        n = sample.rgb.shape[0]
        rgb[i, :n] = sample.rgb
        sn[i, :n] = sample.sn
        mask[i, :n] = True
        labels[i] = sample.label

    return {"rgb": rgb, "sn": sn, "mask": mask, "label": labels}


def build_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = PIRNFeatureDataset(data_root=data_root, split="train")
    test_ds = PIRNFeatureDataset(data_root=data_root, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_pad_tokens,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_pad_tokens,
        drop_last=False,
    )
    return train_loader, test_loader

