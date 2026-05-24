"""Logging / metrics / config loading utilities."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, TensorDataset


def load_config(path: str | Path) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(data_cfg: dict) -> tuple[Dataset, Dataset]:
    """Load preprocessed train/val tensors from disk.

    The default loader expects four files in `data_cfg["dir"]`:
      - x_train.pt, x_val.pt        (B, in_dim) features
      - p_train.pt, p_val.pt        (B, user_ctx_dim) user-context features
      - y_train.pt, y_val.pt        (B, n_tasks) binary labels
    """
    root = Path(data_cfg["dir"])
    x_tr, x_va = torch.load(root / "x_train.pt"), torch.load(root / "x_val.pt")
    p_tr, p_va = torch.load(root / "p_train.pt"), torch.load(root / "p_val.pt")
    y_tr, y_va = torch.load(root / "y_train.pt"), torch.load(root / "y_val.pt")
    return TensorDataset(x_tr, p_tr, y_tr), TensorDataset(x_va, p_va, y_va)


@torch.no_grad()
def evaluate_auc(
    backbone, loader, n_tasks: int, device: str
) -> list[float]:
    """Compute per-task AUC on the validation loader."""
    backbone.eval()
    ys: list[list[float]] = [[] for _ in range(n_tasks)]
    ps: list[list[float]] = [[] for _ in range(n_tasks)]
    for x, _, labels in loader:
        x, labels = x.to(device), labels.to(device)
        logits = backbone(x)
        for i in range(n_tasks):
            ps[i].extend(torch.sigmoid(logits[i]).cpu().tolist())
            ys[i].extend(labels[:, i].cpu().tolist())
    backbone.train()
    return [round(roc_auc_score(ys[i], ps[i]), 4) for i in range(n_tasks)]
