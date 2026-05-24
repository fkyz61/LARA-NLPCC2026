"""
Per-Layer Gradient Summary for LARA.

For each task T_i and each parameter block l, we extract a 2-scalar
summary [cosine_alignment, log_norm] and project the concatenation
across L blocks to a d-dimensional embedding phi_i(t).

Reference: Section 3.4 of the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn

EPS = 1e-8


def per_layer_summary(
    grads: list[torch.Tensor],         # per-task: list of L gradient tensors
    consensus: list[torch.Tensor],     # per-block consensus g_bar^(l)
) -> torch.Tensor:
    """Return tensor of shape (L, 2) for a single task.

    grads[l] and consensus[l] should be flattenable to 1-D vectors.
    """
    feats = []
    for g_l, gbar_l in zip(grads, consensus):
        g_flat = g_l.flatten()
        gbar_flat = gbar_l.flatten()
        cos = (g_flat @ gbar_flat) / (
            g_flat.norm() * gbar_flat.norm() + EPS
        )
        log_norm = torch.log1p(g_flat.norm())
        feats.append(torch.stack([cos, log_norm]))
    return torch.stack(feats)  # (L, 2)


class GradientProjection(nn.Module):
    """Project the L*2-dim summary to a d-dimensional embedding phi_i."""

    def __init__(self, n_blocks: int, d_model: int, hidden: int = 64):
        super().__init__()
        in_dim = n_blocks * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        """summary: (B, N, L, 2) -> phi: (B, N, d)."""
        B, N, L, _ = summary.shape
        flat = summary.view(B, N, L * 2)
        return self.net(flat)
