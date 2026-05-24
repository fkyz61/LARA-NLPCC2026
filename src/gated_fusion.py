"""
Personalized Gated Fusion module for LARA.

For each task T_i, the Coordinator's Collaboration Weight Matrix C(t, u)
aggregates per-task gradient embeddings into a context vector c_i. A
gating MLP then produces per-layer scalar gates z_i^(l) that modulate
the raw gradient g_i^(l) before the optimizer step.

Reference: Section 3.6 of the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GatingNetwork(nn.Module):
    """Per-block scalar gate generator.

    Args:
        d_model: Internal dimension d.
        n_blocks: Number of parameter blocks L (one embedding table +
                  L-1 MLP layers in the paper, L=8).
        hidden: Sequence of hidden widths. Defaults to (128, 64).
    """

    def __init__(
        self,
        d_model: int = 64,
        n_blocks: int = 8,
        hidden: tuple[int, ...] = (128, 64),
    ):
        super().__init__()
        # Input is [c_i; e_i; phi_i^(l)] of shape (2*d + 2,) — c_i and e_i
        # are d-dim, phi_i^(l) is the 2-scalar per-layer summary.
        in_dim = 2 * d_model + 2
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, 1))  # scalar gate
        self.mlp = nn.Sequential(*layers)
        self.n_blocks = n_blocks

    def forward(
        self,
        c_i: torch.Tensor,        # (B, d)    aggregated context for task i
        e_i: torch.Tensor,        # (d,)      frozen LLM embedding for task i
        phi_il: torch.Tensor,     # (B, 2)    per-layer summary for block l
    ) -> torch.Tensor:
        """Return scalar gate z_i^(l) in (0, 1) for the given block."""
        B = c_i.shape[0]
        e_b = e_i.unsqueeze(0).expand(B, -1)
        inp = torch.cat([c_i, e_b, phi_il], dim=-1)
        z = torch.sigmoid(self.mlp(inp)).squeeze(-1)
        return z


def aggregate_context(
    C: torch.Tensor,    # (B, N, N) Collaboration Weight Matrix
    phi: torch.Tensor,  # (B, N, d) per-task gradient embeddings
) -> torch.Tensor:
    """Compute c_i(t, u) = sum_j C_{ij} phi_j for each task i.

    Returns:
        Tensor of shape (B, N, d).
    """
    # (B, N, N) @ (B, N, d) -> (B, N, d)
    return torch.bmm(C, phi)
