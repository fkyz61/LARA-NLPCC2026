"""
MMoE backbone with shared-bottom structure and per-task ReLU towers.

This is the same backbone used by every method in the paper (LARA,
DRGrad, PCGrad, CAGrad, AdaTask, SNR, PLE baselines, etc.).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class TaskTower(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, int] = (256, 128)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MMoEBackbone(nn.Module):
    """Multi-gate Mixture-of-Experts backbone (Ma et al., 2018).

    Args:
        in_dim: Raw input feature dimension.
        n_experts: Number of shared experts (paper uses 8).
        n_tasks: Number of tasks N.
        expert_hidden: Hidden size of each expert.
        expert_out: Output size of each expert (= tower input dim).
    """

    def __init__(
        self,
        in_dim: int,
        n_experts: int = 8,
        n_tasks: int = 5,
        expert_hidden: int = 512,
        expert_out: int = 256,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_experts = n_experts

        self.experts = nn.ModuleList([
            Expert(in_dim, expert_hidden, expert_out) for _ in range(n_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(in_dim, n_experts) for _ in range(n_tasks)
        ])
        self.towers = nn.ModuleList([
            TaskTower(expert_out) for _ in range(n_tasks)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return list of N per-task logits, each of shape (B,)."""
        # Run all experts: (B, E, out_dim)
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)

        logits = []
        for i in range(self.n_tasks):
            gate = F.softmax(self.gates[i](x), dim=-1)        # (B, E)
            fused = torch.einsum("be,beo->bo", gate, expert_out)
            logits.append(self.towers[i](fused))
        return logits
