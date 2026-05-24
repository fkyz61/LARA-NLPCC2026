"""
Coordinator Network for LARA.

Reads three signals at every training step:
    (i)   per-task gradient embeddings phi_i(t),
    (ii)  frozen LLM task embeddings e_i,
    (iii) user-context vector psi_u,
and emits the Collaboration Weight Matrix C(t, u) in R^{N x N} via
row-wise softmax over learned logits.

Reference: Section 3.5 of the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinatorNetwork(nn.Module):
    """Lightweight MLP that maps stacked task signals to an N x N matrix.

    Args:
        n_tasks: Number of tasks N.
        d_model: Internal dimension d shared by phi, e, psi vectors.
        hidden: Sequence of hidden widths for the coordinator MLP.
                Defaults to (512, 256) per the paper.
        use_llm: If False, replace e_i with zeros (ablation: w/o LLM Semantic).
        use_user_context: If False, replace psi_u with zeros
                          (ablation: w/o user conditioning).
    """

    def __init__(
        self,
        n_tasks: int,
        d_model: int = 64,
        hidden: tuple[int, ...] = (512, 256),
        use_llm: bool = True,
        use_user_context: bool = True,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.d_model = d_model
        self.use_llm = use_llm
        self.use_user_context = use_user_context

        # Input to the MLP for a single task slot is [phi; e; psi] -> 3 * d_model.
        # The MLP outputs N logits per task slot, which form one row of C.
        in_dim = 3 * d_model
        out_dim = n_tasks

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        phi: torch.Tensor,        # (B, N, d) per-task gradient embeddings
        e: torch.Tensor,          # (N, d)    frozen LLM task embeddings
        psi_u: torch.Tensor,      # (B, d)    user-context vector
    ) -> torch.Tensor:
        """Return the Collaboration Weight Matrix C of shape (B, N, N).

        C[b, i, j] is the weight task i assigns to gradient info from task j
        for example b in the batch.
        """
        B, N, d = phi.shape
        assert N == self.n_tasks and d == self.d_model

        if not self.use_llm:
            e = torch.zeros_like(e)
        if not self.use_user_context:
            psi_u = torch.zeros_like(psi_u)

        # Broadcast e (N, d) -> (B, N, d) and psi_u (B, d) -> (B, N, d).
        e_b = e.unsqueeze(0).expand(B, -1, -1)
        psi_b = psi_u.unsqueeze(1).expand(-1, N, -1)

        # Stack inputs per task slot: (B, N, 3*d).
        inp = torch.cat([phi, e_b, psi_b], dim=-1)

        # Row logits H in (B, N, N).
        H = self.mlp(inp)

        # Row-wise softmax over the second N axis (the "j" axis).
        C = F.softmax(H, dim=-1)
        return C


class UserContextEncoder(nn.Module):
    """Simple MLP encoder mapping raw user features p_u -> psi_u in R^d.

    The paper uses a low-dimensional user-ID embedding concatenated with
    aggregated engagement statistics; the same architecture is used here.
    """

    def __init__(self, in_dim: int, d_model: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, p_u: torch.Tensor) -> torch.Tensor:
        return self.net(p_u)
