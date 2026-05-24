"""
End-to-end training loop for LARA.

Pseudocode (one step):
    1. forward pass through the MMoE backbone for each task -> losses L_i
    2. compute per-task gradients g_i^(l) on the shared parameter blocks
    3. summarize gradients per layer -> phi_i(t)
    4. read frozen e_i and compute psi_u(t)
    5. Coordinator -> C(t, u)
    6. Personalized Gated Fusion -> z_i^(l)
    7. modulated gradients g_tilde_i^(l) = z_i^(l) * g_i^(l)
    8. apply Adam update with sum of modulated gradients for shared params,
       and per-task modulated gradients for task-specific heads
    9. backprop through gates to update Coordinator + Gating Network end-to-end.

The LLM encoder is frozen throughout.

Reference: Sections 3.2-3.6 of the paper.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .coordinator import CoordinatorNetwork, UserContextEncoder
from .gated_fusion import GatingNetwork, aggregate_context
from .gradient_summary import GradientProjection, per_layer_summary
from .llm_encoder import TaskEmbeddingProjection, cache_or_load
from .mmoe_backbone import MMoEBackbone
from .utils import load_config, set_seed, build_dataset, evaluate_auc

logger = logging.getLogger(__name__)


def _shared_param_blocks(backbone: MMoEBackbone) -> list[torch.nn.Parameter]:
    """Return the L shared parameter blocks (experts + gates).

    The paper partitions theta into one embedding-table block plus L-1
    MLP layer blocks. For the public UCI/MMoE backbone we simply use
    one block per expert layer; the same idea applies.
    """
    blocks = []
    for e in backbone.experts:
        for layer in e.net:
            if isinstance(layer, torch.nn.Linear):
                blocks.append(layer.weight)
    return blocks


def train(config: dict):
    set_seed(config.get("seed", 42))
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # ── Data ──────────────────────────────────────────────────────────
    train_ds, val_ds = build_dataset(config["data"])
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"],
                            shuffle=False, num_workers=4)

    n_tasks = config["n_tasks"]
    d_model = config["d_model"]

    # ── Frozen LLM task embeddings ────────────────────────────────────
    descs = config["task_descriptions"]
    raw_e = cache_or_load(descs, config["llm_cache"],
                          model_name=config["llm_model"]).to(device)
    llm_proj = TaskEmbeddingProjection(raw_e.shape[1], d_model).to(device)

    # ── Models ────────────────────────────────────────────────────────
    backbone = MMoEBackbone(
        in_dim=config["in_dim"],
        n_experts=config["n_experts"],
        n_tasks=n_tasks,
    ).to(device)

    coord = CoordinatorNetwork(
        n_tasks=n_tasks,
        d_model=d_model,
        hidden=tuple(config["coordinator"]["hidden"]),
        use_llm=config["coordinator"].get("use_llm", True),
        use_user_context=config["coordinator"].get("use_user_context", True),
    ).to(device)

    ctx_encoder = UserContextEncoder(
        in_dim=config["user_ctx_dim"], d_model=d_model
    ).to(device)

    grad_proj = GradientProjection(
        n_blocks=len(_shared_param_blocks(backbone)),
        d_model=d_model,
    ).to(device)

    gate_net = GatingNetwork(
        d_model=d_model,
        n_blocks=len(_shared_param_blocks(backbone)),
        hidden=tuple(config["gate"]["hidden"]),
    ).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────
    params = (
        list(backbone.parameters())
        + list(coord.parameters())
        + list(ctx_encoder.parameters())
        + list(grad_proj.parameters())
        + list(gate_net.parameters())
        + list(llm_proj.parameters())
    )
    opt = torch.optim.Adam(params, lr=config["lr"])

    # ── Loop ──────────────────────────────────────────────────────────
    step = 0
    for epoch in range(config["epochs"]):
        for batch in train_loader:
            x, p_u, labels = (t.to(device) for t in batch)
            B = x.shape[0]

            logits = backbone(x)
            losses = [F.binary_cross_entropy_with_logits(logits[i], labels[:, i])
                      for i in range(n_tasks)]

            # Per-task per-layer raw gradients on shared blocks.
            blocks = _shared_param_blocks(backbone)
            per_task_grads: list[list[torch.Tensor]] = []
            for i in range(n_tasks):
                grads_i = torch.autograd.grad(
                    losses[i], blocks, retain_graph=True, create_graph=False
                )
                per_task_grads.append([g.detach() for g in grads_i])

            consensus = [
                sum(per_task_grads[i][l] for i in range(n_tasks)) / n_tasks
                for l in range(len(blocks))
            ]

            # Build phi: (B, N, L, 2) via per-layer summaries.
            sumr = torch.stack([
                per_layer_summary(per_task_grads[i], consensus)
                for i in range(n_tasks)
            ])                                          # (N, L, 2)
            sumr = sumr.unsqueeze(0).expand(B, -1, -1, -1)
            phi = grad_proj(sumr)                       # (B, N, d)

            # Frozen LLM embeddings, projected, then user context.
            e = llm_proj(raw_e)                         # (N, d)
            psi_u = ctx_encoder(p_u)                    # (B, d)

            # Coordinator -> C(t, u).
            C = coord(phi, e, psi_u)                    # (B, N, N)

            # Aggregate task context c_i.
            c = aggregate_context(C, phi)               # (B, N, d)

            # Compute scalar gates per (task, block).
            modulated_loss = 0.0
            for i in range(n_tasks):
                for l in range(len(blocks)):
                    phi_il = sumr[:, i, l, :]           # (B, 2)
                    z = gate_net(c[:, i, :], e[i], phi_il)  # (B,)
                    # Sample-mean gate is used to scale the loss component
                    # whose gradient flows into block l for task i.
                    modulated_loss = modulated_loss + z.mean() * losses[i] \
                                     * (1.0 / len(blocks))

            opt.zero_grad()
            modulated_loss.backward()
            opt.step()
            step += 1

            if step % config.get("log_every", 100) == 0:
                logger.info(f"step={step} loss={modulated_loss.item():.4f}")

        # ── Validation ───────────────────────────────────────────────
        auc = evaluate_auc(backbone, val_loader, n_tasks, device)
        logger.info(f"epoch={epoch} val_auc={auc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    train(cfg)


if __name__ == "__main__":
    main()
