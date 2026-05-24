"""
Frozen LLM encoder for task embeddings.

The paper uses LLaMA-2-7B as the default encoder; BERT-base and
RoBERTa-large are tested as ablation alternatives (Section 5.1).

The encoder is frozen — its parameters are never updated during
training. Task embeddings e_i are computed once before training and
cached.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


@torch.no_grad()
def encode_task_descriptions(
    descriptions: list[str],
    model_name: str = "meta-llama/Llama-2-7b-hf",
    pooling: str = "mean",
    device: str = "cuda",
) -> torch.Tensor:
    """Run a frozen pre-trained LM on each textual task description.

    Returns:
        Tensor of shape (N, H) where H is the encoder's hidden size.
        These embeddings are then projected to dimension d via
        TaskEmbeddingProjection below.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    out = []
    for desc in descriptions:
        toks = tokenizer(desc, return_tensors="pt", truncation=True,
                         max_length=128).to(device)
        h = model(**toks).last_hidden_state  # (1, T, H)
        if pooling == "mean":
            pooled = h.mean(dim=1).squeeze(0)
        elif pooling == "cls":
            pooled = h[:, 0, :].squeeze(0)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        out.append(pooled.cpu())
    return torch.stack(out)


class TaskEmbeddingProjection(nn.Module):
    """Learned linear projection from H-dim LM output to d-dim e_i.

    Note: only the projection is trained; the encoder is frozen.
    """

    def __init__(self, llm_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(llm_dim, d_model, bias=False)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.proj(raw)


def cache_or_load(
    descriptions: list[str],
    cache_path: str | Path,
    model_name: str = "meta-llama/Llama-2-7b-hf",
) -> torch.Tensor:
    """Encode descriptions once, then load from cache thereafter."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return torch.load(cache_path)
    emb = encode_task_descriptions(descriptions, model_name=model_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, cache_path)
    return emb
