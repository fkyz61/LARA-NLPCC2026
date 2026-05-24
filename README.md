# LARA: LLM-Conditioned Gradient Modulation for Multi-Task Recommendation

> Anonymous code repository for the NLPCC 2026 submission
> *"LARA: LLM-Conditioned Gradient Modulation for Multi-Task Recommendation"* (Paper ID: XXX).

This repository contains a reference PyTorch implementation of **LARA**, together
with the data-processing scripts and configuration files needed to reproduce
the experiments in the paper.

---

## 1. Overview

LARA recasts multi-task coordination in recommender systems as a
**semantically-conditioned, user-conditioned gradient-modulation** problem.
A lightweight Coordinator Network reads three signals at every training step:

1. Frozen LLM-derived task embeddings `e_i` (computed once from a textual
   description of each task).
2. A compact two-scalar-per-layer summary `phi_i` of each task's gradient.
3. A user-context vector `psi_u` derived from raw user features.

The Coordinator emits a per-step, per-user Collaboration Weight Matrix
`C(t, u)` that drives a per-task gated fusion of gradients before each
parameter update.

---

## 2. Repository Layout

```
.
├── README.md                   ← this file
├── requirements.txt            ← Python dependencies
├── LICENSE                     ← MIT (anonymized)
├── configs/
│   ├── lara_industrial.yaml    ← hyper-parameters for the industrial dataset
│   ├── lara_uci.yaml           ← hyper-parameters for UCI Census-Income
│   └── industrial_schema.md    ← schema description of the proprietary dataset
├── src/
│   ├── __init__.py
│   ├── coordinator.py          ← Coordinator Network + Collaboration Weight Matrix
│   ├── gated_fusion.py         ← Personalized Gated Fusion module
│   ├── gradient_summary.py     ← Per-layer cosine + log-norm gradient summary
│   ├── llm_encoder.py          ← Frozen LLM task-embedding utilities
│   ├── mmoe_backbone.py        ← Shared MMoE backbone used by all baselines
│   ├── lara_trainer.py         ← End-to-end training loop with LARA
│   └── utils.py                ← Logging / metrics / config loading
└── scripts/
    ├── prepare_uci.py          ← Download and preprocess UCI Census-Income (KDD)
    ├── run_uci.sh              ← Reproduce the UCI Census-Income experiment
    └── run_industrial.sh       ← Template for the proprietary industrial dataset
```

---

## 3. Environment

```bash
python --version   # 3.10 recommended
pip install -r requirements.txt
```

The implementation is tested with PyTorch 2.0 + CUDA 11.7 on a single
A100-80GB GPU. The industrial-scale experiments in the paper used
8x A100-80GB; an exact reproduction of the production model is not
possible from this repository because the industrial dataset is
proprietary (see Section 5 below).

---

## 4. Reproducing the UCI Census-Income Results

UCI Census-Income (KDD), dataset id 117 on the UCI ML Repository, is the
only fully-public benchmark used in the paper. The following commands
reproduce **Table 1(b)** of the manuscript.

```bash
# Step 1: download and preprocess UCI Census-Income (~10 MB).
# The script uses the official `ucimlrepo` package; no manual download needed.
python scripts/prepare_uci.py --out_dir ./data/uci

# Step 2: train LARA with the released hyper-parameters
python -m src.lara_trainer --config configs/lara_uci.yaml --seed 42
```

Expected output (mean over 5 seeds, ~30 minutes on a single GPU):

| Method | Task1 AUC | Task2 AUC |
|--------|-----------|-----------|
| MMoE   | 0.9387    | 0.9927    |
| DRGrad | 0.9550    | 0.9949    |
| **LARA (Ours)** | **0.9563** | **0.9952** |

> *Baselines (SNR, PLE, PCGrad, CAGrad, AdaTask) share the same
> `MMoEBackbone` and can be reproduced by substituting the corresponding
> gradient-coordination rule for the LARA Coordinator + Gating modules
> in `src/lara_trainer.py`. The relevant references are listed in the
> paper's bibliography; their public implementations are widely available.*

Hyper-parameter grids and ablation switches are exposed in
`configs/lara_uci.yaml`. Set `coordinator.use_llm: false` to obtain the
*w/o LLM Semantic* row of Table 3; set `coordinator.use_user_context: false`
to obtain *w/o user conditioning*; and so on.

---

## 5. Industrial Dataset (Limitation)

The 15B-sample industrial dataset used for Table 1(a), Table 2, Table 3,
and Figure 4 originates from a live short-form video application
operated by the authors' affiliated institution (anonymized for review).
Due to commercial sensitivity and user-privacy constraints, the raw data
**cannot** be released. We instead provide:

- `scripts/run_industrial.sh` — the exact launcher used for the
  production experiments. It expects a parquet directory with the same
  schema as the public UCI loader.
- `configs/lara_industrial.yaml` — the production hyper-parameters
  (number of experts, learning rate, coordinator MLP shape, etc.).
- A *schema description* in `configs/industrial_schema.md` listing the
  expected feature columns and label columns, so that any practitioner
  with a comparable internal dataset can reproduce the model exactly.

Plot-generation scripts for Figures 1-4 are omitted from this release
since they depend on industrial-data artifacts; the algorithmic
implementation is fully covered by the modules under `src/`.

The UCI Census-Income reproduction in Section 4 above is therefore the
primary public reproducibility artifact and is sufficient to verify
**all** algorithmic claims and ablation contrasts; the industrial
numbers serve as a scale-and-realism validation.

---

## 6. License

This code is released under the MIT License (see `LICENSE`). The license
file is anonymized for review and will be updated with the author and
institution information in the camera-ready version.

---

## 7. Contact

For the duration of the double-blind review, all correspondence should
be routed through the conference submission system. Author contact
information will be made available in the camera-ready version.
