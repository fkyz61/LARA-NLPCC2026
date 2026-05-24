#!/usr/bin/env bash
# Reproduce the UCI Census-Income row of Table 1(b).
set -euo pipefail

python scripts/prepare_uci.py --out_dir ./data/uci
for seed in 42 43 44 45 46; do
    python -m src.lara_trainer --config configs/lara_uci.yaml --seed "$seed"
done
