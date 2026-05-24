#!/usr/bin/env bash
# Launcher template for the proprietary industrial experiments.
# The data path must be set in your environment; see configs/industrial_schema.md
# for the expected schema.
set -euo pipefail

if [ -z "${LARA_INDUSTRIAL_DATA_DIR:-}" ]; then
    echo "Set LARA_INDUSTRIAL_DATA_DIR to a parquet directory matching" \
         "configs/industrial_schema.md before running."
    exit 1
fi

# Override the data path in the config via environment expansion.
python -m src.lara_trainer --config configs/lara_industrial.yaml --seed 42
