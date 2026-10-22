#!/bin/bash
# Usage: bash scripts/train_onerec_grpo.sh [SPLIT=beauty] [NUM_GPUS=1] [trainer arguments...]
# Published three-seed matrix: use scripts/run_matrix.py --preset amazon2014_20261022.
set -euo pipefail
SPLIT="${1:-beauty}"
NUM_GPUS="${2:-1}"
if [[ ! "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_GPUS must be a positive integer" >&2
    exit 2
fi
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -gt 0 ]; then shift; fi
GIN_CONFIG="config/onerec/amazon/grpo.gin"
if [ "${NUM_GPUS}" -gt 1 ]; then
    exec accelerate launch --multi_gpu --num_processes "${NUM_GPUS}" \
        -m genrec.trainers.onerec_grpo_trainer "${GIN_CONFIG}" --split "${SPLIT}" "$@"
else
    exec python -m genrec.trainers.onerec_grpo_trainer \
        "${GIN_CONFIG}" --split "${SPLIT}" "$@"
fi
