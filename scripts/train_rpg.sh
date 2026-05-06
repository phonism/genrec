#!/bin/bash
# RPG Training - Recommendation with Parallel Generation (Meta, KDD 2025)
# Usage: bash scripts/train_rpg.sh [SPLIT] [NUM_GPUS]
#   SPLIT: dataset split name (default: beauty)
#   NUM_GPUS: number of GPUs to use (default: 1)

set -e

SPLIT="${1:-beauty}"
NUM_GPUS="${2:-1}"

echo "=== RPG Training ==="
echo "Split: ${SPLIT}"
echo "GPUs:  ${NUM_GPUS}"
echo ""

GIN_CONFIG="config/rpg/amazon.gin"

if [ "${NUM_GPUS}" -gt 1 ]; then
    accelerate launch \
        --multi_gpu \
        --num_processes "${NUM_GPUS}" \
        -m genrec.trainers.rpg_trainer \
        "${GIN_CONFIG}" \
        --split "${SPLIT}"
else
    python -m genrec.trainers.rpg_trainer \
        "${GIN_CONFIG}" \
        --split "${SPLIT}"
fi
