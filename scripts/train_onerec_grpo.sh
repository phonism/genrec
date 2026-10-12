#!/bin/bash
# OneRec GRPO Training (Stage 3)
# Usage: bash scripts/train_onerec_grpo.sh [SPLIT] [NUM_GPUS]
#   SPLIT: dataset split name (default: Arts_Crafts_and_Sewing)
#   NUM_GPUS: number of GPUs to use (default: 2)

set -e

SPLIT="${1:-Arts_Crafts_and_Sewing}"
NUM_GPUS="${2:-2}"

echo "=== OneRec GRPO Training ==="
echo "Split: ${SPLIT}"
echo "GPUs:  ${NUM_GPUS}"
echo ""

# Replace {split} placeholder in gin config
GIN_CONFIG="config/onerec/amazon2023/grpo.gin"

if [ "${NUM_GPUS}" -gt 1 ]; then
    accelerate launch \
        --config_file config/accelerate_ddp.yaml \
        --num_processes "${NUM_GPUS}" \
        -m genrec.trainers.onerec_grpo_trainer \
        "${GIN_CONFIG}" \
        --split "${SPLIT}"
else
    python -m genrec.trainers.onerec_grpo_trainer \
        "${GIN_CONFIG}" \
        --split "${SPLIT}"
fi
