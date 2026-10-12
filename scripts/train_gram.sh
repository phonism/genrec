#!/bin/bash
# Train GRAM model on Amazon dataset
#
# Usage:
#   ./scripts/train_gram.sh Beauty         # Single GPU, Beauty dataset
#   ./scripts/train_gram.sh Beauty 2       # 2 GPUs, Beauty dataset
#   ./scripts/train_gram.sh Toys 4         # 4 GPUs, Toys dataset
#   ./scripts/train_gram.sh Sports 2 "--gin train_gram.epochs=50"  # Override config

SPLIT="${1:-Beauty}"
NUM_GPUS="${2:-2}"
EXTRA_ARGS="${3:-}"

CONFIG="config/gram/amazon/gram.gin"

echo "=== Training GRAM ==="
echo "Split: ${SPLIT}"
echo "GPUs:  ${NUM_GPUS}"
echo "Config: ${CONFIG}"
echo ""

if [ "${NUM_GPUS}" -eq 1 ]; then
    python -m genrec.trainers.gram_trainer \
        "${CONFIG}" \
        --split "${SPLIT}" \
        ${EXTRA_ARGS}
else
    accelerate launch \
        --num_processes "${NUM_GPUS}" \
        --mixed_precision bf16 \
        -m genrec.trainers.gram_trainer \
        "${CONFIG}" \
        --split "${SPLIT}" \
        ${EXTRA_ARGS}
fi
