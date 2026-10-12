#!/bin/bash
# End-to-end K=256 pipeline:
#   0) wait for Qwen3-1.7B download (task 16) — base model was wiped, re-downloading
#   1) wait for RQ-KMeans task (13) — already done, this is a no-op
#   2) verify codebook health
#   3) train SFT (4 GPU, sft_k256_v2.gin)
#   4) verify SFT checkpoint
#   5) train GRPO (4 GPU, grpo_k256_v2.gin)
set -e
cd /root/workspace/code/genrec
source venv/bin/activate

DL_EXIT=/tmp/thanos-tasks/task-16.exit
RQK_TASK_EXIT=/tmp/thanos-tasks/task-13.exit
SFT_CFG=config/onerec/amazon/sft_k256_v2.gin
GRPO_CFG=config/onerec/amazon/grpo_k256_v2.gin
ACC_CFG=config/accelerate_4gpu.yaml
SFT_BEST=out/onerec/amazon/beauty/sft_k256_v2_1.7b_seq50/checkpoint_best
QWEN_DIR=/root/workspace/models_hub/Qwen3-1.7B

echo "=== [stage 0/5] waiting for Qwen3-1.7B download (task 16) ==="
while [ ! -f "$DL_EXIT" ]; do sleep 30; done
DL_E=$(cat "$DL_EXIT")
echo "  download finished, exit=$DL_E"
if [ "$DL_E" != "0" ]; then
    echo "❌ Qwen3-1.7B download failed, aborting"; exit 10
fi
if [ ! -f "$QWEN_DIR/config.json" ] || [ ! -f "$QWEN_DIR/tokenizer.json" ]; then
    echo "❌ Qwen3-1.7B looks incomplete at $QWEN_DIR"; ls -la "$QWEN_DIR" || true
    exit 11
fi
echo "  Qwen3-1.7B ready at $QWEN_DIR"

echo ""
echo "=== [stage 1/5] waiting for RQ-KMeans (task 13) ==="
while [ ! -f "$RQK_TASK_EXIT" ]; do sleep 30; done
RQK_EXIT=$(cat "$RQK_TASK_EXIT")
echo "  RQ-KMeans finished, exit=$RQK_EXIT"
if [ "$RQK_EXIT" != "0" ]; then
    echo "❌ RQ-KMeans failed, aborting pipeline"; exit 1
fi

echo ""
echo "=== [stage 2/5] verifying codebook health ==="
python scripts/verify_rqkmeans_k256.py
VERIFY1=$?
if [ "$VERIFY1" != "0" ]; then
    echo "❌ codebook unhealthy (exit $VERIFY1), aborting pipeline"
    echo "   inspect: out/onerec/amazon/beauty/rqkmeans_k256/rqkmeans.pt"
    exit 2
fi

echo ""
echo "=== [stage 3/5] training SFT (4 GPU, ~8h expected) ==="
accelerate launch --config_file "$ACC_CFG" \
    -m genrec.trainers.onerec_sft_trainer "$SFT_CFG" --split beauty
SFT_EXIT=$?
if [ "$SFT_EXIT" != "0" ]; then
    echo "❌ SFT failed (exit $SFT_EXIT), aborting pipeline"; exit 3
fi

echo ""
echo "=== [stage 4/5] verifying SFT best checkpoint ==="
if [ ! -d "$SFT_BEST" ]; then
    echo "❌ SFT best checkpoint missing at $SFT_BEST"; exit 4
fi
ls -la "$SFT_BEST" | head -5

echo ""
echo "=== [stage 5/5] training GRPO (4 GPU, ~3h expected) ==="
accelerate launch --config_file "$ACC_CFG" \
    -m genrec.trainers.onerec_grpo_trainer "$GRPO_CFG" --split beauty
GRPO_EXIT=$?
if [ "$GRPO_EXIT" != "0" ]; then
    echo "❌ GRPO failed (exit $GRPO_EXIT)"; exit 5
fi

echo ""
echo "=== ✅ Pipeline complete ==="
