#!/bin/bash
set -e
cd /root/workspace/code/genrec
source venv/bin/activate

DL_EXIT=/tmp/thanos-tasks/task-22.exit
ACC_CFG=config/accelerate_toys.yaml
SFT_CFG="${SFT_CFG:-config/onerec/amazon/sft_k256_v2.gin}"
PORT="${MAIN_PORT:-29501}"

echo "=== [toys] using SFT config: $SFT_CFG (port $PORT) ==="

while [ ! -f "$DL_EXIT" ]; do sleep 30; done

echo "=== [toys 1/3] ensure Toys item embeddings (GPU 7) ==="
CUDA_VISIBLE_DEVICES=7 python -c "
from genrec.data.amazon import AmazonItemDataset
ds = AmazonItemDataset(
    root='dataset/amazon', split='toys', train_test_split='all',
    encoder_model_name='/root/workspace/models_hub/Qwen3-Embedding-8B',
)
print('Toys embeddings shape:', ds.embeddings.shape)
"

echo "=== [toys 2/3] RQ-KMeans K=256 ==="
[ -f out/onerec/amazon/toys/rqkmeans_k256/rqkmeans.pt ] && echo "  already fit; skipping" \
    || python genrec/trainers/rqkmeans_trainer.py config/onerec/amazon/rqkmeans_k256.gin --split toys

echo "=== [toys 3/3] SFT 4-GPU on 0,1,2,7 (~17h) ==="
accelerate launch --config_file "$ACC_CFG" --main_process_port "$PORT" \
    -m genrec.trainers.onerec_sft_trainer "$SFT_CFG" --split toys

echo "✅ Toys done"
