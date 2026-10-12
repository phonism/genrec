#!/bin/bash
# Sports SFT pipeline. Override SFT_CFG to use a different gin file.
set -e
cd /root/workspace/code/genrec
source venv/bin/activate

DL_EXIT=/tmp/thanos-tasks/task-22.exit
ACC_CFG=config/accelerate_4gpu.yaml
SFT_CFG="${SFT_CFG:-config/onerec/amazon/sft_k256_v2.gin}"

echo "=== [sports] using SFT config: $SFT_CFG ==="

echo "=== [sports 1/5] waiting for Qwen3-Embedding-8B (task 22) ==="
while [ ! -f "$DL_EXIT" ]; do sleep 30; done

PQ=dataset/amazon/processed/sports/item_emb_Qwen3-Embedding-8B.parquet
if [ -f "$PQ" ]; then
    ./venv/bin/python -c "import pandas as pd; pd.read_parquet('$PQ')" 2>/dev/null \
        || { echo "  removing corrupt $PQ"; rm -f "$PQ"; }
fi

echo "=== [sports 2/5] ensure Sports item embeddings (GPU 3) ==="
CUDA_VISIBLE_DEVICES=3 python -c "
from genrec.data.amazon import AmazonItemDataset
ds = AmazonItemDataset(
    root='dataset/amazon', split='sports', train_test_split='all',
    encoder_model_name='/root/workspace/models_hub/Qwen3-Embedding-8B',
)
print('Sports embeddings shape:', ds.embeddings.shape)
"

echo "=== [sports 3/5] RQ-KMeans K=256 ==="
[ -f out/onerec/amazon/sports/rqkmeans_k256/rqkmeans.pt ] && echo "  already fit; skipping" \
    || python genrec/trainers/rqkmeans_trainer.py config/onerec/amazon/rqkmeans_k256.gin --split sports

echo "=== [sports 4/5] SFT 4-GPU on 3-6 (~17h) ==="
accelerate launch --config_file "$ACC_CFG" \
    -m genrec.trainers.onerec_sft_trainer "$SFT_CFG" --split sports

echo "✅ Sports done"
