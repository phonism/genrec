#!/bin/bash
# Home: waits for SPORTS_DONE sentinel, then runs on GPU 3-6.
set -e
cd /root/workspace/code/genrec
source venv/bin/activate

SPORTS_DONE="${SPORTS_DONE:?usage: SPORTS_DONE=/path/to/sentinel run_v2_home.sh}"
ACC_CFG=config/accelerate_4gpu.yaml
SFT_CFG="${SFT_CFG:-config/onerec/amazon/sft_k256_v2.gin}"

echo "=== [home] using SFT config: $SFT_CFG ==="
echo "=== [home 0/3] waiting for sports sentinel ($SPORTS_DONE) ==="
while [ ! -f "$SPORTS_DONE" ]; do sleep 60; done

echo "=== [home 1/3] ensure Home processed + embeddings (GPU 3) ==="
CUDA_VISIBLE_DEVICES=3 python -c "
from genrec.data.amazon import AmazonItemDataset
ds = AmazonItemDataset(
    root='dataset/amazon', split='home', train_test_split='all',
    encoder_model_name='/root/workspace/models_hub/Qwen3-Embedding-8B',
)
print('Home embeddings shape:', ds.embeddings.shape)
"

echo "=== [home 2/3] RQ-KMeans K=256 ==="
[ -f out/onerec/amazon/home/rqkmeans_k256/rqkmeans.pt ] && echo "  already fit; skipping" \
    || python genrec/trainers/rqkmeans_trainer.py config/onerec/amazon/rqkmeans_k256.gin --split home

echo "=== [home 3/3] SFT 4-GPU on 3-6 (~17h) ==="
accelerate launch --config_file "$ACC_CFG" \
    -m genrec.trainers.onerec_sft_trainer "$SFT_CFG" --split home

echo "✅ Home done"
