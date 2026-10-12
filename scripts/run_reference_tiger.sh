#!/bin/bash
# Reference TIGER 完整训练流程
# Usage: bash scripts/run_reference_tiger.sh [dataset]
# Example: bash scripts/run_reference_tiger.sh Beauty

set -e

DATASET=${1:-Beauty}
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REF_DIR="${BASE_DIR}/reference/TIGER"

echo "=========================================="
echo "Reference TIGER Pipeline - ${DATASET}"
echo "=========================================="

# ==========================================
# Step 1: 数据预处理 (process.ipynb)
# ==========================================
echo ""
echo "[Step 1] Data Preprocessing"
echo "Please run: cd ${REF_DIR}/data && jupyter notebook process.ipynb"
echo "Or run the python script below..."

cd ${REF_DIR}/data

# 检查原始数据
if [ ! -f "reviews_${DATASET}_5.json.gz" ]; then
    echo ""
    echo "Download raw data first:"
    echo "  wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_${DATASET}_5.json.gz"
    echo "  wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_${DATASET}.json.gz"
    exit 1
fi

# 检查是否已处理
if [ -f "${DATASET}/item_emb.parquet" ]; then
    echo "✓ Data already processed, skipping..."
else
    echo "Running process.ipynb via papermill or manually..."
    # 如果有papermill可以自动运行notebook
    # papermill process.ipynb process_output.ipynb -p dataset_name ${DATASET}
    echo "Please run process.ipynb manually for dataset: ${DATASET}"
    exit 1
fi

# ==========================================
# Step 2: RQVAE 训练
# ==========================================
echo ""
echo "[Step 2] Training RQVAE..."

cd ${REF_DIR}/rqvae

python main.py \
    --data_path ../data/${DATASET}/item_emb.parquet \
    --ckpt_dir ./ckpt/${DATASET} \
    --num_emb_list 256 256 256 \
    --e_dim 32 \
    --layers 512 256 128 64 \
    --lr 1e-3 \
    --epochs 3000 \
    --batch_size 1024 \
    --warmup_epochs 50 \
    --eval_step 50 \
    --sk_epsilons 0.0 0.0 0.003 \
    --kmeans_init True \
    --device cuda:0

echo "✓ RQVAE training complete"

# ==========================================
# Step 3: 生成 Codes
# ==========================================
echo ""
echo "[Step 3] Generating codes..."
echo "Please edit generate_code.py to set correct checkpoint path, then run:"
echo "  cd ${REF_DIR}/rqvae && python generate_code.py"

# 找到最新的checkpoint
LATEST_CKPT_DIR=$(ls -td ./ckpt/${DATASET}/*/ 2>/dev/null | head -1)
if [ -n "${LATEST_CKPT_DIR}" ]; then
    echo "Latest checkpoint dir: ${LATEST_CKPT_DIR}"
    echo "Update generate_code.py line 47-48:"
    echo "  dataset = \"${DATASET}\""
    echo "  ckpt_path = \"${LATEST_CKPT_DIR}best_collision_model.pth\""
fi

# ==========================================
# Step 4: TIGER 训练
# ==========================================
echo ""
echo "[Step 4] Training TIGER..."

cd ${REF_DIR}/model

python main.py \
    --dataset_path ../data/${DATASET} \
    --code_path ../data/${DATASET}/${DATASET}_t5_rqvae.npy \
    --batch_size 256 \
    --infer_size 96 \
    --num_epochs 200 \
    --lr 1e-4 \
    --num_layers 4 \
    --num_decoder_layers 4 \
    --d_model 128 \
    --d_ff 1024 \
    --num_heads 6 \
    --vocab_size 1025 \
    --max_len 20 \
    --beam_size 30 \
    --early_stop 10 \
    --log_path ./logs/${DATASET}_tiger.log \
    --save_path ./ckpt/${DATASET}_tiger.pth \
    --device cuda:0

echo ""
echo "=========================================="
echo "✓ Done! Check logs at: ${REF_DIR}/model/logs/${DATASET}_tiger.log"
echo "=========================================="
