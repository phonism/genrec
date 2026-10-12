#!/usr/bin/env bash
# 远程 GPU 服务器一次性环境初始化脚本
# 在远程服务器上执行：bash /root/workspace/code/genrec/scripts/setup-remote.sh
set -euo pipefail

GENREC_DIR="/root/workspace/code/genrec"
VENV_DIR="$GENREC_DIR/venv"

echo "=== GenRec 远程环境初始化 ==="

# 1. 创建 Python venv
if [ ! -d "$VENV_DIR" ]; then
  echo "▶ 创建 Python 虚拟环境..."
  python3 -m venv "$VENV_DIR"
  echo "✓ venv 创建完成"
else
  echo "✓ venv 已存在，跳过"
fi

# 2. 激活 venv 并安装依赖
echo "▶ 安装依赖..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$GENREC_DIR/requirements.txt"
echo "✓ 依赖安装完成"

# 3. 配置 WandB（如果设置了 WANDB_API_KEY）
if [ -n "${WANDB_API_KEY:-}" ]; then
  echo "▶ 配置 WandB..."
  wandb login "$WANDB_API_KEY"
  echo "✓ WandB 配置完成"
else
  echo "⚠ 未设置 WANDB_API_KEY，跳过 WandB 配置"
fi

# 4. 创建必要目录
echo "▶ 创建输出目录..."
mkdir -p "$GENREC_DIR/out"
mkdir -p "$GENREC_DIR/dataset"
mkdir -p "$GENREC_DIR/models_hub"
echo "✓ 目录创建完成"

# 5. 验证 PyTorch + CUDA
echo "▶ 验证 PyTorch + CUDA..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
        print(f'    Memory: {mem:.0f} GB')
else:
    print('  ⚠ CUDA 不可用！请检查驱动和 PyTorch 版本')
"

echo ""
echo "=== 初始化完成 ==="
echo "提示："
echo "  - 数据集需手动放到 $GENREC_DIR/dataset/"
echo "  - 预训练模型需手动放到 $GENREC_DIR/models_hub/"
echo "  - 激活环境：source $VENV_DIR/bin/activate"
