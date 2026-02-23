# 安装指南

## 系统要求

### 硬件

**最低配置：**

- CPU: 4 核心
- RAM: 8 GB
- 存储: 20 GB 可用空间

**推荐配置：**

- CPU: 8+ 核心
- RAM: 16+ GB
- GPU: NVIDIA GPU (8GB+ VRAM)
- 存储: 50+ GB SSD

### 软件

- Python 3.9 - 3.12
- CUDA 11.0+（GPU 训练）
- Git

## 安装方法

### 方法一：从源码安装（推荐）

```bash
git clone https://github.com/phonism/genrec.git
cd genrec
pip install -e .
```

### 方法二：完整安装

包含 Triton、TorchRec、PEFT 等可选依赖：

```bash
pip install -e ".[full]"
```

### 方法三：开发环境安装

```bash
git clone https://github.com/phonism/genrec.git
cd genrec
pip install -e ".[dev]"
pre-commit install
```

### 方法四：仅安装依赖

```bash
pip install -r requirements.txt
```

## 核心依赖

- `torch>=2.0.0` - 深度学习框架
- `accelerate>=0.31.0` - 分布式训练
- `gin-config>=0.5.0` - 配置管理
- `sentence-transformers>=3.0.0` - 文本编码（RQVAE 使用）
- `transformers>=4.40.0` - Hugging Face Transformers（LCRec、NoteLLM 使用）
- `wandb>=0.19.0` - 实验跟踪
- `pandas>=1.5.0`、`polars>=1.9.0` - 数据处理
- `einops>=0.8.0` - 张量操作

### 可选依赖

```bash
# COBRA 等高级模型需要
pip install triton torchvision torch-geometric peft
```

## GPU 支持

### 检查 CUDA

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

### 安装 CUDA 版本 PyTorch

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 验证安装

```bash
python -c "
from genrec.models import SASRec, HSTU, RqVae, Tiger, LCRec, Cobra
print('GenRec 安装成功')
"
```

## 常见问题

### ImportError: No module named 'torch'

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### CUDA out of memory

通过 `--gin "train.batch_size=16"` 减小批量大小。

### sentence-transformers 下载慢

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Docker（可选）

```dockerfile
FROM pytorch/pytorch:2.6.0-cuda12.1-cudnn9-devel
WORKDIR /app
COPY . .
RUN pip install -e .
```

## 下一步

1. [快速开始指南](getting-started.md)
2. [数据集准备](dataset/overview.md)
3. [API 文档](api/index.md)
