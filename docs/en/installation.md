# Installation Guide

## System Requirements

### Hardware

**Minimum:**

- CPU: 4 cores
- RAM: 8 GB
- Storage: 20 GB free space

**Recommended:**

- CPU: 8+ cores
- RAM: 16+ GB
- GPU: NVIDIA GPU (8GB+ VRAM)
- Storage: 50+ GB SSD

### Software

- Python 3.9 - 3.12
- CUDA 11.0+ (for GPU training)
- Git

## Installation Methods

### Method 1: Install from Source (Recommended)

```bash
git clone https://github.com/phonism/genrec.git
cd genrec
pip install -e .
```

### Method 2: Full Installation

Includes optional dependencies like Triton, TorchRec, and PEFT:

```bash
pip install -e ".[full]"
```

### Method 3: Development Installation

```bash
git clone https://github.com/phonism/genrec.git
cd genrec
pip install -e ".[dev]"
pre-commit install
```

### Method 4: Dependencies Only

```bash
pip install -r requirements.txt
```

## Core Dependencies

- `torch>=2.0.0` - Deep learning framework
- `accelerate>=0.31.0` - Distributed training
- `gin-config>=0.5.0` - Configuration management
- `sentence-transformers>=3.0.0` - Text encoding (for RQVAE)
- `transformers>=4.40.0` - Hugging Face Transformers (for LCRec, NoteLLM)
- `wandb>=0.19.0` - Experiment tracking
- `pandas>=1.5.0`, `polars>=1.9.0` - Data processing
- `einops>=0.8.0` - Tensor operations

### Optional Dependencies

```bash
# For COBRA and advanced models
pip install triton torchvision torch-geometric peft
```

## GPU Support

### Check CUDA

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

### Install CUDA-enabled PyTorch

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Verify Installation

```bash
python -c "
from genrec.models import SASRec, HSTU, RqVae, Tiger, LCRec, Cobra
print('GenRec installed successfully')
"
```

## Common Issues

### ImportError: No module named 'torch'

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### CUDA out of memory

Reduce batch size via `--gin "train.batch_size=16"`.

### sentence-transformers download slow

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Docker (Optional)

```dockerfile
FROM pytorch/pytorch:2.6.0-cuda12.1-cudnn9-devel
WORKDIR /app
COPY . .
RUN pip install -e .
```

## Next Steps

1. [Getting Started Guide](getting-started.md)
2. [Dataset Preparation](dataset/overview.md)
3. [API Documentation](api/index.md)
