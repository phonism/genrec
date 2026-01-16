# Installation Guide

This guide provides detailed installation instructions for the genrec framework.

## System Requirements

### Hardware Requirements

**Minimum Configuration:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 20 GB free space

**Recommended Configuration:**
- CPU: 8+ cores
- RAM: 16+ GB  
- GPU: NVIDIA GPU (8GB+ VRAM)
- Storage: 50+ GB SSD

### Software Requirements

- Python 3.8 - 3.11
- CUDA 11.0+ (if using GPU)
- Git

## Installation Methods

### Method 1: Install from Source (Recommended)

#### 1. Clone Repository

```bash
git clone https://github.com/phonism/genrec.git
cd genrec
```

#### 2. Create Virtual Environment

**Using conda:**
```bash
conda create -n genrec python=3.10
conda activate genrec
```

**Using venv:**
```bash
python -m venv genrec_env
source genrec_env/bin/activate  # Linux/Mac
# or
genrec_env\Scripts\activate     # Windows
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Method 2: Development Installation

If you plan to modify the code or contribute:

```bash
git clone https://github.com/phonism/genrec.git
cd genrec

# Create development environment
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Install pre-commit hooks
pre-commit install
```

## Dependencies Overview

### Core Dependencies

```yaml
# Deep learning frameworks
torch==2.6.0
torchvision==0.21.0
torch_geometric==2.6.1

# Distributed training
accelerate==0.31.0

# Configuration management
gin_config==0.5.0

# Data processing
pandas==1.5.3
polars==1.9.0
numpy==1.24.3

# Text processing
sentence_transformers==3.3.1

# Experiment tracking
wandb==0.19.0

# Utilities
einops==0.8.0
tqdm==4.65.0
```

### Optional Dependencies

```bash
# Recommendation-specific libraries (optional)
pip install fbgemm_gpu==1.1.0
pip install torchrec==1.1.0

# Development tools (optional)
pip install black isort flake8 pytest
```

## GPU Support

### Check CUDA Installation

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
```

### Install CUDA-enabled PyTorch

If the automatic installation doesn't include CUDA:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1  
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Verify Installation

### Basic Verification

```bash
python -c "
import torch
import pandas as pd
import sentence_transformers
print('✓ Basic dependencies installed successfully')
"
```

### Framework Verification

```bash
python -c "
from genrec.data.p5_amazon import P5AmazonItemDataset
from genrec.models.rqvae import RqVae
print('✓ genrec installed successfully')
"
```

### GPU Verification

```bash
python -c "
import torch
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU model: {torch.cuda.get_device_name(0)}')
    print('✓ GPU support available')
else:
    print('⚠ GPU not available, will use CPU')
"
```

## Common Issues

### Q: ImportError: No module named 'torch'

**Solution:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# or install CUDA version (see above)
```

### Q: CUDA out of memory

**Solution:**
- Reduce batch size: `train.batch_size=16`
- Enable gradient accumulation: `train.gradient_accumulate_every=4`
- Use mixed precision: `train.mixed_precision_type="fp16"`

### Q: sentence-transformers download slow

**Solution:**
```bash
# Set environment variable to use mirror
export HF_ENDPOINT=https://hf-mirror.com

# or pre-download model
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/sentence-t5-xl')
"
```

### Q: Dataset download fails

**Solution:**
```bash
# Manually set proxy
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port

# or manually download dataset to dataset/ directory
```

### Q: Windows path issues

**Solution:**
```bash
# Use forward slashes or raw strings
train.dataset_folder="dataset/amazon"
# or
train.dataset_folder=r"dataset\amazon"
```

## Performance Optimization

### System-level Optimization

```bash
# Linux: Increase shared memory
echo 'vm.overcommit_memory=1' >> /etc/sysctl.conf

# Set PyTorch thread count
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### Memory Optimization

```python
# Set before training
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

## Docker Installation (Optional)

### Using Pre-built Image

```bash
docker pull pytorch/pytorch:2.6.0-cuda12.1-cudnn9-devel

docker run -it --gpus all -v $(pwd):/workspace pytorch/pytorch:2.6.0-cuda12.1-cudnn9-devel
cd /workspace
pip install -r requirements.txt
```

### Custom Dockerfile

```dockerfile
FROM pytorch/pytorch:2.6.0-cuda12.1-cudnn9-devel

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "-c", "print('genrec ready!')"]
```

## Next Steps

After installation, you can:

1. Read the [Getting Started Guide](getting-started.md)
2. Learn about [Dataset Preparation](dataset/overview.md)  
3. Start your [First Training Experiment](training/rqvae.md)
4. Check [API Documentation](api/index.md)

If you encounter other issues, please check our [FAQ](faq.md) or submit an Issue on GitHub.