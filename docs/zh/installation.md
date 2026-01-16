# 安装指南

本指南提供了 genrec 框架的详细安装说明。

## 系统要求

### 硬件要求

**最低配置：**
- CPU: 4 核心
- RAM: 8 GB
- 存储: 20 GB 可用空间

**推荐配置：**
- CPU: 8+ 核心
- RAM: 16+ GB  
- GPU: NVIDIA GPU (8GB+ VRAM)
- 存储: 50+ GB SSD

### 软件要求

- Python 3.8 - 3.11
- CUDA 11.0+ (如果使用 GPU)
- Git

## 安装方法

### 方法一：从源码安装（推荐）

#### 1. 克隆仓库

```bash
git clone https://github.com/phonism/genrec.git
cd genrec
```

#### 2. 创建虚拟环境

**使用 conda:**
```bash
conda create -n genrec python=3.10
conda activate genrec
```

**使用 venv:**
```bash
python -m venv genrec_env
source genrec_env/bin/activate  # Linux/Mac
# 或
genrec_env\Scripts\activate     # Windows
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 方法二：开发环境安装

如果您计划修改代码或贡献代码：

```bash
git clone https://github.com/phonism/genrec.git
cd genrec

# 创建开发环境
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 开发依赖

# 安装预提交钩子
pre-commit install
```

## 依赖包说明

### 核心依赖

```yaml
# 深度学习框架
torch==2.6.0
torchvision==0.21.0
torch_geometric==2.6.1

# 分布式训练
accelerate==0.31.0

# 配置管理
gin_config==0.5.0

# 数据处理
pandas==1.5.3
polars==1.9.0
numpy==1.24.3

# 文本处理
sentence_transformers==3.3.1

# 实验跟踪
wandb==0.19.0

# 工具库
einops==0.8.0
tqdm==4.65.0
```

### 可选依赖

```bash
# 推荐系统专用库（可选）
pip install fbgemm_gpu==1.1.0
pip install torchrec==1.1.0

# 开发工具（可选）
pip install black isort flake8 pytest
```

## GPU 支持

### CUDA 安装检查

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
```

### 安装 CUDA 版本的 PyTorch

如果自动安装的不是 CUDA 版本：

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1  
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 验证安装

### 基本验证

```bash
python -c "
import torch
import pandas as pd
import sentence_transformers
print('✓ 基础依赖安装成功')
"
```

### 框架验证

```bash
python -c "
from genrec.data.p5_amazon import P5AmazonItemDataset
from genrec.models.rqvae import RqVae
print('✓ genrec 安装成功')
"
```

### GPU 验证

```bash
python -c "
import torch
print(f'GPU 数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 型号: {torch.cuda.get_device_name(0)}')
    print('✓ GPU 支持可用')
else:
    print('⚠ GPU 不可用，将使用 CPU')
"
```

## 常见问题

### Q: ImportError: No module named 'torch'

**解决方案:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# 或者安装 CUDA 版本（见上文）
```

### Q: CUDA out of memory

**解决方案:**
- 减小批量大小: `train.batch_size=16`
- 启用梯度累积: `train.gradient_accumulate_every=4`
- 使用混合精度: `train.mixed_precision_type="fp16"`

### Q: sentence-transformers 下载慢

**解决方案:**
```bash
# 设置环境变量使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或者预下载模型
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/sentence-t5-xl')
"
```

### Q: 数据集下载失败

**解决方案:**
```bash
# 手动设置代理
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port

# 或者手动下载数据集到 dataset/ 目录
```

### Q: Windows 下路径问题

**解决方案:**
```bash
# 使用正斜杠或原始字符串
train.dataset_folder="dataset/amazon"
# 或
train.dataset_folder=r"dataset\amazon"
```

## 性能优化

### 系统级优化

```bash
# Linux: 增加共享内存
echo 'vm.overcommit_memory=1' >> /etc/sysctl.conf

# 设置 PyTorch 线程数
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### 内存优化

```python
# 在训练前设置
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

## Docker 安装（可选）

### 使用预构建镜像

```bash
docker pull pytorch/pytorch:2.6.0-cuda12.1-cudnn9-devel

docker run -it --gpus all -v $(pwd):/workspace pytorch/pytorch:2.6.0-cuda12.1-cudnn9-devel
cd /workspace
pip install -r requirements.txt
```

### 自定义 Dockerfile

```dockerfile
FROM pytorch/pytorch:2.6.0-cuda12.1-cudnn9-devel

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "-c", "print('genrec ready!')"]
```

## 下一步

安装完成后，您可以：

1. 阅读[快速开始指南](getting-started.md)
2. 了解[数据集准备](dataset/overview.md)  
3. 开始[第一个训练实验](training/rqvae.md)
4. 查看[API 文档](api/index.md)

如果遇到其他问题，请查看我们的 [FAQ](faq.md) 或在 GitHub 上提交 Issue。