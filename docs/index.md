# genrec

基于 PyTorch 的生成式推荐系统研究框架。

## 特性

- 🚀 **现代架构**: 基于 Transformer 和变分自编码器的生成式推荐
- 📊 **丰富数据集**: 支持 P5 Amazon 等主流推荐数据集
- 🔧 **易于扩展**: 模块化设计，支持自定义数据集和模型
- 🎯 **端到端**: 从数据预处理到模型训练的完整流程
- 📈 **高性能**: 优化的训练流程和推理性能

## 快速开始

### 安装

```bash
git clone https://github.com/phonism/genrec.git
cd genrec
pip install -e .
```

### 简单示例

```python
from genrec.data import P5AmazonItemDataset
from genrec.models import RqVae

# 加载数据集
dataset = P5AmazonItemDataset(
    root="data/amazon",
    split="beauty"
)

# 训练 RQVAE 模型
model = RqVae(
    vocab_size=len(dataset),
    embedding_dim=256
)

# 开始训练...
```

## 架构概览

genrec 包含两个核心模型：

1. **RQVAE (Residual Quantized VAE)**: 学习物品的向量量化表示
2. **TIGER (Transformer-based Generative Retrieval)**: 基于用户历史序列生成推荐

## 核心组件

- **数据处理**: 支持多种推荐数据集格式
- **模型架构**: RQVAE + TIGER 双阶段训练
- **训练框架**: 基于 PyTorch Lightning 的现代化训练
- **配置管理**: 灵活的 Gin 配置系统

## 贡献

我们欢迎社区贡献！请查看[贡献指南](zh/contributing.md)了解详情。

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](https://github.com/phonism/genrec/blob/main/LICENSE) 文件了解详情。