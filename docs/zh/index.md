# genrec

基于 PyTorch 的生成式推荐系统研究框架。

## 概述

genrec 是一个模块化的推荐系统研究框架，实现了多种最新的生成式推荐算法。该框架提供了干净的代码架构、灵活的配置系统以及易于扩展的数据处理管道。

## 核心特性

- ✨ **模块化设计**: 清晰的组件分离，易于理解和扩展
- 🔧 **配置驱动**: 基于 Gin 的灵活配置系统
- 📊 **多种模型**: RQVAE、TIGER 等最新生成式推荐模型
- 🎯 **数据集支持**: P5 Amazon 等主流推荐数据集
- 🚀 **分布式训练**: 基于 Accelerate 的多 GPU 训练支持
- 📈 **实验跟踪**: 集成 Weights & Biases 进行实验管理
- 🔍 **缓存优化**: 智能的数据预处理缓存机制

## 支持的模型

### RQVAE (Residual Quantized Variational Autoencoder)
- 基于向量量化的变分自编码器
- 支持多种量化策略：Gumbel-Softmax、STE、Rotation Trick、Sinkhorn
- 用于学习物品的语义表示

### TIGER (Recommender Systems with Generative Retrieval)
- 基于 Transformer 的生成式检索模型
- 使用语义 ID 进行序列建模
- 支持 Trie 约束的生成过程

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 训练 RQVAE

```bash
python genrec/trainers/rqvae_trainer.py config/rqvae/p5_amazon.gin
```

### 训练 TIGER

```bash
python genrec/trainers/tiger_trainer.py config/tiger/p5_amazon.gin
```

## 项目结构

```
genrec/
├── genrec/          # 核心代码
│   ├── data/                        # 数据处理模块
│   │   ├── configs.py               # 配置类定义
│   │   ├── base_dataset.py          # 数据集抽象基类
│   │   ├── p5_amazon.py             # P5 Amazon 数据集
│   │   ├── processors/              # 数据处理器
│   │   └── dataset_factory.py       # 数据集工厂
│   ├── models/                      # 模型实现
│   │   ├── rqvae.py                 # RQVAE 模型
│   │   └── tiger.py                 # TIGER 模型
│   ├── modules/                     # 基础模块
│   │   ├── embedding.py             # 嵌入层
│   │   ├── encoder.py               # 编码器
│   │   ├── loss.py                  # 损失函数
│   │   └── metrics.py               # 评估指标
│   └── trainers/                    # 训练脚本
│       ├── rqvae_trainer.py         # RQVAE 训练器
│       └── tiger_trainer.py         # TIGER 训练器
├── config/                          # 配置文件
│   ├── rqvae/                       # RQVAE 配置
│   └── tiger/                       # TIGER 配置
└── docs/                           # 文档
```

## 主要改进

相比原始实现，我们的重构版本提供了：

1. **更清晰的代码结构**: 模块化设计，职责分明
2. **配置化管理**: 支持灵活的参数配置和实验管理
3. **通用性增强**: 易于扩展到新的数据集和模型
4. **性能优化**: 缓存机制和内存效率提升
5. **更好的文档**: 完整的 API 文档和使用示例

## 基准结果

| 数据集 | 模型 | 指标 | 结果 |
|--------|------|------|------|
| P5 Amazon-Beauty | TIGER | Recall@10 | 0.42 |

## 贡献

欢迎提交 Issue 和 Pull Request！请参考我们的[贡献指南](contributing.md)。

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](../LICENSE) 文件。

## 引用

如果您在研究中使用了本框架，请引用相关论文：

```bibtex
@inproceedings{rqvae2023,
  title={RQ-VAE Recommender},
  author={Botta, Edoardo},
  year={2023}
}

@article{tiger2023,
  title={TIGER: Recommender Systems with Generative Retrieval},
  year={2023}
}
```