# GenRec

生成式推荐模型库。

## 概述

GenRec 是一个基于 PyTorch 的框架，提供生成式推荐模型的统一实现。框架采用干净的代码架构、基于 Gin-config 的实验管理，支持多种数据集和模型。

## 核心特性

- **7 种模型**: SASRec、HSTU、RQVAE、TIGER、LCRec、COBRA、NoteLLM
- **多数据集**: Amazon 2014（Beauty、Sports、Toys、Clothing）和 Amazon 2023（32 个类别）
- **Gin-Config**: 灵活的实验配置和参数覆盖
- **W&B 集成**: 基于 Weights & Biases 的实验跟踪
- **可复现**: 统一的 Recall@K 和 NDCG@K 评估指标

## 支持的模型

| 模型 | 类型 | 描述 |
|------|------|------|
| **SASRec** | 基线 | 自注意力序列推荐 |
| **HSTU** | 基线 | 层次化序列转换单元 |
| **RQVAE** | 生成式 | 残差量化 VAE，用于生成语义 ID |
| **TIGER** | 生成式 | 基于 Trie 约束解码的生成式检索 |
| **LCRec** | 生成式 | 基于 LLM 的协同语义推荐 |
| **COBRA** | 生成式 | 级联稀疏-稠密表示 |
| **NoteLLM** | 生成式 | 可检索的 LLM 笔记推荐（实验性） |

## 快速开始

```bash
# 安装
git clone https://github.com/phonism/genrec.git
cd genrec
pip install -e .

# 训练 SASRec（Amazon Beauty）
python genrec/trainers/sasrec_trainer.py config/sasrec/amazon.gin --split beauty

# 训练 TIGER（需要先训练 RQVAE）
python genrec/trainers/rqvae_trainer.py config/tiger/amazon/rqvae.gin --split beauty
python genrec/trainers/tiger_trainer.py config/tiger/amazon/tiger.gin --split beauty
```

## 项目结构

```
genrec/
├── genrec/
│   ├── models/          # SASRec, HSTU, RQVAE, TIGER, LCRec, COBRA, NoteLLM
│   ├── trainers/        # 每个模型的训练脚本
│   ├── modules/         # Transformer、嵌入层、指标、损失函数等
│   └── data/            # Amazon 2014 & 2023 数据集实现
├── config/              # Gin 配置文件
│   ├── sasrec/          # SASRec 配置
│   ├── hstu/            # HSTU 配置
│   ├── tiger/           # TIGER 配置（amazon/、amazon2023/）
│   ├── lcrec/           # LCRec 配置（amazon/、amazon2023/）
│   └── cobra/           # COBRA 配置
├── scripts/             # 工具脚本
└── docs/                # 文档（中英双语）
```

## 基准结果

完整基准表格请参见 [README](https://github.com/phonism/genrec)，涵盖 Amazon 2014 和 Amazon 2023 数据集。

## 贡献

欢迎提交 Issue 和 Pull Request！请参考[贡献指南](contributing.md)。

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](../LICENSE) 文件。

## 引用

```bibtex
@software{genrec2025,
  title = {GenRec: A Model Zoo for Generative Recommendation},
  author = {Qi Lu},
  year = {2025},
  url = {https://github.com/phonism/genrec}
}
```
