# COBRA

稀疏与稠密相遇：级联稀疏-稠密表示的统一生成式推荐。

## 概述

COBRA 是一个混合生成式推荐模型，结合了稀疏语义 ID 和稠密向量表示。它使用交错的稀疏-稠密架构来捕获离散和连续的物品特征。

## 架构

```
物品表示 = [sparse_id_0, sparse_id_1, ..., sparse_id_C, dense_vector]

输入序列:
[s0_1, s1_1, s2_1, d_1, s0_2, s1_2, s2_2, d_2, ..., s0_n, s1_n, s2_n, d_n]
     └── 物品 1 ──┘        └── 物品 2 ──┘              └── 物品 n ──┘

       ↓
Transformer 解码器
       ↓
稀疏头（码本预测）+ 稠密头（向量重建）
```

### 核心组件

- **稀疏表示**: 来自 RQ-VAE 的 C 个码本令牌
- **稠密表示**: 连续嵌入向量
- **交错输入**: 交替的稀疏和稠密令牌
- **双预测头**: 分别用于稀疏和稠密输出的头

## 训练流程

### 第一步：训练 RQ-VAE

```bash
python genrec/trainers/rqvae_trainer.py config/cobra/amazon/rqvae.gin
```

### 第二步：训练 COBRA

```bash
python genrec/trainers/cobra_trainer.py config/cobra/amazon/cobra.gin
```

## 配置

```gin
# config/cobra/amazon/cobra.gin

# 训练
train.epochs = 100
train.batch_size = 32
train.learning_rate = 1e-4

# 模型架构
train.n_codebooks = 3
train.id_vocab_size = 256
train.d_model = 768
train.decoder_n_layers = 2
train.decoder_num_heads = 6

# 损失权重
train.sparse_loss_weight = 1.0
train.dense_loss_weight = 1.0

# 编码器
train.encoder_type = "pretrained"
train.encoder_model_name = %MODEL_HUB_SENTENCE_T5_BASE
```

## 损失函数

COBRA 使用组合损失：

```
L = λ_sparse * L_sparse + λ_dense * L_dense
```

- **L_sparse**: 码本预测的交叉熵损失
- **L_dense**: 向量重建的余弦相似度损失

## 模型 API

```python
from genrec.models import Cobra

model = Cobra(
    n_codebooks=3,
    id_vocab_size=256,
    d_model=768,
    decoder_n_layers=2,
    decoder_num_heads=6,
    encoder_type="pretrained",
)

# 训练前向
output = model(input_ids, encoder_input_ids)
loss = output.loss_sparse + output.loss_dense

# 生成
generated = model.generate(
    input_ids=history_ids,
    encoder_input_ids=history_texts,
    n_candidates=10,
)
```

## 与 TIGER 的对比

| 方面 | TIGER | COBRA |
|------|-------|-------|
| 表示 | 仅稀疏 | 稀疏 + 稠密 |
| 物品编码 | 语义 ID | 语义 ID + 嵌入 |
| 输入格式 | 扁平序列 | 交错序列 |
| 预测 | 码本令牌 | 码本令牌 + 向量 |

## 参考文献

- COBRA: Sparse Meets Dense - Unified Generative Recommendations with Cascaded Sparse-Dense Representations
