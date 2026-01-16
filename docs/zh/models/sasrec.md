# SASRec

自注意力序列推荐模型。

## 概述

SASRec（Self-Attentive Sequential Recommendation）是一个基线模型，使用自注意力机制从序列交互数据中捕获用户行为模式。

## 架构

```
用户序列: [item_1, item_2, ..., item_n]
       ↓
物品嵌入层
       ↓
位置编码
       ↓
Transformer 编码器 (自注意力)
       ↓
下一个物品预测
```

### 核心组件

- **物品嵌入**: 所有物品的可学习嵌入
- **位置编码**: 可学习的位置嵌入
- **自注意力层**: 带因果掩码的多头注意力
- **预测头**: 与物品嵌入的点积

## 配置

```gin
# config/sasrec/amazon.gin

train.epochs = 200
train.batch_size = 128
train.learning_rate = 1e-3
train.max_seq_len = 50

# 模型架构
train.hidden_dim = 64
train.num_heads = 2
train.num_layers = 2
train.dropout = 0.2
```

## 训练

```bash
# 在 Amazon Beauty 上训练
python genrec/trainers/sasrec_trainer.py config/sasrec/amazon.gin

# 在其他数据集上训练
python genrec/trainers/sasrec_trainer.py config/sasrec/amazon.gin --split sports
python genrec/trainers/sasrec_trainer.py config/sasrec/amazon.gin --split toys
```

## 评估指标

- **Recall@K**: 前K个推荐中相关物品的比例
- **NDCG@K**: 归一化折损累积增益

## 基准结果

### Amazon 2014 Beauty

| 模型 | R@5 | R@10 | N@5 | N@10 |
|------|-----|------|-----|------|
| SASRec | 0.0469 | 0.0688 | 0.0305 | 0.0375 |

## 模型 API

```python
from genrec.models import SASRec

model = SASRec(
    num_items=10000,
    hidden_dim=64,
    num_heads=2,
    num_layers=2,
    max_seq_len=50,
    dropout=0.2,
)

# 前向传播
logits = model(item_ids, attention_mask)
```

## 参考文献

- [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) (ICDM 2018)
