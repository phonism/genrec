# HSTU

层次序列转换单元。

## 概述

HSTU（Hierarchical Sequential Transduction Unit）是一个增强的序列推荐模型，通过三个关键创新改进了 SASRec：

1. **SiLU 注意力**: 用 SiLU 激活替代 softmax，保留偏好强度
2. **更新门**: 添加逐元素门控实现层次特征交互
3. **时间偏置**: 可选的对数桶时间位置偏置

## 架构

```
用户序列: [item_1, item_2, ..., item_n] + [timestamp_1, ..., timestamp_n]
       ↓
物品嵌入 + 位置编码
       ↓
HSTU 层 (SiLU 注意力 + 更新门 + 时间 RAB)
       ↓
RMS 归一化
       ↓
下一个物品预测
```

### 与 SASRec 的主要区别

| 组件 | SASRec | HSTU |
|------|--------|------|
| 注意力 | softmax(QK^T/√d) | SiLU(QK^T + RAB) |
| 输出 | Attention @ V | Norm(Attention @ V) ⊙ U |
| 时间 | 无 | 对数桶时间偏置 |
| 矩阵 | Q, K, V (3个) | Q, K, V, U (4个) |

### SiLU vs Softmax

- **Softmax**: 归一化注意力权重，丢失绝对偏好强度
- **SiLU**: 非归一化激活，保留偏好幅度（合成测试中差距达44.7%）

## 配置

```gin
# config/hstu/amazon.gin

train.epochs = 200
train.batch_size = 128
train.learning_rate = 1e-3
train.max_seq_len = 50

# 模型架构
train.hidden_dim = 64
train.num_heads = 2
train.num_layers = 2
train.dropout = 0.2
train.use_temporal_bias = True  # 启用时间 RAB
```

## 训练

```bash
# 使用时间偏置训练（默认）
python genrec/trainers/hstu_trainer.py config/hstu/amazon.gin

# 不使用时间偏置训练（类似 SASRec）
python genrec/trainers/hstu_trainer.py config/hstu/amazon.gin \
    --gin "train.use_temporal_bias=False"

# 在其他数据集上训练
python genrec/trainers/hstu_trainer.py config/hstu/amazon.gin --split sports
```

## 基准结果

### Amazon 2014 Beauty

| 模型 | R@5 | R@10 | N@5 | N@10 |
|------|-----|------|-----|------|
| SASRec | 0.0469 | 0.0688 | 0.0305 | 0.0375 |
| HSTU | 0.0486 | 0.0708 | 0.0340 | 0.0412 |

## 模型 API

```python
from genrec.models import HSTU

model = HSTU(
    num_items=10000,
    hidden_dim=64,
    num_heads=2,
    num_layers=2,
    max_seq_len=50,
    dropout=0.2,
    use_temporal_bias=True,
)

# 带时间戳的前向传播
logits = model(item_ids, timestamps, attention_mask)
```

## 参考文献

- [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152) (ICML 2024)
