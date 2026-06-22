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
train.epochs = 20
train.batch_size = 32
train.learning_rate = 1e-4

# 模型架构
train.n_codebooks = 3
train.id_vocab_size = 32      # 32x32x32 码本，与论文一致
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

## 复现状态

**已实现，但尚未复现成功。** Amazon 2014 Beauty（leave-one-out，完整评测集，`checkpoint_final.pt` = epoch 19）：

| 指标 | R@5 | R@10 | N@5 | N@10 |
|------|-----|------|-----|------|
| honest item-level（test，n_beam=32） | 0.0054 | **0.0099** | 0.0033 | 0.0048 |
| honest item-level（valid） | 0.0060 | 0.0101 | 0.0036 | 0.0049 |
| sem-id 口径（test，膨胀参考） | 0.0071 | 0.0127 | 0.0045 | 0.0063 |
| **论文** | 0.0537 | **0.0725** | 0.0395 | 0.0456 |

比论文低约 7 倍。修复了三个 bug/问题，再定位剩余差距：

1. **评测口径碰撞膨胀（已修）。** 原本比目标的**语义 ID 元组**而非 item id；32×32×32 码本碰撞率 0.43，召回桶里任意 item 都算命中（约 1.3× 虚高，且让 BeamFusion 的 dense 重排隐形）。现改为比 **item id**，sem-id 仅作参考。`beam_fusion` 已向量化、`generate()` encoder 已分块（修 OOM）。

2. **encoder / RQ-VAE 错配（已修——这个会让任何全新训练悄悄崩）。** trainer 把**模型的** encoder 名字（`sentence-t5-base`）传给数据集，覆盖了"数据集该用 `sentence-t5-xl`（rqvae 训练所用）算 item 向量"的 gin 绑定。结果 item 向量落在错误空间 → 语义 ID 从 **6901 塌成 529**。修法：不覆盖数据集 encoder（base/xl 共用 T5 tokenizer，模型看到的文本不变）。

3. **解码 beam 太窄（已修）。** `n_beam` 20→32（码本上限）：item R@10 +13%。

**剩余差距 = 模型天花板，不是量化器也不是解码。** RQ-VAE 码本健康（perplexity≈28-32/32），稀疏分支在 **teacher forcing 下其实学会了**（逐码本 0.26/0.21/0.43；eval 的逐码本统计的是**召回 item** 的码、不是模型生成的码）。瓶颈是模型对留出 item 的内在能力偏低，需更深/更久训练的 decoder 才能抬起来——4 层 decoder 的 pilot 让 teacher-forced 上限稳步上升（epoch9 train recall 已超 2 层基线 epoch19），但 item-level 要等上限够高才动。逼近论文是开放式训练工程（论文未公开超参）。`generate()` 仍缺 KV-cache，评测慢。

工具：`scripts/eval_cobra_itemlevel.py`（item/sem-id 双口径）、`scripts/diag_cobra_{loss,gen,final}.py`（teacher-forced vs 生成诊断）。

## 参考文献

- COBRA: Sparse Meets Dense - Unified Generative Recommendations with Cascaded Sparse-Dense Representations
