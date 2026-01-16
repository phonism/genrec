# LCRec

通过集成协同语义适配大语言模型进行推荐。

## 概述

LCRec 通过码本令牌集成协同语义，将大语言模型（LLM）适配用于推荐任务。它使用 RQ-VAE 生成的语义 ID 表示物品，并微调 LLM 来生成物品推荐。

## 架构

```
物品文本 → RQ-VAE → 语义 ID [<C0_x>, <C1_y>, ..., <C4_z>]
                          ↓
用户历史 + 提示词 → LLM (Qwen) → 生成的语义 ID
                          ↓
                    约束解码
                          ↓
                    推荐物品
```

### 核心组件

- **语义 ID 生成**: 5 个码本的 RQ-VAE（每个 256 个码）
- **码本令牌**: 添加到 LLM 词表的特殊令牌 `<Ci_j>`
- **约束解码**: 带前缀约束的束搜索
- **多任务训练**: seqrec、item2index、index2item 任务

## 训练流程

### 第一步：训练 RQ-VAE 生成语义 ID

```bash
python genrec/trainers/rqvae_trainer.py config/lcrec/amazon/rqvae.gin --split beauty
```

### 第二步：微调 LLM

```bash
python genrec/trainers/lcrec_trainer.py config/lcrec/amazon/lcrec.gin --split beauty
```

### 调试模式（快速测试）

```bash
python genrec/trainers/lcrec_trainer.py config/lcrec/amazon/lcrec_debug.gin
```

## 配置

```gin
# config/lcrec/amazon/lcrec.gin

# 训练
train.epochs = 4
train.batch_size = 32
train.learning_rate = 2e-5
train.max_length = 512

# 模型
train.pretrained_path = %MODEL_HUB_QWEN3_1_7B
train.use_lora = False

# 码本
train.num_codebooks = 5
train.codebook_size = 256

# 评估
train.eval_beam_width = 10
```

## 任务

LCRec 支持三种训练任务：

| 任务 | 输入 | 输出 |
|------|------|------|
| **seqrec** | 用户历史 | 下一个物品语义 ID |
| **item2index** | 物品文本 | 物品语义 ID |
| **index2item** | 语义 ID | 物品文本 |

## 模型 API

```python
from genrec.models import LCRec

model = LCRec(pretrained_path="Qwen/Qwen2.5-1.5B")
model.add_codebook_tokens(num_codebooks=5, codebook_size=256)

# 生成推荐
outputs = model.generate(
    input_ids=prompt_ids,
    max_new_tokens=6,
    num_beams=10,
)
```

## 参考文献

- [LC-Rec: Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation](https://arxiv.org/abs/2311.09049)
