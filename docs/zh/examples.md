# 可运行示例

项目以 trainer + Gin 为主入口。直接写 Python 训练循环容易漏掉 checkpoint、评测口径和 provenance，因此先给 CLI 例子，再给只用于理解 API 的小片段。

## 单模型 smoke test

```bash
python genrec/trainers/sasrec_trainer.py \
  config/sasrec/amazon.gin --split beauty \
  --gin 'train.epochs=1' \
  --gin 'train.patience=1' \
  --gin 'train.wandb_logging=False' \
  --gin 'train.save_dir_root="out/_smoke/sasrec/beauty"'
```

smoke test 只验证代码路径，不代表模型质量。正式结果不要复用 `_smoke` checkpoint。

## Gin 多参数覆盖

```bash
python genrec/trainers/hstu_trainer.py \
  config/hstu/amazon.gin --split sports \
  --gin 'train.seed=7' \
  --gin 'train.loss_type="sampled_softmax"' \
  --gin 'train.ss_sampling_strategy="local"' \
  --gin 'train.num_negatives=256' \
  --gin 'train.save_dir_root="out/hstu_ss/amazon/sports/s7"'
```

推荐从专用 `amazon_ss.gin` 开始，而不是在 CE config 上覆盖一长串参数；上例主要展示语法。

## 切换评测协议

```bash
python genrec/trainers/sasrec_trainer.py \
  config/sasrec/amazon.gin --split beauty \
  --gin 'AmazonSASRecDataset.evaluation_protocol="global_temporal"' \
  --gin 'train.seed=42' \
  --gin 'train.save_dir_root="out/sasrec/amazon/beauty/global_temporal/s42"'
```

协议必须进入输出路径，避免 leave-one-out 的 `results.json` 被覆盖。

## 量化器健康报告

```bash
python scripts/verify_codebook_health.py \
  --kind rqkmeans \
  --checkpoint out/onerec/amazon/beauty/rqkmeans/rqkmeans.pt \
  --dataset-folder dataset/amazon \
  --split beauty \
  --encoder /data/models/Qwen3-Embedding-8B
```

换 RQ-VAE 时使用 `--kind rqvae`。脚本报告逐层 usage/dead codes/max frequency 和完整 SID collision；不要只看其中一列。

## 结果矩阵

```bash
python scripts/run_matrix.py --help
python scripts/run_matrix.py \
  --models sasrec hstu \
  --splits beauty sports toys \
  --seeds 42 43 44 \
  --dry-run
```

以本机脚本 `--help` 为准；先 dry-run 检查 model→trainer/config 映射，再提交远程 cell。完成后：

```bash
python scripts/validate_matrix_results.py --help
python scripts/aggregate_results.py --help
```

validator 会拒绝 metric level/protocol/provenance 不完整的主结果。

## 直接调用 SASRec

```python
import torch
from genrec.models.sasrec import SASRec

model = SASRec(
    num_items=100,
    max_seq_len=5,
    embed_dim=16,
    num_heads=2,
    num_blocks=2,
    ffn_dim=64,
    dropout=0.0,
    loss_type="ce",
)

input_ids = torch.tensor([[0, 0, 4, 8, 3]])
targets = torch.tensor([[0, 0, 8, 3, 9]])
logits, loss = model(input_ids, targets)
top5 = model.predict(input_ids, top_k=5)

print(logits.shape)  # [1, 5, 101]
print(float(loss), top5)
```

真实评测应调用 trainer 的 full-dataloader 逻辑；这个片段没有数据协议和结果保存。

## 直接调用 HSTU

```python
import torch
from genrec.models.hstu import HSTU

model = HSTU(
    num_items=100,
    max_seq_len=5,
    embed_dim=16,
    num_heads=2,
    num_blocks=1,
    dropout=0.0,
    use_temporal_bias=True,
)

items = torch.tensor([[0, 2, 5, 7, 8]])
times = torch.tensor([[0, 100, 200, 260, 400]])
logits, _ = model(items, times)
print(logits.shape)  # [1, 5, 101]
```

sampled-softmax 的 score normalization 由 trainer 构造；不要仅在这个片段把 `sampled_softmax=True` 而忘记 `score_l2_norm/temperature`。

## 加载 RQ-VAE 并生成 SID

```python
import pandas as pd
import numpy as np
import torch

from genrec.models.rqvae import RqVae

df = pd.read_parquet(
    "dataset/amazon/processed/beauty/item_emb_sentence-t5-xl.parquet"
)
x = torch.tensor(np.stack(df["embedding"].values), dtype=torch.float32)

model = RqVae.from_pretrained(
    "out/tiger/amazon/beauty/rqvae/checkpoint_epoch_4999.pt",
    map_location="cpu",
).eval()

with torch.no_grad():
    sid = model.get_semantic_ids(x).sem_ids
print(sid.shape)  # [num_items, num_codebooks]
```

不要用不存在的 `load_from_checkpoint`，也不要脱离 mapping 单独保存 SID 数组。

## SID→item 的诚实解析

```python
import torch
from genrec.modules.metrics import SemanticIdItemMapper

catalog_sids = [[1, 2], [1, 2], [3, 4]]
training_sequences = [[0, 0, 1, 2], [1, 2, 2, 0]]
mapper = SemanticIdItemMapper.from_sequences(
    catalog_sids,
    training_sequences,
    train_holdout=2,
)

beam = torch.tensor([[[1, 2], [1, 2], [3, 4], [9, 9]]])
items = mapper.map_predictions(beam)
print(items)
```

同一 SID 两次会映射到同一代表 item，再被去重；不存在的 SID 被丢弃，缺位填 `-1`。

## 检查 `results.json`

```python
import json
from pathlib import Path

payload = json.loads(Path("out/sasrec/amazon/beauty/results.json").read_text())
for key in ["model", "split", "seed", "metrics", "config"]:
    print(key, payload.get(key))
```

实际 schema 还包含源码 provenance；不要用这个读取片段重写结果文件，生成应统一走 `save_run_results`。

## OneRec 三阶段

```bash
python genrec/trainers/rqkmeans_trainer.py \
  config/onerec/amazon/rqkmeans.gin --split beauty

accelerate launch --config_file config/accelerate_4gpu.yaml \
  -m genrec.trainers.onerec_sft_trainer \
  config/onerec/amazon/sft.gin --split beauty

accelerate launch --config_file config/accelerate_4gpu.yaml \
  -m genrec.trainers.onerec_grpo_trainer \
  config/onerec/amazon/grpo.gin --split beauty
```

GRPO 前先运行 `python scripts/check_grpo_readiness.py --help`，检查 SFT checkpoint、tokenizer 和 reward 稀疏度。

## 下一步

- [完整命令与依赖](getting-started.md)
- [配置系统](architecture/configuration.md)
- [评测系统](architecture/evaluation-system.md)
- [新增模型/数据集](architecture/extension-guide.md)
