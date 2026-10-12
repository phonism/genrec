# 训练 RQ-VAE

RQ-VAE 是 TIGER、LCRec、COBRA 的前置 tokenizer。训练时最重要的不是只看总 loss，而是确保 item encoder、checkpoint 几何、码本健康和下游配置全部匹配。

## 1. 选择下游配置

| 下游 | 配置 | 关键几何 |
|---|---|---|
| TIGER | `config/tiger/amazon/rqvae.gin` | 3 × 256，latent 32 |
| LCRec | `config/lcrec/amazon/rqvae.gin` | 5 × 256，latent 64 |
| COBRA | `config/cobra/amazon/rqvae.gin` | 3 × 32，latent 32 |

不能训练一个“通用 RQ-VAE”后把 checkpoint 传给所有模型。hidden dims、codebook size/layers 与 sentence encoder 都是 checkpoint 契约。

## 2. 准备数据与模型

```bash
python scripts/prepare_amazon2014.py \
  --splits beauty --kinds reviews meta

ls /data/models/sentence-t5-xl
```

首次运行会生成：

```text
dataset/amazon/processed/beauty/item_emb_sentence-t5-xl.parquet
```

若 `config/base.gin` 不是本机路径，从 CLI 覆盖模型常量。

## 3. 训练

```bash
python genrec/trainers/rqvae_trainer.py \
  config/tiger/amazon/rqvae.gin --split beauty \
  --gin 'MODEL_HUB_SENTENCE_T5_XL="/data/models/sentence-t5-xl"' \
  --gin 'train.wandb_logging=False'
```

当前正式配置使用 epoch 模式：5000 epochs、batch 1024、lr `1e-3`、50 epochs warmup、每 50 epochs eval/save。trainer 也支持 `iterations`，但 `epochs` 与 `iterations` 必须且只能设置一个。

## 4. 第一次 K-Means 初始化

在第一个训练 step，trainer 会收集最多约 20,000 条 item embedding，触发每层 codebook bootstrap。初始化只应发生一次：

- fresh run：`use_kmeans_init=True`；
- resume/inference：加载函数会把 `do_kmeans_init=False`、`kmeans_initted=True`；
- checkpoint 恢复测试见 `tests/test_rqvae_checkpoint.py`。

若恢复后第一次调用 SID 全变，优先检查是否绕过了 `load_pretrained` / `from_pretrained`。

## 5. 监控项

trainer 记录：

| 指标 | 含义 | 限制 |
|---|---|---|
| `total_loss` | reconstruction + RQ loss | 不直接代表下游推荐质量 |
| `reconstruction_loss` | item embedding 重构 | 受 embedding normalize 影响 |
| `rqvae_loss` | 各层 codebook + commitment | 需结合 usage 看 |
| `p_unique_ids` | 当前 batch 内 SID unique 调试值 | 不是全目录 collision |
| `collision_rate` | eval 时遍历训练 item 得到完整 SID collision | 聚合前会跨 rank gather 全部 SID |
| `emb_avg_norm_i` | 各层量化向量 norm | 用于发现层塌缩/尺度异常 |

训练温度当前代码固定为 `0.2`，不是随 epoch anneal；推 SID 常用 `0.001`，eval 模式下仍走最近邻硬分配。

只有所有层达到配置的最小 usage 才会写 `checkpoint_best_health.pt`；门槛前的最佳诊断快照写入 `checkpoint_best_candidate.pt`，不能供下游消费。

## 6. checkpoint

epoch 模式输出：

```text
out/<downstream>/amazon/<split>/rqvae/checkpoint_epoch_<zero-based-epoch>.pt
```

文件包含 model、`model_config`、optimizer、scheduler 和 epoch/iter。当前 TIGER、COBRA 下游 config 指向 `checkpoint_epoch_4999.pt`；LCRec 下游 config 指向 `checkpoint_epoch_1999.pt`，即使其 RQ-VAE config 本身仍配置 5000 epochs。checkpoint 路径是各下游的显式契约，运行前应以对应 `.gin` 为准，不要根据总 epoch 数猜测。

恢复训练：

```bash
python genrec/trainers/rqvae_trainer.py \
  config/tiger/amazon/rqvae.gin --split beauty \
  --gin 'train.pretrained_rqvae_path="/abs/path/checkpoint_epoch_2499.pt"'
```

恢复会同时加载 optimizer/scheduler。只做推理则使用：

```python
from genrec.models.rqvae import RqVae
model = RqVae.from_pretrained(path, map_location="cpu").eval()
```

## 7. 训练后健康检查

```bash
python scripts/verify_codebook_health.py \
  --kind rqvae \
  --checkpoint out/tiger/amazon/beauty/rqvae/checkpoint_epoch_4999.pt \
  --dataset-folder dataset/amazon \
  --split beauty \
  --encoder /data/models/sentence-t5-xl
```

同时建议导出完整 SID：

```bash
python scripts/dump_sem_ids.py --help
python scripts/report_sid_collisions.py --help
```

验收至少包括每层 usage/dead codes/max frequency、完整 SID collision rate、collision group 分布。高 usage 不等于完整 SID 唯一。

## 8. 下游前检查

- [ ] parquet encoder 与 RQ-VAE `input_dim` 匹配；
- [ ] 下游 dataset 的 `rqvae_hidden_dims/embed_dim/codebook_size/n_layers` 一致；
- [ ] checkpoint 加载后不会重新初始化；
- [ ] SID 行顺序与 item mapping 一致；
- [ ] collision 已记录；
- [ ] 生成模型主评测将 SID 解析到 item，而不是 tuple 命中。

## 常见故障

- **CUDA OOM 在初始化阶段发生**：第一个 step 会聚合大 batch；降低初始化样本或 batch，而不只是普通训练 batch。
- **parquet 维度不对**：删除/重建正确 encoder 的缓存前先保存 provenance；不要强行截断向量。
- **最后一层一个 code 吞掉大量 item**：codebook collapse；检查 K、item 数和浅残差尺度。小目录不要使用 8192 大码本。
- **loss 降但 collision 高**：重构目标不保证 SID 唯一；这正是需要独立 health 指标的原因。
- **多卡 collision 与离线脚本不同**：以全目录离线 health script 为准，trainer 内集合目前不跨 rank 合并。

模型与公式见 [RQ-VAE](../models/rqvae.md)。
