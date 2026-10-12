# Trainer API

当前仓库没有 `Trainer`、`BaseTrainer`、`RQVAETrainer` 或统一 callback 生命周期。每个文件暴露一个独立的 `@gin.configurable` 顶层函数，由脚本末尾的 `parse_config(); train*()` 启动。本页是这些真实入口、共享工具、collator/evaluator 与产物格式的索引。

## 导入方式

入口函数大量重名，应显式从模块导入并起别名：

```python
from genrec.trainers.sasrec_trainer import train as train_sasrec
from genrec.trainers.tiger_trainer import train as train_tiger
from genrec.trainers.rqvae_trainer import train as train_rqvae
from genrec.trainers.onerec_sft_trainer import train_sft
```

[`genrec.trainers.__init__`](https://github.com/phonism/genrec/blob/main/genrec/trainers/__init__.py) 只真正导入共享 utility。它的 `__all__` 虽列出 `rqvae_trainer` 等模块名，却没有把这些模块绑定为包属性；不要依赖 `from genrec.trainers import *` 获取 trainer。

所有训练入口的有效返回值都是 `None`。评测结果通过日志、W&B 和 `results.json` 交付，不从函数返回。

## 入口总表

| 流程 | 函数 / 源文件 | 现有配置 | 运行时 |
|---|---|---|---|
| SASRec | `train` / [`sasrec_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/sasrec_trainer.py) | [`config/sasrec/amazon.gin`](https://github.com/phonism/genrec/blob/main/config/sasrec/amazon.gin) | Accelerate |
| HSTU | `train` / [`hstu_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/hstu_trainer.py) | [`config/hstu/`](https://github.com/phonism/genrec/tree/main/config/hstu) | Accelerate |
| RQ-VAE | `train` / [`rqvae_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/rqvae_trainer.py) | 下游专用 `rqvae.gin` | Accelerate |
| RQ-KMeans | `fit_rqkmeans` / [`rqkmeans_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/rqkmeans_trainer.py) | [`rqkmeans.gin`](https://github.com/phonism/genrec/blob/main/config/onerec/amazon/rqkmeans.gin) | CPU / scikit-learn |
| TIGER | `train` / [`tiger_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/tiger_trainer.py) | [`tiger.gin`](https://github.com/phonism/genrec/blob/main/config/tiger/amazon/tiger.gin) | 单进程 PyTorch |
| LCRec | `train` / [`lcrec_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/lcrec_trainer.py) | [`config/lcrec/`](https://github.com/phonism/genrec/tree/main/config/lcrec) | Accelerate |
| COBRA | `train` / [`cobra_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/cobra_trainer.py) | [`config/cobra/`](https://github.com/phonism/genrec/tree/main/config/cobra) | Accelerate |
| OneRec SFT | `train_sft` / [`onerec_sft_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/onerec_sft_trainer.py) | [`sft.gin`](https://github.com/phonism/genrec/blob/main/config/onerec/amazon/sft.gin) | Accelerate/FSDP-aware save |
| OneRec GRPO | `train_grpo` / [`onerec_grpo_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/onerec_grpo_trainer.py) | [`grpo.gin`](https://github.com/phonism/genrec/blob/main/config/onerec/amazon/grpo.gin) | Accelerate |
| OneRec CLM | `train_clm` / [`onerec_clm_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/onerec_clm_trainer.py) | [`clm.gin`](https://github.com/phonism/genrec/blob/main/config/onerec/amazon/clm.gin) | Accelerate |
| RPG | `train_rpg` / [`rpg_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/rpg_trainer.py) | [`config/rpg/`](https://github.com/phonism/genrec/tree/main/config/rpg) | Accelerate |
| NoteLLM | `train` / [`notellm_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/notellm_trainer.py) | [`notellm/amazon.gin`](https://github.com/phonism/genrec/blob/main/config/notellm/amazon.gin) | Accelerate |
| GRAM | `train_gram` / [`gram_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/gram_trainer.py) | [`gram/amazon/gram.gin`](https://github.com/phonism/genrec/blob/main/config/gram/amazon/gram.gin) | Accelerate |

## 判别式 baseline 签名 { #discriminative-baseline-signatures }

### SASRec

```python
train(
    epochs=200, batch_size=128, learning_rate=1e-3, weight_decay=0.0,
    max_seq_len=50, embed_dim=64, num_heads=2, num_blocks=2,
    ffn_dim=256, dropout=0.2, loss_type="bce",
    dataset_folder="dataset/amazon", split="beauty",
    do_eval=True, eval_every_epoch=1, eval_batch_size=256,
    patience=50,
    save_dir_root="out/sasrec/amazon/beauty", save_every_epoch=50,
    wandb_logging=False, wandb_project="sasrec_training",
    wandb_log_interval=100,
    amp=True, mixed_precision_type="bf16", seed=42,
)
```

dataset 类在函数内固定为 `AmazonSASRecDataset`，无法用 Gin 注入其他 adapter。`loss_type="bce"` 时 collator 采一个逐位置负例；`SASRec` 把任何其他字符串都走 full cross-entropy，没有枚举校验。验证按 item-level `Recall@10` early stop。

### HSTU

```python
train(
    epochs=200, batch_size=128, learning_rate=1e-3, weight_decay=0.0,
    max_seq_len=50, embed_dim=64, num_heads=2, num_blocks=2,
    dropout=0.2, dqk=None, dv=None,
    use_position_bias=True, concat_ua=False,
    num_position_buckets=32, num_time_buckets=64,
    use_temporal_bias=True,
    loss_type="ce", num_negatives=128,
    ss_temperature=0.05, ss_l2_norm=True,
    ss_sampling_strategy="local",
    dataset_folder="dataset/amazon", split="beauty",
    dataset_type="amazon",
    do_eval=True, eval_every_epoch=10, eval_batch_size=256,
    patience=50,
    save_dir_root="out/hstu/amazon/beauty", save_every_epoch=50,
    wandb_logging=False, wandb_project="hstu_training",
    wandb_log_interval=100,
    amp=True, mixed_precision_type="bf16", seed=42,
)
```

`dataset_type` 只接受 `"amazon"` 或 `"ml1m"`，`loss_type` 只接受 `"ce"` 或
`"sampled_softmax"`，否则立即 `ValueError`。sampled-softmax 的 `local` 是 Meta 的
local-table sampler：每个正例独立、有放回地从完整目录采样；`in_batch` 是单独选项；
`global` 保留 GenRec 历史上整 batch 共用且去重的目录负例。未知采样器不会再静默回退。

!!! note "Amazon 2023 走兼容 raw schema"
    `sasrec_trainer.train()` 固定构造 `AmazonSASRecDataset`，HSTU 的 `dataset_type="amazon"` 固定构造 `AmazonHSTUDataset`；它们没有选择 [`Amazon2023SASRecDataset`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon2023.py)。这是当前的刻意路由：`prepare_amazon2023()` 会把 2023 数据写成 Amazon 2014 兼容的 gzip 字段/目录，`DATASET_CONFIGS` 已列出 `books` 与 `arts_crafts_and_sewing`。因此必须先运行预处理，并且只有登记了文件名的 split 能由这两个 trainer 读取。

## Quantizer 签名

### RQ-VAE

完整签名、两种训练模式与 checkpoint 契约见 [RQ-VAE API](rqvae.md#trainer-api)。关键约束是 `epochs` 与 `iterations` 必须二选一。

### RQ-KMeans

```python
fit_rqkmeans(
    dataset=AmazonItemDataset,
    dataset_folder="dataset/amazon",
    encoder_model_name="./models_hub/Qwen3-Embedding-8B",
    n_layers=3, codebook_size=8192,
    save_dir_root="out/onerec/amazon/{split}/rqkmeans",
    random_state=42, max_iter=100, batch_size=4096, seed=42,
)
```

dataset 调用为 `dataset(root=..., train_test_split="all", encoder_model_name=...)`，并必须暴露 float NumPy `embeddings[N,D]`。函数拟合全部 item、打印碰撞统计，写 `rqkmeans.pt` 与 `results.json`。当前仓库 Gin 把默认 `codebook_size=8192` 覆盖为 256。

## SID 生成推荐签名

### TIGER

完整签名、`t5_collate()`、beam 输出与 item-level 解析见 [TIGER API](tiger.md#trainer)。TIGER 是唯一不走 `setup_accelerator()` 的主要 trainer。

### LCRec

```python
train(
    epochs=4, batch_size=8, learning_rate=5e-5,
    weight_decay=0.01, warmup_ratio=0.01,
    gradient_accumulate_every=2, max_length=512,
    pretrained_path="Qwen/Qwen2.5-1.5B", use_lora=True,
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    num_codebooks=5, codebook_size=256,
    dataset=None, dataset_folder="dataset/amazon",
    max_seq_len=20, max_text_len=128,
    pretrained_rqvae_path="./out/lcrec/amazon/beauty/rqvae/checkpoint.pt",
    do_eval=True, eval_every_epoch=1,
    eval_batch_size=64, eval_beam_width=10,
    save_dir_root="out/lcrec/amazon/beauty", save_every_epoch=1,
    wandb_logging=False, wandb_project="lcrec_training",
    wandb_run_name=None, wandb_log_interval=10,
    split_batches=True, amp=True, mixed_precision_type="bf16",
    max_train_samples=0, max_eval_samples=0, debug_logging=False,
    eval_only=False, checkpoint_path=None, seed=42,
)
```

`dataset=None` 的默认值不可直接运行，必须由 Gin 注入 callable。dataset 需接受 `train_test_split`、RQ-VAE 路径等参数，并提供 `sem_ids_list`、`sequences`、`evaluation_protocol`。`checkpoint_path` 是 `LCRec.save_pretrained()` 目录，不是 optimizer checkpoint。`eval_only=True` 只评 valid 后返回。

### COBRA

```python
train(
    epochs=100, batch_size=32, learning_rate=1e-4, weight_decay=0.01,
    dataset_folder="dataset/amazon",
    save_dir_root="out/cobra/amazon/beauty", dataset=None,
    split_batches=True, amp=False,
    wandb_logging=False, wandb_project="cobra_training",
    wandb_run_name=None, wandb_log_interval=10,
    mixed_precision_type="fp16", gradient_accumulate_every=1,
    save_every_epoch=10, eval_valid_every_epoch=5,
    eval_test_every_epoch=10, do_eval=True,
    encoder_n_layers=1, encoder_hidden_dim=768,
    encoder_num_heads=8, encoder_vocab_size=32128,
    id_vocab_size=256, n_codebooks=3, d_model=384,
    max_len=1024, temperature=0.2, queue_size=1024,
    decoder_n_layers=8, decoder_num_heads=6, decoder_dropout=0.1,
    encoder_type="light", num_warmup_steps=500, max_seq_len=20,
    pretrained_rqvae_path="./out/rqvae/amazon/beauty/checkpoint.pt",
    encoder_model_name="sentence-transformers/sentence-t5-xl",
    resume_from_checkpoint=None,
    sparse_loss_weight=1.0, dense_loss_weight=1.0,
    use_dense=True, seed=42,
)
```

`dataset` 同样必须由 Gin 注入。resume 文件必须是 COBRA 自己的 dict checkpoint，含 `model`、`optimizer`、`scheduler`、`epoch`。`encoder_type="pretrained"` 选择 `SentenceT5Encoder`，任何其他字符串都会静默走 `LightT5Encoder`，没有枚举校验。dataset 的 embedding encoder 和文本字段必须与 RQ-VAE 训练时一致。trainer 只用 valid 选择 `best_model.pt`，所有 rank 重载后运行一次 test；`eval_test_every_epoch` 仅为旧配置兼容保留。

## OneRec 签名

### SFT

```python
train_sft(
    epochs=4, batch_size=8, learning_rate=3e-4, weight_decay=0.01,
    warmup_steps=20, warmup_ratio=0.0, adam_beta2=0.999,
    gradient_accumulate_every=2, max_length=512,
    pretrained_path="Qwen/Qwen2.5-1.5B", use_lora=True,
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    num_codebooks=3, codebook_size=8192,
    dataset=None, dataset_folder="dataset/amazon2023",
    max_seq_len=20, max_text_len=128,
    do_eval=True, eval_every_epoch=1,
    eval_batch_size=64, eval_beam_width=10,
    save_dir_root="out/onerec/amazon2023/{split}/sft",
    save_every_epoch=1,
    wandb_logging=False, wandb_project="onerec_sft",
    wandb_run_name=None, wandb_log_interval=10,
    split_batches=True, amp=True, mixed_precision_type="bf16",
    max_train_samples=0, max_eval_samples=0, debug_logging=False,
    eval_only=False, checkpoint_path=None,
    early_stopping_patience=3, attention_dropout=0.0,
    label_smoothing=0.0, enabled_tasks=None, seed=42,
)
```

当前 `config/onerec/amazon/sft.gin` 将默认 K=8192/旧路径覆盖为 Amazon 2014、K=256 和 RQ-KMeans。`enabled_tasks` 函数参数当前只记录在 config；实际 task 筛选由 dataset 的 Gin binding 完成。`label_smoothing` 在 trainer 自定义 token loss 路径使用。

### GRPO

```python
train_grpo(
    epochs=2, batch_size=4, learning_rate=1e-5, weight_decay=0.01,
    warmup_steps=10, gradient_accumulate_every=4,
    max_length=512, max_grad_norm=0.3,
    num_generations=16, temperature=1.0,
    beta=1e-3, clip_eps=0.2,
    reward_type="exact", use_beam_search=False, add_gt=False,
    num_codebooks=3, codebook_size=8192,
    sft_checkpoint_path="./out/onerec/amazon2023/{split}/sft/checkpoint_final",
    use_lora=True, lora_r=16, lora_alpha=32, lora_dropout=0.05,
    rl_dataset=None, dataset_folder="dataset/amazon2023", max_seq_len=20,
    eval_dataset=None,
    do_eval=True, eval_every_epoch=1,
    eval_batch_size=4, eval_beam_width=10,
    save_dir_root="out/onerec/amazon2023/{split}/grpo",
    save_every_epoch=1,
    wandb_logging=False, wandb_project="onerec_grpo",
    wandb_run_name=None, wandb_log_interval=10,
    split_batches=True, amp=True, mixed_precision_type="bf16",
    max_train_samples=0, max_eval_samples=0,
    debug_logging=False, seed=42,
)
```

`rl_dataset` 是必需 callable；`eval_dataset=None` 时即使 `do_eval=True` 也不建立验证/测试 loader。策略从 SFT `save_pretrained` 目录加载。采样使用 catalog `PrefixTrie`；tokenizer/codebook 不一致的 item 会被 warning 后排除。`reward_type` 的分派在训练循环中只支持实际实现的 `exact|hierarchical|rank`，未知值抛 `ValueError`。

### CLM

```python
train_clm(
    epochs=5, batch_size=32, learning_rate=5e-5, weight_decay=0.01,
    warmup_steps=50, gradient_accumulate_every=2, max_length=256,
    pretrained_path="Qwen/Qwen2.5-1.5B", use_lora=False,
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    num_codebooks=3, codebook_size=256,
    dataset=None, dataset_folder="dataset/amazon", max_seq_len=50,
    do_eval=True, eval_every_epoch=1,
    eval_batch_size=64, eval_beam_width=10,
    save_dir_root="out/onerec/amazon/{split}/clm", save_every_epoch=1,
    wandb_logging=False, wandb_project="onerec_clm",
    wandb_run_name=None, wandb_log_interval=10,
    split_batches=True, amp=True, mixed_precision_type="bf16",
    max_train_samples=0, max_eval_samples=0, debug_logging=False,
    eval_only=False, checkpoint_path=None,
    early_stopping_patience=3, seed=42,
)
```

`dataset=None` 时显式回退 `AmazonOneRecCLMDataset`，这是与 SFT/GRPO 的差异。模型会添加 codebook token 和 `<|item_sep|>`；checkpoint 必须保留 tokenizer。训练样本是一位用户的平铺 code token 流，而不是 SFT prompt-response。

## 其他生成模型签名

### RPG

```python
train_rpg(
    n_embd=448, n_layer=2, n_head=4, n_inner=1024,
    n_codebook=32, codebook_size=256, temperature=0.07,
    embd_pdrop=0.5, attn_pdrop=0.5, resid_pdrop=0.0,
    num_beams=50, n_edges=50, propagation_steps=3, chunk_size=1024,
    epochs=150, batch_size=256, learning_rate=3e-4,
    weight_decay=0.01, warmup_steps=10000, max_grad_norm=1.0,
    max_seq_len=50, dataset_folder="dataset/amazon", split="beauty",
    encoder_model_name="sentence-transformers/sentence-t5-xl",
    sent_emb_pca=0,
    do_eval=True, eval_every_epoch=1, eval_batch_size=256,
    eval_with_graph=False, patience=20,
    save_dir_root="out/rpg/amazon/{split}", save_every_epoch=50,
    wandb_logging=False, wandb_project="rpg_training",
    wandb_run_name=None, wandb_log_interval=100,
    amp=True, mixed_precision_type="bf16", seed=42,
)
```

dataset 类固定为 `AmazonRPGDataset`。验证选择指标是 item-level `NDCG@10`；最终 headline 是 direct decoding item metric，`eval_with_graph=True` 时 graph metric 放在 `results.json.extra`。

### NoteLLM

```python
train(
    epochs=5, batch_size=4, gradient_accumulate_every=8,
    learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.05,
    pretrained_path="Qwen/Qwen2.5-1.5B",
    freeze_lm=False, use_lora=True,
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    dataset_folder="dataset/amazon", split="beauty",
    max_seq_len=50, max_text_length=256,
    eval_batch_size=32, catalog_batch_size=64, patience=2,
    enable_csft=True, csft_alpha=0.01,
    save_dir_root="out/notellm/amazon/{split}",
    wandb_logging=False, wandb_project="notellm_training",
    wandb_run_name=None,
    amp=True, mixed_precision_type="bf16", seed=42,
)
```

`epochs`、train/eval batch size 和 text length 必须为正，否则 `ValueError`。当 `use_lora=True` 且没有 PEFT 时明确抛 `RuntimeError`，不会静默全参训练。dataset 固定为 `AmazonNoteLLMDataset`；训练默认组合对称 GCL 与 CSFT，并按精确 catalog item ID 屏蔽重复物品形成的伪负例。评测每次编码完整 catalog，排除 query item 后做 exact full-catalog item retrieval。best 目录同时保存可直接复用的 `adapter/` 与包含训练状态的内部 checkpoint。

当前 [`config/notellm/amazon.gin`](https://github.com/phonism/genrec/blob/main/config/notellm/amazon.gin) 同时绑定 `train.split` 与 `AmazonNoteLLMDataset.split`；标准 `--split sports` 会一致替换两处及输出目录，不再需要额外的 Gin override。

### GRAM

```python
train_gram(
    backbone="t5-small", max_his=20, use_position_embedding=True,
    epochs=30, batch_size=32, learning_rate=1e-3,
    weight_decay=0.01, warmup_ratio=0.05,
    gradient_accumulate_every=2, max_grad_norm=1.0,
    dataset=None, dataset_folder="dataset/gram", split="Beauty",
    item_prompt_max_len=128, target_max_len=50,
    item_id_file="item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt",
    item_prompt="all_text", top_k_similar_item=10,
    cf_model="sasrec", his_sep=" ; ",
    skip_empty_his=True, reverse_history=True,
    num_beams=50, eval_batch_size=1, length_penalty=0.0,
    do_eval=True, eval_every_epoch=1,
    metrics="hit@5,hit@10,ndcg@5,ndcg@10",
    save_dir_root="out/gram/{split}", save_every_epoch=5,
    wandb_logging=False, wandb_project="gram",
    wandb_run_name=None, wandb_log_interval=10,
    split_batches=True, amp=True, mixed_precision_type="bf16",
    max_train_samples=0, max_eval_samples=0,
    eval_only=False, checkpoint_path=None, seed=42,
)
```

dataset 为 `None` 时回退 `AmazonGRAMDataset`。`metrics` 以逗号分隔，每项必须含 `@K`，否则解析时失败。candidate trie 来自 `train_ds.all_items`。`checkpoint_path` 是 raw state dict 文件；`eval_only=True` 加载后分别评 valid/test 并返回。

## Collator 与 evaluator 索引

这些 helper 没有统一 protocol，返回 schema 必须按源文件使用：

| API | 输入 / 输出摘要 |
|---|---|
| `sasrec_trainer.evaluate(model, dataloader, accelerator, top_ks=[1,5,10])` | dataloader dict → item Recall/NDCG dict；跨 rank reduce。 |
| `hstu_trainer.evaluate(..., use_temporal_bias=True, top_ks=[1,5,10])` | 同上，额外消费 timestamps。 |
| `rpg_trainer.evaluate(..., top_ks=[1,5,10], use_graph=False)` | item-level full catalog/graph decoding指标。 |
| `notellm_trainer.evaluate_full_catalog(..., topks=(1,5,10))` | exact catalog embedding retrieval；返回大小写规范化前的 metric dict。 |
| `cobra_collate_fn(batch, pad_id=0, n_codebooks=3, is_train=True)` | 返回 `input_ids`、`encoder_input_ids`、`target_sem_ids`、`target_item_ids`；空 batch 会访问 `batch[0]` 而失败。 |
| `lcrec_collate_fn(batch, tokenizer, max_length=512, num_codebooks=5, is_eval=False)` | train 时含 labels，prompt token label 置 `-100`；eval 时左 padding且无 labels。 |
| `ConstrainedDecodingHelper(num_codebooks, codebook_size, tokenizer)` | 位置级 codebook token mask，不约束 SID tuple 必须在 catalog 中。 |
| `onerec_collate_fn(...)` / `evaluate_onerec(...)` | 把 OneRec task 名映射到 LCRec task；默认跳过慢的 index2item。 |
| `PrefixTrie(sem_ids_list, num_codebooks, codebook_size, tokenizer)` | GRPO catalog tuple 级约束，`get_prefix_allowed_tokens_fn()` 可传 HF generate。 |
| `rl_collate_fn(batch, tokenizer, max_length=512, num_codebooks=3)` | 左 padding prompt；返回 target SID/item/task。 |
| `CLMTokenMapper` / `clm_collate_fn` / `clm_eval_collate_fn` | item 序列 ↔ code token stream；train 右 padding，eval 左 padding。 |
| `GRAMCollator` | 位于数据模块，见 [数据集 API](datasets.md#rpg-and-gram)。 |
| `t5_collate` | 见 [TIGER API](tiger.md#t5_collate)。 |

### GRPO reward

```python
compute_rule_reward(generated_sem_ids, target_sem_ids) -> list[float]
compute_hierarchical_reward(
    generated_sem_ids, target_sem_ids, num_codebooks,
) -> list[float]
compute_rank_reward(
    generated_sem_ids, target_sem_ids, num_codebooks,
) -> list[float]
```

- `rule`：完整 tuple 相等为 1，否则 0。
- `hierarchical`：从第一个 code 起的连续 prefix 匹配长度 `m` 得 `10**(m-1)`，无匹配为 0。
- `rank`：先算 exact；整个 group 无 exact 时返回全 0，有 exact 时才给错误候选 rank penalty。`num_codebooks` 参数在当前 `compute_rank_reward()` 内未使用。

## 共享 utility

源文件：[`trainer_utils.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/trainer_utils.py)

### `TrainerConfig`

```python
@dataclass
class TrainerConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulate_every: int = 1
    num_warmup_steps: int = 500
    dataset_folder: str = "dataset/amazon"
    save_dir_root: str = "out/"
    split_batches: bool = True
    amp: bool = False
    mixed_precision_type: str = "fp16"
    wandb_logging: bool = False
    wandb_project: str = "training"
    wandb_run_name: str | None = None
    wandb_log_interval: int = 10
    save_every_epoch: int = 10
    eval_valid_every_epoch: int = 5
    eval_test_every_epoch: int = 10
    do_eval: bool = True
    resume_from_checkpoint: str | None = None
```

它只是未被各 trainer 继承/消费的便利 dataclass；创建它不会配置 Gin 或启动训练。

### 随机性与执行环境

```python
set_seed(seed: int, deterministic: bool = False) -> None
get_git_sha() -> str
setup_accelerator(
    split_batches=True,
    gradient_accumulate_every=1,
    amp=False,
    mixed_precision_type="fp16",
) -> accelerate.Accelerator
setup_wandb(project, run_name=None, config=None, step_metrics=None) -> None
get_parameter_count(model) -> int
log_training_info(logger, model, train_dataloader,
                  valid_dataloader=None, test_dataloader=None,
                  device=None) -> None
```

`set_seed()` 设置 Python、NumPy、CPU/CUDA RNG 与 `PYTHONHASHSEED`；运行中才设置环境变量不能追溯改变当前解释器已产生的 hash。`deterministic=True` 会打开 cudnn deterministic 并关闭 benchmark。`setup_accelerator()` 配置 30 分钟 process-group timeout；`amp=False` 时 mixed precision 固定为 `"no"`。

### 统一结果 JSON

```python
save_run_results(
    save_dir, model, split, seed, metrics,
    config=None, extra=None,
) -> str
```

写入 `save_dir/results.json` 并返回路径。schema：

```json
{
  "model": "...",
  "split": "...",
  "seed": 42,
  "git_sha": "...",
  "source_bundle": null,
  "source_bundle_sha256": null,
  "timestamp": "UTC ISO-8601",
  "metrics": {},
  "config": {},
  "execution": {"cell_id": "...", "physical_gpus": "..."},
  "extra": {}
}
```

`execution`/`extra` 仅有内容时出现；非 JSON 对象由内部 `_jsonable()` 转为 `repr()`。再次调用会覆盖同目录的 `results.json`，不会追加历史。

### 通用 checkpoint helper

```python
save_checkpoint(
    path, model, optimizer, scheduler=None,
    epoch=None, step=None, extra=None, logger=None,
) -> None

load_checkpoint(
    path, model, optimizer=None, scheduler=None,
    device=torch.device("cpu"), logger=None,
) -> dict
```

标准 dict 至少含 `model` 和 `optimizer`，可含 `scheduler`、`epoch`、`step`；`extra` 用 `dict.update()` 合入，若传同名 key 会覆盖标准字段。`save_checkpoint()` 会 `os.makedirs(os.path.dirname(path))`，所以只传不含目录的文件名会因空目录字符串失败。`load_checkpoint()` 要求 `model` key 存在，optimizer/scheduler key 缺失则跳过；它使用常规 `torch.load()`，只加载可信文件。

## 各 trainer 的 checkpoint 不统一

| Trainer | 产物格式 | 可否直接恢复 optimizer |
|---|---|---|
| SASRec / HSTU / RPG / GRAM / TIGER | `*.pt` 是 raw model `state_dict` | 否 |
| COBRA | dict：`epoch/model/optimizer/scheduler` | 是，仅 COBRA resume 路径 |
| RQ-VAE | dict：`epoch|iter/model/model_config/optimizer/scheduler` | 是 |
| RQ-KMeans | `RqKmeans.save()` artifact | 不适用 |
| LCRec / OneRec SFT/GRPO/CLM | Hugging Face 风格模型+tokenizer目录 | 否；不含 optimizer/scheduler/epoch |
| NoteLLM | `best_model.pt`：`trainable_state_dict` + best metadata；另存 tokenizer 目录 | 否；需同一 base model/结构，`strict=False` 载入 |

“checkpoint”这个名称不能跨 trainer 推断格式。加载前先看对应保存代码，不要统一交给 `load_checkpoint()`。

## 可选依赖的失败策略

- LCRec、OneRec SFT 和 OneRec CLM 请求 LoRA 但未安装 PEFT 时只写 warning，并继续全参数训练；这可能显著改变显存与实验语义。
- OneRec GRPO 在该情况下把 `use_lora` 改为 `False`，当 `beta>0` 时另建一份冻结 reference model。
- NoteLLM 则明确抛 `RuntimeError`，拒绝静默回退。
- RPG 的 OPQ 数据生成需要 FAISS；RQ-KMeans 需要 scikit-learn；GRAM/LCRec/OneRec/NoteLLM 需要 Transformers 与对应本地或可下载权重。
- W&B 仅在各入口的 `wandb_logging=True` 时初始化，但会调用 `wandb.login()`，无凭据/网络时异常直接向上传播。

## CLI

所有入口从仓库根目录运行，并接收 `CONFIG --split ... --gin ...`：

```bash
python genrec/trainers/sasrec_trainer.py config/sasrec/amazon.gin --split beauty
python genrec/trainers/hstu_trainer.py config/hstu/amazon.gin --split beauty

python genrec/trainers/rqvae_trainer.py config/tiger/amazon/rqvae.gin --split beauty
python genrec/trainers/tiger_trainer.py config/tiger/amazon/tiger.gin --split beauty
python genrec/trainers/lcrec_trainer.py config/lcrec/amazon/lcrec.gin --split beauty
python genrec/trainers/cobra_trainer.py config/cobra/amazon/cobra.gin --split beauty

python genrec/trainers/rqkmeans_trainer.py config/onerec/amazon/rqkmeans.gin --split beauty
python genrec/trainers/onerec_sft_trainer.py config/onerec/amazon/sft.gin --split beauty
python genrec/trainers/onerec_grpo_trainer.py config/onerec/amazon/grpo.gin --split beauty
python genrec/trainers/onerec_clm_trainer.py config/onerec/amazon/clm.gin --split beauty

python -m genrec.trainers.rpg_trainer config/rpg/amazon.gin --split beauty
python genrec/trainers/notellm_trainer.py config/notellm/amazon.gin --split beauty
python genrec/trainers/notellm_trainer.py config/notellm/amazon.gin --split sports
python genrec/trainers/gram_trainer.py config/gram/amazon/gram.gin --split Beauty
```

Gin 解析细节见 [Gin 与 CLI API](configs.md)。同名 `train` selector 只在对应脚本 import 的模块上下文中有意义。

## 扩展新 trainer

仓库内可复用的最小组合是：

```python
@gin.configurable
def train_my_model(dataset=None, save_dir_root="out/my_model", seed=42):
    set_seed(seed)
    accelerator = setup_accelerator()
    # 直接构造 dataset/model/dataloader，并明确 collate schema
    # 所有 rank 共同进入 collective eval；仅 main process 写文件
    save_run_results(
        save_dir_root, "my_model", get_run_split(), seed,
        metrics={"Recall@10": 0.0}, config={"seed": seed},
    )

if __name__ == "__main__":
    parse_config()
    train_my_model()
```

新增代码不要假定 `TrainerConfig` 会自动生效；需要自己声明 Gin 参数、checkpoint schema、分布式同步和 item-level 指标语义。
