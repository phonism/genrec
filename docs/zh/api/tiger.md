# TIGER API

本页描述当前 [`genrec/models/tiger.py`](https://github.com/phonism/genrec/blob/main/genrec/models/tiger.py) 与 [`tiger_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/tiger_trainer.py) 的精确调用接口。架构、论文和实现差异见 [TIGER 模型说明](../models/tiger.md)。

## 实现边界

`Tiger` 包装的是仓库内 [`genrec.modules.t5.T5ForConditionalGeneration`](https://github.com/phonism/genrec/blob/main/genrec/modules/t5.py)，不是 Hugging Face 类。源码顶部和旧配置注释仍写有 “HuggingFace T5”，但实际 import 是本地纯 PyTorch T5；它只提供当前训练所需的 forward、greedy/beam search 与 KV cache。

## Token 约定

假设每个 item 有 `C=sem_id_dim` 个 code，每个 codebook 大小为 `K=codebook_size`。第 `c` 个 codebook 的原始 code `q_c ∈ [0,K-1]` 被编码为：

```text
token = q_c + c * K + 1
```

因此词表大小是 `C*K+1`，token `0` 同时作为 padding、decoder start 和 EOS。history 是多个 item 的 code 依次展平后的 token 流；target label 是一个 item 的 `C` 个 offset token，不额外附加 EOS。

这个 offset 只提供 codebook 位置分区；当前 generation 没有 logits mask、trie 或 constrained decoding。模型可在任意位置生成词表内任意 token，解码后超出 `[0,K-1]` 的 raw code 会形成未知 SID，item mapper 会跳过它。

## `Tiger`

```python
from genrec.models.tiger import Tiger

Tiger(config: dict[str, Any])
```

识别的 config 键及默认值：

| 键 | 默认值 | 用途 |
|---|---:|---|
| `num_layers` | `4` | encoder 层数 |
| `num_decoder_layers` | `4` | decoder 层数 |
| `d_model` | `128` | hidden size |
| `d_ff` | `1024` | FFN size |
| `num_heads` | `6` | attention head 数 |
| `d_kv` | `64` | 每个 head 的 K/V 维度 |
| `dropout_rate` | `0.1` | T5 dropout |
| `vocab_size` | `769` | 默认对应 `3*256+1` |
| `pad_token_id` | `0` | padding 和 decoder start |
| `eos_token_id` | `0` | EOS |
| `feed_forward_proj` | `"relu"` | `relu` 或本地 T5 支持的 gated/activation 名称 |
| `sem_id_dim` | `3` | generation 的 code 数 |

未知键被忽略。构造函数不校验 `vocab_size == sem_id_dim*codebook_size+1`，因为 `codebook_size` 根本不是模型参数；trainer 负责保持一致。

### 参数统计

```python
model.num_parameters -> tuple[int, int]
```

返回 `(全部可训练参数量, input embedding 可训练参数量)`。本地 T5 的 input embedding、decoder embedding 与 LM head 权重共享。

### 训练前向

```python
loss, logits = model(
    input_ids: LongTensor,             # [B, L]
    attention_mask: Tensor | None,     # [B, L]，1=有效、0=padding
    labels: LongTensor | None,         # [B, C]
)
```

- 传入 labels：`loss` 是忽略 label `-100` 的 scalar cross entropy，`logits` 为 `[B,C,V]`。
- `labels=None` 时还需要底层 decoder input；`Tiger.forward()` 没有暴露 `decoder_input_ids`，所以这个入口实际用于有 labels 的训练。直接无 labels 调用通常会在底层 decoder 失败。
- `Tiger` 不检查 label 长度是否等于 `sem_id_dim`。

### 生成

```python
predictions = model.generate(
    input_ids: LongTensor,             # [B, L]
    attention_mask: Tensor | None = None,
    num_beams: int = 20,
)
```

模型调用本地 T5：

```python
max_length = sem_id_dim + 1
num_return_sequences = num_beams
```

当 `num_beams > 1` 时返回固定形状 `[B*num_beams, C+1]`；第 0 列是 decoder start token `0`，trainer 在评测前用 `preds[:,1:]` 删除它。当 `num_beams=1` 时走 greedy 分支；如果所有样本提前生成 EOS `0`，长度可能短于 `C+1`，不能假定固定 shape。

beam search 当前不执行 length penalty、EOS filtering、sampling、repetition penalty 或 prefix constraint；`num_beams < 1` 没有显式校验，会由后续 Tensor 操作失败。

## `t5_collate`

```python
from genrec.trainers.tiger_trainer import t5_collate

t5_collate(
    batch: list[SeqData],
    codebook_size: int = 256,
    sem_id_dim: int = 3,
    max_history_tokens: int = 60,
) -> dict[str, Tensor]
```

输入 [`SeqData`](base-dataset.md#shared-sample-types) 的 schema：

```python
SeqData(
    user_id: int,
    item_ids: list[int],    # history raw SID，已按 item 展平
    target_ids: list[int],  # target raw SID，长度 C
    target_item: int = -1,  # 原始 catalog item id
)
```

返回：

| key | dtype / shape | 说明 |
|---|---|---|
| `input_ids` | long `[B,max_history_tokens]` | offset token；超长时保留尾部，左侧补 0 |
| `attention_mask` | long `[B,max_history_tokens]` | history 位置为 1，padding 为 0 |
| `labels` | long `[B,C]` | offset target token |
| `raw_targets` | long `[B,C]` | 未 offset 的 target SID |
| `target_items` | long `[B]` | 原始 catalog item ID |

所有样本的 `target_ids` 必须等长，否则 `torch.tensor(label_list)` 会报错。`max_history_tokens` 应是 item history 上限乘 `sem_id_dim`；若截断值不是 `C` 的倍数，可能从一个 item 的 SID 中间截断。collate 不验证原始 code 范围。

## 评测 helper

[`tiger_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/tiger_trainer.py) 还保留三项 SID 位置 helper：

```python
calculate_pos_index(preds, labels, maxk=20) -> BoolTensor  # [B,K,C]、[B,C] -> [B,K]
recall_at_k(pos_index, k) -> float
ndcg_at_k(pos_index, k) -> float
```

`calculate_pos_index()` 只保留每行第一个匹配；参数 `maxk` 当前未使用。当前 trainer 的主评测路径不调用这些 helper，而使用 `TopKAccumulator`。

## Trainer

```python
@gin.configurable
def train(
    epochs=200,
    batch_size=256,
    infer_batch_size=96,
    learning_rate=1e-4,
    weight_decay=0.0,
    dataset_folder="dataset/amazon",
    save_dir_root="out/tiger/",
    dataset=None,
    wandb_logging=True,
    wandb_project="tiger_training",
    wandb_log_interval=10,
    num_layers=4,
    num_decoder_layers=4,
    d_model=128,
    d_ff=1024,
    num_heads=6,
    d_kv=64,
    dropout_rate=0.1,
    feed_forward_proj="relu",
    codebook_size=256,
    sem_id_dim=3,
    max_seq_len=20,
    pretrained_rqvae_path="./out/tiger/amazon/{split}/rqvae/checkpoint_epoch_4999.pt",
    lr_schedule="none",
    num_warmup_steps=0,
    beam_size=30,
    early_stop_patience=10,
    eval_test_every_epoch=2,
    eval_valid_every_n_epochs=1,
    seed=42,
) -> None
```

trainer 是普通函数，不存在 `TIGERTrainer` 类。它自行选择 `cuda`/`cpu`，不使用 Accelerate，也不支持多进程同步。dataset 必须能按下列形式构造：

```python
dataset(
    root=dataset_folder,
    train_test_split="train" | "valid" | "test",
    max_seq_len=max_seq_len,
    pretrained_rqvae_path=pretrained_rqvae_path,
)
```

并提供：

- `__getitem__() -> SeqData`；
- `sem_ids_list`：按 item ID 排列的 raw SID；
- `sequences`：按 user 排列的 raw item ID 序列；
- `evaluation_protocol` 字符串。

仓库配置使用 [`AmazonSeqDataset`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon.py)。其 `rqvae_input_dim/embed_dim/hidden_dims/codebook_size/n_layers` 必须与 checkpoint 一致；这些参数通过 Gin 类绑定，而不是传给 `train()`。

## 评测语义

每次评测有两条指标流：

1. offset token 去掉 start 后直接比较，得到 SID-level pre-dedup 诊断；
2. 还原 raw SID，经 `SemanticIdItemMapper` 用训练 prefix 的流行度把每个 SID 确定性解析为一个 item，去重后计算 item-level Recall/NDCG。

主验证/early stop 使用 item-level `Recall@10`。`target_item < 0` 会抛 `ValueError`，因此使用默认 `SeqData.target_item=-1` 的旧数据适配器不能静默参与评测。指标只计算 K=5、10；`beam_size` 小于 10 时 `TopKAccumulator` 会按现有候选数截断。

## Scheduler 与停止条件

- `lr_schedule="none"`：普通 Adam，无 scheduler。
- `lr_schedule="cosine"`：按 batch step 的 cosine；`num_warmup_steps>0` 时先 `LinearLR(start_factor=0.01)`。
- 其他字符串也走无 scheduler 分支，不抛错。
- 每 `eval_valid_every_n_epochs` 做验证，epoch 0 无条件验证。
- 仅严格大于历史最佳的 item Recall@10 才保存。默认初值为 0；若第一次及以后都恰为 0，不会产生 `best_model.pt`。
- `early_stop_counter` 只在实际验证的 epoch 增加；达到 patience 后停止。
- `wandb_log_interval` 当前未在循环中使用；W&B 按 epoch 写聚合日志，不按该 step 间隔写。

## 运行与产物

先训练匹配的 RQ-VAE，再训练 TIGER：

```bash
python genrec/trainers/rqvae_trainer.py config/tiger/amazon/rqvae.gin --split beauty
python genrec/trainers/tiger_trainer.py config/tiger/amazon/tiger.gin --split beauty
```

覆盖示例：

```bash
python genrec/trainers/tiger_trainer.py config/tiger/amazon/tiger.gin \
  --split sports \
  --gin 'train.wandb_logging=False' \
  --gin 'train.beam_size=20'
```

`save_dir_root` 中的主要产物：

| 文件 | 内容 |
|---|---|
| `best_model.pt` | 仅 `model.state_dict()`，没有 model config、optimizer、scheduler 或 epoch |
| `tiger_<timestamp>.log` | train/valid/test 与碰撞解析日志 |
| `results.json` | 最佳验证 epoch 对应的 item-level test metrics；`extra` 中含 best epoch、SID 参考指标和解析策略 |

加载 `best_model.pt` 时必须用相同 config 重建 `Tiger` 后调用 `load_state_dict()`：

```python
state = torch.load("out/tiger/amazon/beauty/best_model.pt", map_location="cpu")
model = Tiger(model_config)
model.load_state_dict(state)
```

## 常见误用

- 不存在旧文档中的 `generate_with_trie()`、`TIGERTrainer` 或通用 `Trainer.fit()`。
- 当前 `Tiger.generate()` 没有 trie/constrained decoding；不要把 offset token 误称为合法性约束。
- generation 输出含开头的 decoder start `0`；还原 SID 前必须删除第一列并减去每个 codebook 的 offset。
- `best_model.pt` 不是完整恢复训练 checkpoint。
- `AmazonSeqDataset.add_disambiguation=True` 会给 SID 增加第 4 个 suffix；若 trainer 仍配置 `sem_id_dim=3` 会错位。当前 TIGER Gin 显式设为 `False`。
- trainer 的评测 autocast 上下文写死为 CUDA bfloat16；无 CUDA 环境虽可构造 CPU device，但这条路径主要按 CUDA 环境设计。
