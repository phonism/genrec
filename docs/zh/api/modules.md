# 公用模块 API

本页索引 [`genrec/modules/`](https://github.com/phonism/genrec/tree/main/genrec/modules) 中当前可调用的组件。它们是模型间复用的 PyTorch 工具，不构成独立的稳定 SDK；模型专用层仍应查看对应模型源码。

## 包级导出

[`genrec.modules.__init__`](https://github.com/phonism/genrec/blob/main/genrec/modules/__init__.py) 只重导出以下符号：

```python
from genrec.modules import (
    ReconstructionLoss, CategoricalReconstructionLoss, QuantizeLoss,
    TopKAccumulator, L2Norm, RMSNorm,
    SemIdEmbedding, UserIdEmbedding,
    parse_config, setup_logger,
)
```

`SemanticIdItemMapper`、encoder、Transformer、K-means、scheduler 和纯 PyTorch T5 均须从具体模块导入。

## Embedding

源文件：[`embedding.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/embedding.py)

### `SemIdEmbedding`

```python
SemIdEmbedding(
    num_embeddings: int,
    sem_ids_dim: int,
    embeddings_dim: int,
)

forward(
    input_ids: Tensor,       # [B, T]
    token_type_ids: Tensor,  # [B, T]
) -> Tensor                 # [B, T, D]
```

第 `c` 个码本的 code `i` 被映射为全局索引 `c * num_embeddings + i`。Embedding 表大小为 `sem_ids_dim * num_embeddings + 1`，最后一个位置是 padding；调用者必须自行把 padding code/type 组合成该索引，类本身不校验 code 范围。

### `UserIdEmbedding`

```python
UserIdEmbedding(num_embeddings: int, embeddings_dim: int)
forward(input_ids: Tensor) -> Tensor  # [B, T] -> [B, T, D]
```

用户 ID 先按 `num_embeddings` 取模，因此不同原始 ID 可能哈希碰撞。

## 文本 encoder 与 MLP

源文件：[`encoder.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/encoder.py)

| 类 | 构造与输入 | 返回值 / 依赖 |
|---|---|---|
| `LightT5Encoder` | `(n_layers=1, hidden_dim=768, output_dim=768, num_heads=8, ff_dim=2048, vocab_size=32128, max_seq_len=512, dropout=0.1)`；`forward([B,L] 或 [B,T,L])` | L2 归一化的 `[B,D]` 或 `[B,T,D]`。随机初始化，token `0` 当 padding；输入不是 2D/3D 时 `ValueError`。名称虽含 T5，内部使用 `nn.TransformerEncoder`。 |
| `SentenceT5Encoder` | `(model_name="./models_hub/sentence-t5-base", output_dim=768)`；同上 | 从 `sentence-transformers` 模型抽取 encoder/pooling/可选 dense，并启用 gradient checkpointing；返回归一化向量。依赖模型目录的模块布局包含 `"0"`、`"1"`。 |
| `ErnieEncoder` | `(model_name="./models_hub/ernie-3.0-medium-zh", out_dim=768)`；`tokenize(texts)`、`encode(texts)`、`forward([B,L] 或 [B,T,L])` | Hugging Face `AutoTokenizer/AutoModel` 的 CLS 向量；`encode()` 最终 reshape 为 `[-1,out_dim]`。 |
| `BgeEncoder` | `(model_name="./models_hub/bge-base-zh")`；`forward(input_ids, attention_mask, token_type_ids)` | 输入各为 `[B,L]` 或 `[B,T,L]`，返回 L2 归一化 CLS `[B,D]` / `[B,T,D]`；`encode(texts, device="cuda")` 的 `device` 参数当前未被使用。 |
| `MLP` | `(input_dim, hidden_dims, out_dim, dropout=0.0, normalize=False)`；`forward(x)` | 最后一维必须等于 `input_dim`，否则 `AssertionError`；中间层使用 SiLU，可选 dropout，最后可选 L2 归一化。 |

`LightT5Encoder` 和 `SentenceT5Encoder` 的位置长度由输入 `L` 决定；前者若 `L > max_seq_len`，位置 embedding 查表会越界。
四个 encoder 都只在内部 `T > 1` 时恢复三维形状；输入恰为 `[B,1,L]` 时实际返回 `[B,D]`，而不是 `[B,1,D]`。需要保留单 item 维的调用方应自行 `unsqueeze(1)`。

## Transformer

源文件：[`transformer.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/transformer.py)

### 注意两个同名类

仓库中有两个不同的 `T5Attention`：

```python
from genrec.modules.transformer import T5Attention  # 通用 encoder-decoder 组件
from genrec.modules.t5 import T5Attention           # 完整 T5 内部层
```

不要用模糊的包级导入替换它们。

### 通用 Transformer 签名

```python
T5Attention(
    d_model, n_heads, dropout=0.0,
    is_cross_attention=False, has_relative_bias=True,
    num_relative_buckets=32, max_distance=128,
    d_kv=0, scale_attn=True, bidirectional=True,
)
forward(
    query, key=None, value=None,
    attn_mask=None, key_padding_mask=None,
    position_bias=None, need_weights=False,
) -> tuple[Tensor, Tensor | None]
```

`query`/`key`/`value` 采用 `[B,L,D]`。输出是 attention 结果 `[B,Q,D]` 与可在层间共享的 position bias `[1,H,Q,K]`。`need_weights` 当前不改变返回值；self-attention 忽略传入的 `key/value`，直接从 `query` 生成 K/V。`d_kv=0` 时要求 `d_model % n_heads == 0`。

其余层：

```python
FeedForward(dim, hidden_dim=2048, dropout=0.1)

MultiHeadAttention(
    dim, num_heads, dropout=0.1, kv_dim=None, bias=True,
    is_cross_attention=False, d_kv=0,
    has_relative_bias=True, scale_attn=True, bidirectional=True,
)
forward(query, key=None, value=None, attn_mask=None,
        key_padding_mask=None, position_bias=None)
    -> tuple[Tensor, Tensor | None]

TransformerBlock(
    dim, num_heads, dropout=0.1, norm_cls=RMSNorm,
    ff_hidden_dim=2048, cross_attn=False, d_kv=0,
    has_relative_bias=True, scale_attn=True,
    decoder_bidirectional=True,
)
forward(x, *, context=None, attn_mask=None,
        key_padding_mask=None, memory_key_padding_mask=None,
        position_bias=None) -> tuple[Tensor, Tensor | None]
```

高层容器均接收 batch-first `[B,L,D]`：

```python
TransformerEncoder(
    dim, depth, num_heads, dropout=0.1, norm_cls=RMSNorm,
    ff_hidden_dim=2048, d_kv=0, share_position_bias=False,
    scale_attn=True, add_final_norm=False,
)
forward(src, *, attn_mask=None, key_padding_mask=None) -> Tensor

TransformerDecoder(
    dim, depth, num_heads, dropout=0.1, norm_cls=RMSNorm,
    ff_hidden_dim=2048, d_kv=0, share_position_bias=False,
    scale_attn=True, add_final_norm=False,
    decoder_bidirectional=True,
)
forward(tgt, *, memory, attn_mask=None, key_padding_mask=None,
        memory_key_padding_mask=None) -> Tensor

TransformerEncoderDecoder(
    d_model, nhead, num_encoder_layers, num_decoder_layers,
    dim_feedforward=2048, dropout=0.1, norm_cls=RMSNorm,
    d_kv=0, share_position_bias=False, scale_attn=True,
    add_final_norm=False, decoder_bidirectional=True,
)
forward(src, tgt, *, src_key_padding_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None,
        src_mask=None, tgt_mask=None) -> Tensor
```

`TransformerEncoderDecoder.forward()` 在未提供 `tgt_mask` 时自动创建上三角 causal additive mask。布尔 padding mask 中 `True` 表示忽略；attention mask 是加到 score 上的 additive mask，不是 Hugging Face 的 0/1 mask。

## 纯 PyTorch T5

源文件：[`t5.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/t5.py)。这是 TIGER 使用的本地实现；运行时不依赖 Hugging Face `transformers`，但参数布局用于与其 T5 权重对齐。

```python
@dataclass
class T5Config:
    vocab_size: int = 32128
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 6
    num_decoder_layers: int | None = None
    num_heads: int = 8
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "relu"
    is_encoder_decoder: bool = True
    use_cache: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    decoder_start_token_id: int | None = None
    is_decoder: bool = False
    tie_word_embeddings: bool = True
```

```python
T5ForConditionalGeneration(config: T5Config)

forward(
    input_ids=None, attention_mask=None,
    decoder_input_ids=None, decoder_attention_mask=None,
    labels=None, encoder_outputs=None,
    use_cache=False, past_key_values=None,
) -> _Seq2SeqLMOutput

generate(
    input_ids, attention_mask=None,
    max_length=20, num_beams=1, num_return_sequences=1,
) -> LongTensor
```

返回对象有 `.loss`、`.logits`、`.past_key_values`。传 `labels` 且不传 `decoder_input_ids` 时会右移 label；交叉熵忽略 `-100`。`generate()` 仅实现 greedy/beam search，不支持采样、logits processor 或 Hugging Face `GenerationConfig`。输出包含 decoder start token；beam 分支是 `[B * num_return_sequences,max_length]`，greedy 是 `[B,L]` 且所有样本同时生成 EOS 时 `L` 可小于 `max_length`。

## Loss

源文件：[`loss.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/loss.py)

| API | 返回 |
|---|---|
| `ReconstructionLoss()(x_hat, x)` | 对最后一维求和的逐样本平方误差，形状通常 `[B]`，没有 batch mean。 |
| `CategoricalReconstructionLoss(n_cat_feats)(x_hat, x)` | 前部连续特征平方误差，加末尾 `n_cat_feats` 个 logit 的逐元素 BCE，再对特征求和；返回 `[B]`。 |
| `QuantizeLoss(commitment_weight=1.0)(query, value)` | codebook loss + `commitment_weight * commitment loss`，只沿最后一维求和；返回保留前导维度的 Tensor。 |

`CategoricalReconstructionLoss` 把 `x_hat` 的分类部分当 logits，调用者不要预先 sigmoid。单独构造时应保证 `n_cat_feats > 0`：传 0 会让 `x_hat[:, :-0]` 成为空切片并返回零 reconstruction；纯连续特征应使用 `ReconstructionLoss`。`RqVae` 已在构造时做了这处分派。

## Metric 与 SID→item 解析

源文件：[`metrics.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/metrics.py)

```python
SemanticIdItemMapper(
    semantic_ids: Sequence[Sequence[int]],
    item_popularity: Mapping[int, int] | Sequence[int] | None = None,
    padding_item_id: int = -1,
)

SemanticIdItemMapper.from_sequences(
    semantic_ids, sequences, train_holdout=2, padding_item_id=-1,
)

map_predictions(
    predicted_semantic_ids: Tensor,  # [B, K, C]
    topk: int | None = None,
    deduplicate: bool = True,
) -> Tensor                          # [B, topk]
```

同一 SID 碰撞时，选择训练流行度最高的 item；并列时选较小 item ID。`from_sequences()` 默认从每个序列末尾扣除 2 个交互，避免用验证/测试目标算流行度。未知 SID 被跳过，结果不足 `topk` 时填 `padding_item_id`。输入不是 3D 或 `topk < 1` 时抛 `ValueError`。诊断属性包括 `collision_rate`、`num_unique_sids`、`num_collision_items` 和 `max_collision_group`。

```python
TopKAccumulator(ks: list[int] = [1, 5, 10])
accumulate(actual: Tensor, top_k: Tensor) -> None
sync_across_processes(accelerator) -> None
reduce() -> dict[str, float]
reset() -> None
```

`actual` 是 `[B,D]`，`top_k` 是 `[B,K,D]`；匹配要求最后一维全部相等。item-level 评测可把 item ID 表成 `D=1`。分布式评测必须由所有 rank 共同调用 `sync_across_processes()`，否则 collective 会挂起；`reduce()` 输出 `Recall@K` 与 `NDCG@K`。

## K-means 与 Gumbel

源文件：[`kmeans.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/kmeans.py)、[`gumbel.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/gumbel.py)

```python
kmeans_init_(tensor: Tensor, x: Tensor) -> None
Kmeans(k: int, max_iters: int | None = None, stop_threshold=1e-10)
Kmeans.run(x: Tensor) -> KmeansOutput

class KmeansOutput(NamedTuple):
    centroids: Tensor   # [K, D]
    assignment: Tensor  # [B]
```

`tensor` 与 `x` 必须为 2D，否则 `AssertionError`。中心从 `B` 个样本中无放回抽取，因此 `k > B` 会由 NumPy 抛错；空簇会随机重置为某个样本。`max_iters=None` 表示只由收敛阈值停止，没有硬迭代上限。

```python
sample_gumbel(shape: tuple, device, eps=1e-20) -> Tensor
gumbel_softmax_sample(logits, temperature, device) -> Tensor
```

后者返回 softmax 概率，不执行 straight-through hard one-hot。

## Normalization 与 scheduler

源文件：[`normalize.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/normalize.py)、[`scheduler.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/scheduler.py)

- `l2norm(x, dim=-1, eps=1e-12)` 与 `L2Norm(dim=-1, eps=1e-12)`。
- `RMSNorm(dim, eps=1e-6)`：float32 计算 RMS，再转回输入 dtype。
- `SwishLayerNorm(hidden_dim, eps=1e-5)`：`SiLU(LayerNorm(x))`。
- `RootMeanSquareLayerNorm(hidden_size, eps=1e-6)`：T5 风格 RMSNorm。
- `InverseSquareRootScheduler(optimizer, warmup_steps, last_epoch=-1)`：线性 warmup 后按 `sqrt(warmup_steps / step)` 衰减；`warmup_steps=0` 会让衰减阶段 scale 为 0，应避免。

## 通用工具

源文件：[`utils.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/utils.py)

| API | 行为 |
|---|---|
| `setup_logger(save_dir, name="train")` | 创建目录、console/file handler；同名 logger 已有 handler 时直接复用。 |
| `reset_kv_cache(fn)` | 装饰器，调用前后执行 `self.decoder.reset_kv_cache()`；若被装饰函数抛异常，末尾清理不会执行。 |
| `reset_encoder_cache(fn)` | 仅 `self.jagged_mode` 时清空 `self.transformer.cached_enc_output`。 |
| `eval_mode(fn)` | 临时切到 eval，再恢复原训练状态；异常时同样没有 `finally` 恢复。 |
| `select_columns_per_row(x, indices)` | 按每行索引选择，返回形如 `[B,M,...]`；batch 大小和列数用 `assert` 检查。 |
| `maybe_repeat_interleave(x, repeats, dim)` | Tensor 调用 `repeat_interleave`；非 Tensor 原样返回。 |
| `parse_config()` / `get_run_split()` | 见 [Gin 与 CLI](configs.md)。 |
| `compute_debug_metrics(batch, model_output=None, prefix="")` | 返回序列长度分位数；传 model output 时还读取 `model_output.loss_d`。 |

## 依赖边界

- 基础 Tensor 组件依赖 PyTorch；K-means 另依赖 NumPy、einops。
- `SentenceT5Encoder` 依赖 `sentence-transformers`；`ErnieEncoder`/`BgeEncoder` 依赖 Hugging Face `transformers` 和本地/可访问的预训练权重。
- [`t5.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/t5.py) 本身不依赖 `transformers`。
- metric 的分布式同步依赖传入兼容 Hugging Face Accelerate 的 `accelerator` 对象。
