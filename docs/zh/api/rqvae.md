# RQ-VAE API

本页只描述当前 [`genrec/models/rqvae.py`](https://github.com/phonism/genrec/blob/main/genrec/models/rqvae.py) 的调用契约。模型原理、论文映射和下游关系见 [RQ-VAE 模型说明](../models/rqvae.md)。RQ-VAE 在这里是 item embedding → semantic ID 的量化器，不是直接产生推荐列表的 recommender。

## 导入

```python
from genrec.models.rqvae import (
    RqVae,
    RqVaeOutput,
    RqVaeComputedLosses,
    Quantize,
    QuantizeOutput,
    QuantizeForwardMode,
    QuantizeDistance,
)
```

`RqVae`、`QuantizeForwardMode` 也由 `genrec.models` 重导出；其余符号建议从实现模块导入。

## Enum

```python
class QuantizeForwardMode(Enum):
    GUMBEL_SOFTMAX = 1
    STE = 2
    ROTATION_TRICK = 3
    SINKHORN = 4

class QuantizeDistance(Enum):
    L2 = 1
    COSINE = 2
```

`QuantizeForwardMode` 用 `@gin.constants_from_enum` 注册，可在 Gin 中写 `%genrec.models.rqvae.QuantizeForwardMode.STE`。`QuantizeDistance` 没有注册为 Gin 常量；`RqVae` 当前也固定为每层使用 L2 距离，没有把 distance 暴露到构造函数。

## 单层 `Quantize`

```python
Quantize(
    embed_dim: int,
    n_embed: int,
    do_kmeans_init: bool = True,
    codebook_normalize: bool = False,
    sim_vq: bool = False,
    commitment_weight: float = 0.25,
    forward_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
    distance_mode: QuantizeDistance = QuantizeDistance.L2,
)
```

公开接口：

```python
quantizer.weight -> Tensor                 # [K, E]，原始 embedding 权重
quantizer.device -> torch.device
quantizer.get_item_embeddings(item_ids)   # [..., E]，经过可选 projection/L2Norm
quantizer.distances(x: Tensor) -> Tensor  # [B, E] -> [B, K]
quantizer(x, temperature) -> QuantizeOutput

class QuantizeOutput(NamedTuple):
    embeddings: Tensor  # [B, E]
    ids: Tensor         # [B]
    loss: Tensor        # [B]
```

`forward()` 要求 `x.shape[-1] == embed_dim`，不满足时触发 `AssertionError`。第一次前向若 `do_kmeans_init=True`，会直接用当前 batch 做 K-means 初始化；样本数少于 `n_embed` 时底层无放回抽样会报错。

训练态的梯度路径由 `forward_mode` 决定：

| mode | 训练态输出 |
|---|---|
| `GUMBEL_SOFTMAX` | 对 `-distance` 加 Gumbel 噪声，softmax 后对 codebook 加权；`ids` 仍是加噪前最近邻。 |
| `STE` | 最近邻 embedding，使用 `x + (embedding - x).detach()`。 |
| `ROTATION_TRICK` | 最近邻 embedding，经 rotation-trick 梯度变换。 |
| `SINKHORN` | 对 batch×code 成本做平衡 OT，取每行 argmax，再用 STE；此时 `ids` 改为 Sinkhorn 分配。 |

eval 态一律使用最近邻，不执行上述训练态分支。L2 距离返回平方欧氏距离；cosine 模式返回负 cosine similarity。零范数输入/codebook 在 cosine 模式没有显式 epsilon，可能产生 NaN。未知 mode 会抛 `ValueError`。

## `RqVae`

```python
RqVae(
    input_dim: int,
    embed_dim: int,
    hidden_dims: list[int],
    codebook_size: int,
    codebook_kmeans_init: bool = True,
    codebook_normalize: bool = False,
    codebook_sim_vq: bool = False,
    codebook_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
    codebook_last_layer_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
    n_layers: int = 3,
    commitment_weight: float = 0.25,
    n_cat_features: int = 18,
)
```

encoder 是 `input_dim → hidden_dims → embed_dim` 的 MLP；decoder 反转 `hidden_dims`，输出 `input_dim`。共有 `n_layers` 个残差量化层，只有最后一层采用 `codebook_last_layer_mode`。`codebook_normalize=True` 只给第一层 codebook 的输出加 L2Norm，同时让 encoder 输出归一化。

### 编解码与 semantic ID

令 `B` 为 batch size、`D=input_dim`、`E=embed_dim`、`C=n_layers`：

```python
model.encode(x: Tensor) -> Tensor  # [B, D] -> [B, E]
model.decode(z: Tensor) -> Tensor  # [B, E] -> [B, D]

model.get_semantic_ids(
    x: Tensor,                # [B, D]
    gumbel_t: float = 0.001,
) -> RqVaeOutput

class RqVaeOutput(NamedTuple):
    embeddings: Tensor   # [B, E, C]
    residuals: Tensor    # [B, E, C]
    sem_ids: Tensor      # [B, C]
    quantize_loss: Tensor  # [B]
```

每层先保存当前 residual，再量化并把 embedding 从 residual 中扣除。完整重建 latent 是 `output.embeddings.sum(dim=-1)`。

### 去碰撞 tokenizer

```python
model.get_semantic_ids_dedup(
    x: Tensor,             # [N, D]
    max_iter: int = 20,
    sk_epsilon: float = 0.003,
    sk_iters: int = 100,
) -> LongTensor           # [N, C]
```

该方法先固定前 `C-1` 层的最近邻 code，仅对碰撞组的最后一层用 Sinkhorn 重新分配。它会调用 `model.eval()` 且不会恢复原训练状态。若达到固定点/迭代上限后仍有碰撞，只写 warning，仍返回有碰撞的 Tensor；调用方需要自行检查唯一率。

`n_layers=1` 可以正常进入“无前置层”的路径。`n_layers=0` 不受支持：最后层访问会越界。

### 训练前向与返回值

```python
model(
    batch: Tensor,  # [B, D]
    gumbel_t: float,
) -> RqVaeComputedLosses

class RqVaeComputedLosses(NamedTuple):
    loss: Tensor                 # scalar，总 loss 的 batch mean
    reconstruction_loss: Tensor # scalar
    rqvae_loss: Tensor           # scalar
    embs_norm: Tensor            # [B, C]
    p_unique_ids: Tensor         # scalar，当前 batch 唯一 tuple 比例
```

`forward()` 带 `@torch.compile(mode="reduce-overhead")`。重建与量化 loss 都先逐样本计算，再求 mean。`n_cat_features > 0` 时，输出的前 `D-n_cat_features` 维先 L2 归一化，末尾维直接作为 BCE logits；输入末尾维应是 0/1 target。`n_cat_features > input_dim` 不会被显式拒绝，但切片语义会错误，应由配置保证。

### 属性与 checkpoint

```python
model.device -> torch.device
model.config -> dict
model.load_pretrained(path: str) -> None
RqVae.from_pretrained(path: str, map_location="cpu") -> RqVae
```

- `load_pretrained()` 用当前对象的几何结构载入 `checkpoint["model"]`；结构不匹配由 `load_state_dict()` 抛错。载入后关闭每层 K-means 初始化，避免第一次推理覆盖 codebook。
- `from_pretrained()` 从 `checkpoint["model_config"]` 过滤构造参数并重建模型，然后载入权重。checkpoint 缺失几何配置时会因必填构造参数缺失而失败；旧 checkpoint 应先按原配置显式构造，再用 `load_pretrained()`。
- 当前 `config` 捕获构造函数的 `locals()`，其中包含内部项，不保证 JSON 可序列化；把它当 checkpoint 元数据，不要当稳定配置 schema。
- 两种载入方式都要求文件含 `"model"` 键；不存在时抛 `KeyError`。`torch.load(..., weights_only=False)` 会反序列化 Python 对象，只应加载可信 checkpoint。

## 最小调用示例

```python
import torch
from genrec.models.rqvae import RqVae, QuantizeForwardMode

model = RqVae(
    input_dim=768,
    embed_dim=32,
    hidden_dims=[512, 256, 128, 64],
    codebook_size=256,
    n_layers=3,
    n_cat_features=0,
    codebook_mode=QuantizeForwardMode.STE,
    codebook_last_layer_mode=QuantizeForwardMode.SINKHORN,
)

item_embeddings = torch.randn(1024, 768)  # 首次 K-means batch 应至少覆盖 K 个样本
losses = model(item_embeddings, gumbel_t=0.2)
losses.loss.backward()

model.eval()
semantic_ids = model.get_semantic_ids(item_embeddings).sem_ids  # [1024, 3]
```

## Trainer API

源文件：[`rqvae_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/rqvae_trainer.py)

```python
@gin.configurable
def train(
    epochs=None,
    iterations=None,
    warmup_epochs=0,
    batch_size=64,
    learning_rate=1e-4,
    weight_decay=0.01,
    dataset_folder="dataset/amazon",
    dataset=P5AmazonReviewsItemDataset,
    pretrained_rqvae_path=None,
    save_dir_root="out/rqvae/amazon/",
    use_kmeans_init=True,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    do_eval=True,
    mixed_precision_type="fp16",
    save_model_every=1000000,
    eval_every=50000,
    # ... vae_*、encoder_model_name、seed
) -> None
```

`epochs` 与 `iterations` 必须二选一：都为 `None` 或同时设置都会抛 `ValueError`。epoch 模式中 `save_model_every`/`eval_every` 按 epoch 计数，iteration 模式按 step 计数。dataset 构造契约为：

```python
dataset(
    root=dataset_folder,
    train_test_split="train" | "eval",
    encoder_model_name=encoder_model_name,
)
```

每个样本必须可转成长度 `vae_input_dim` 的 float Tensor。首次 K-means 初始化会累计约 20,000 个训练样本。训练依赖 Accelerate；DataLoader 固定 `num_workers=16`、`persistent_workers=True` 和较高 prefetch，资源受限环境需要改源码而非 Gin 才能降低 worker 数。

## 运行与产物

必须使用下游模型匹配的 RQ-VAE 配置，因为 code 层数、维度和 codebook 大小不同：

```bash
python genrec/trainers/rqvae_trainer.py config/tiger/amazon/rqvae.gin --split beauty
python genrec/trainers/rqvae_trainer.py config/lcrec/amazon/rqvae.gin --split beauty
python genrec/trainers/rqvae_trainer.py config/cobra/amazon/rqvae.gin --split beauty
```

可重复传 `--gin` 覆盖：

```bash
python genrec/trainers/rqvae_trainer.py config/tiger/amazon/rqvae.gin \
  --split sports \
  --gin 'train.epochs=1000' \
  --gin 'train.wandb_logging=False'
```

checkpoint 文件为 epoch 模式的 `checkpoint_epoch_<epoch>.pt` 或 iteration 模式的 `checkpoint_<iter>.pt`，包含：

```text
epoch 或 iter
model
model_config
optimizer
scheduler
```

训练结束还会在 `save_dir_root` 写 `results.json`；若 `do_eval=False`，其 metrics 可能为空。checkpoint 路径必须与下游 TIGER/LCRec/COBRA 的配置一致。

启用健康评估时，门槛前的最佳诊断快照写到 `checkpoint_best_candidate.pt`；只有所有层达到 `min_codebook_usage` 才会写 `checkpoint_best_health.pt`。论文配方下游只应读取后者。

## 常见误用

- 当前方法名是 `get_semantic_ids()`，不存在旧文档中的 `generate_semantic_ids()`。
- `RqVae.forward()` 返回 NamedTuple，不返回 `(reconstruction, codes)`。
- SID tuple 不必然对应唯一 item；推荐评测前使用 [`SemanticIdItemMapper`](modules.md) 或显式去碰撞策略。
- 不要对小于 codebook 的首 batch 开启 K-means 初始化。
- `get_semantic_ids_dedup()` 只调整最后一个 code，不能保证完全无碰撞。
- 恢复训练请使用 trainer 的 `pretrained_rqvae_path`，它除了模型权重还恢复 optimizer/scheduler；单独调用 `load_pretrained()` 只载入模型。
