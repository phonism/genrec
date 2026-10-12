# Gin 配置系统

## 解析顺序

所有主训练入口使用 [`parse_config()`](https://github.com/phonism/genrec/blob/main/genrec/modules/utils.py)。CLI 形状固定为：

```text
python <trainer.py> CONFIG_PATH [--split SPLIT] [--gin BINDING ...]
```

实现严格按以下顺序执行：

1. 读取 `CONFIG_PATH` 的完整文本；
2. 将文本中每个字面量 `{split}` 替换为 `--split`，默认 `beauty`；
3. `gin.parse_config(config_text)`；
4. 对每个 `--gin` binding 同样替换 `{split}`；
5. `gin.parse_config(overrides)`，因此 CLI 覆盖文件绑定；
6. 把 resolved split/config path 写入进程全局 `_RUN_CONTEXT`。

`get_run_split()` 读取这个上下文供 `results.json` 使用。它不是 Dataset 自己的 split 证明；若 CLI `--split beauty` 却通过 Gin 把 Dataset 绑定成 sports，结果元数据会错误。因此不要对 split 做相互矛盾的覆盖。

`parse_config()` 不会调用 `gin.clear_config()`。正常 CLI 每次新进程没有问题；测试或 notebook 在同一进程多次解析时应显式管理 Gin 全局状态。

## 基础语法

### 顶层函数参数

函数装饰器名称就是 selector：

```gin
train.epochs = 300
train_sft.learning_rate = 8e-5
train_grpo.num_generations = 8
train_clm.max_length = 256
train_rpg.n_codebook = 32
train_gram.num_beams = 50
fit_rqkmeans.codebook_size = 256
```

多个模块都定义 `train`，但每个 config 只 import 对应 trainer 运行时所需对象；入口函数本身已在执行模块中注册。不要把两个使用 `train` selector 的 trainer 放进同一个 Gin 进程。

### configurable class

```gin
AmazonSASRecDataset.evaluation_protocol = "leave_one_out"
AmazonOneRecSFTDataset.rqkmeans_n_layers = 3
```

trainer 显式传入的 constructor 参数优先于 Gin 注入同名参数。维护配置时应检查调用位置，不能只看 class binding。

### 对象引用、宏和 enum

```gin
train.dataset = @AmazonLCRecDataset
AmazonSeqDataset.encoder_model_name = %MODEL_HUB_SENTENCE_T5_XL
train.vae_codebook_mode = %genrec.models.rqvae.QuantizeForwardMode.STE
```

- `@Name` 传入已注册 configurable 对象/类；
- `%NAME` 引用 Gin macro；
- enum 值在 RQ-VAE 配置中通过完整符号 macro 引用。

使用前必须 `import` 注册定义它的模块：

```gin
import genrec.data.amazon_lcrec
import genrec.models.lcrec
```

### include

```gin
include "config/base.gin"
```

相对路径和 Dataset/checkpoint/model path 都依赖当前工作目录。项目命令应从仓库根目录运行。

## CLI 覆盖

每个 binding 单独传一个 `--gin`，shell 外层推荐单引号：

```bash
python genrec/trainers/hstu_trainer.py config/hstu/amazon.gin \
  --split beauty \
  --gin 'train.seed=43' \
  --gin 'train.wandb_logging=False' \
  --gin 'train.loss_type="sampled_softmax"' \
  --gin 'AmazonHSTUDataset.evaluation_protocol="global_temporal"'
```

常见类型：

```bash
--gin 'train.epochs=3'                    # int
--gin 'train.learning_rate=5e-5'          # float
--gin 'train.use_temporal_bias=False'     # bool，Gin 大写
--gin 'train.save_dir_root="out/debug"'   # string，Gin 内仍需引号
--gin 'train.max_train_samples=1000'       # debug cap
```

不要依赖 shell 自动给字符串补引号，也不要把 Python 表达式、环境变量展开或未注册名称当成 Gin 值。

## `{split}` 的边界

替换是纯 `str.replace`，不是模板引擎：

```gin
train.save_dir_root = "out/model/amazon/{split}"
AmazonSeqDataset.split = "{split}"
```

`--split toys` 会把所有字面量替换为 `toys`，包括 include 后解析前的当前文件文本、字符串和注释。它不处理 `{dataset}`、`${VAR}`、Jinja 或 Python format spec。

trainer 函数默认路径中写了 `{split}` 并不代表会自动替换；只有配置文件文本/CLI override 经 `parse_config` 才替换。代码内部只有少数 Dataset 又显式调用 `.format(split=self.split)` 处理 checkpoint path。

## `config/base.gin`

[`base.gin`](https://github.com/phonism/genrec/blob/main/config/base.gin) 当前定义五个宏：

```text
MODEL_HUB_SENTENCE_T5_BASE
MODEL_HUB_SENTENCE_T5_XL
MODEL_HUB_QWEN3_EMBEDDING_8B
MODEL_HUB_QWEN3_1_7B
MODEL_HUB_QWEN3_4B
```

当前值是 179 上 `/root/workspace/models_hub/...` 的绝对路径。文件注释提到环境变量写法，但 `parse_config` 没有环境变量插值实现，Gin 也不会自动展开普通字符串里的 `${MODEL_HUB}`。本地运行应使用 CLI 宏覆盖或维护本地派生配置，例如：

```bash
python genrec/trainers/rqkmeans_trainer.py config/onerec/amazon/rqkmeans.gin \
  --split beauty \
  --gin 'MODEL_HUB_QWEN3_EMBEDDING_8B="/absolute/local/Qwen3-Embedding-8B"'
```

是否能在 CLI 覆盖 include 中定义的 macro，应以 Gin 解析结果验证；最稳妥的是为本机维护不提交 secret 的覆盖命令。不要把 token、账号或私有 URL 写进 `.gin`。

## 配置清单

### baseline

| 文件 | 用途 |
|---|---|
| [`sasrec/amazon.gin`](https://github.com/phonism/genrec/blob/main/config/sasrec/amazon.gin) | Amazon 2014，当前主配置 CE |
| [`sasrec/amazon2023.gin`](https://github.com/phonism/genrec/blob/main/config/sasrec/amazon2023.gin) | 通用 Amazon 适配器读取 2023 兼容文件 |
| [`hstu/amazon.gin`](https://github.com/phonism/genrec/blob/main/config/hstu/amazon.gin) | Amazon 2014 full CE |
| [`hstu/amazon_ss.gin`](https://github.com/phonism/genrec/blob/main/config/hstu/amazon_ss.gin) | sampled softmax；`local` 为每正例独立的完整目录采样，非 in-batch |
| [`hstu/amazon2023.gin`](https://github.com/phonism/genrec/blob/main/config/hstu/amazon2023.gin) | Amazon 2023 兼容文件 |

### RQ-VAE 下游

| family | tokenizer 配置 | recommender 配置 | 关键几何 |
|---|---|---|---|
| TIGER | [`tiger/amazon/rqvae.gin`](https://github.com/phonism/genrec/blob/main/config/tiger/amazon/rqvae.gin) | [`tiger/amazon/tiger.gin`](https://github.com/phonism/genrec/blob/main/config/tiger/amazon/tiger.gin) | encoder 768，embed 32，hidden 512/256/128/64，`C=3,K=256` |
| LCRec | [`lcrec/amazon/rqvae.gin`](https://github.com/phonism/genrec/blob/main/config/lcrec/amazon/rqvae.gin) | [`lcrec/amazon/lcrec.gin`](https://github.com/phonism/genrec/blob/main/config/lcrec/amazon/lcrec.gin) | encoder 768，embed 64，hidden 512/256/128，`C=5,K=256` |
| LCRec 2023 | [`lcrec/amazon2023/rqvae.gin`](https://github.com/phonism/genrec/blob/main/config/lcrec/amazon2023/rqvae.gin) | [`lcrec/amazon2023/lcrec.gin`](https://github.com/phonism/genrec/blob/main/config/lcrec/amazon2023/lcrec.gin) | 与 LCRec 2014 几何相同，根目录不同 |
| COBRA | [`cobra/amazon/rqvae.gin`](https://github.com/phonism/genrec/blob/main/config/cobra/amazon/rqvae.gin) | [`cobra/amazon/cobra.gin`](https://github.com/phonism/genrec/blob/main/config/cobra/amazon/cobra.gin) | embed 32，hidden 512/256/128/64，`C=3,K=32` |

这些 RQ-VAE 互不通用。仅仅 input embedding 都是 768 维，不足以交换 checkpoint。

COBRA 还有：

- [`cobra_sparse_only.gin`](https://github.com/phonism/genrec/blob/main/config/cobra/amazon/cobra_sparse_only.gin)：关闭 dense 训练路径的消融；
- [`cobra_v2.gin`](https://github.com/phonism/genrec/blob/main/config/cobra/amazon/cobra_v2.gin)：更深 decoder 的实验配置。

二者未显式绑定 evaluation protocol 时会使用 Dataset 默认 LOO；矩阵 runner 会通过 CLI 注入协议。

### OneRec

| 文件 | stage |
|---|---|
| [`rqkmeans.gin`](https://github.com/phonism/genrec/blob/main/config/onerec/amazon/rqkmeans.gin) | Qwen3-Embedding-8B，`C=3,K=256` centroid |
| [`sft.gin`](https://github.com/phonism/genrec/blob/main/config/onerec/amazon/sft.gin) | 指令 SFT；当前主 recipe 为 full-param、no label smoothing |
| [`clm.gin`](https://github.com/phonism/genrec/blob/main/config/onerec/amazon/clm.gin) | flat token stream 的替代 SFT |
| [`grpo.gin`](https://github.com/phonism/genrec/blob/main/config/onerec/amazon/grpo.gin) | 从 `sft/checkpoint_best` 开始 RL |

GRPO 同时绑定 RL Dataset 和 eval SFT Dataset；两者的 encoder、RQ-KMeans path、C/K 和 protocol 必须逐项相同。

### 其他生成模型

- [`rpg/amazon.gin`](https://github.com/phonism/genrec/blob/main/config/rpg/amazon.gin)：Beauty；
- [`rpg/amazon_sports.gin`](https://github.com/phonism/genrec/blob/main/config/rpg/amazon_sports.gin) 与 [`amazon_toys.gin`](https://github.com/phonism/genrec/blob/main/config/rpg/amazon_toys.gin)：split 专用 codebook/graph/learning-rate 配置，matrix runner 会按 split 路由；
- [`notellm/amazon.gin`](https://github.com/phonism/genrec/blob/main/config/notellm/amazon.gin)：Qwen3 LoRA + exact full catalog；
- [`gram/amazon/gram.gin`](https://github.com/phonism/genrec/blob/main/config/gram/amazon/gram.gin)：预处理 GRAM 文件 + T5 FiD。

## 配置不变量

### tokenizer/recommender 对齐

每个 SID pipeline 至少把以下字段当作一个不可拆分的 artifact identity：

```text
dataset root + split + raw item order
encoder model/revision + text template + embedding cache
tokenizer type + C + K
RQ-VAE geometry / RQ-KMeans centroids / OPQ factory
downstream SID token vocabulary
evaluation protocol（尤其 RPG OPQ）
```

例如 TIGER 的 `train.sem_id_dim=3` 必须等于 `AmazonSeqDataset.rqvae_n_layers=3`；LCRec 的 `train.num_codebooks=5` 必须与 Dataset/RQ-VAE 一致；COBRA 的 `train.id_vocab_size=32` 必须等于 Dataset 的 RQ-VAE K。

### batch 与显存

`batch_size` 与 `split_batches` 的解释由 Accelerate 决定。`gradient_accumulate_every` 只有传给 `setup_accelerator` 并使用 `accelerator.accumulate` 的 trainer 才生效；不要从其他 config 机械复制到 SASRec/TIGER。

`amp=True` 时 `mixed_precision_type` 才生效。COBRA 主配置保留 `mixed_precision_type="bf16"` 但 `amp=False`，因此实际是全精度；RPG 主配置使用 `mixed_precision_type="no"` 且 amp false。

### 实际值优先于函数默认

Python 签名中的 default 只是无绑定回退。示例：RPG 函数默认 lr 是 `3e-4`，但 Beauty Gin 绑定 `0.01`；正式运行采用 Gin 值。文档引用超参时应链接具体 config，而不是复制函数签名默认。

## matrix runner 如何注入配置

[`run_matrix.py`](https://github.com/phonism/genrec/blob/main/scripts/run_matrix.py) 为每个 cell 追加：

```text
<entry_function>.seed=<seed>
<entry_function>.save_dir_root="out/matrix/<protocol>/<model>/<split>/seed<seed>"
<DatasetClass>.evaluation_protocol="<protocol>"
```

OneRec GRPO 还将上游 SFT 路径指向同矩阵树的 `checkpoint_best`。runner 不重写所有跨阶段参数，所以修改主 config 的 C/K/encoder 时必须同步 tokenizer 和上游 cell。

## 常见故障

| 症状 | 首先检查 |
|---|---|
| `No configurable matching ...` | config 是否 import 定义该 class/function 的模块；selector 是否拼错 |
| 路径中还出现 `{split}` | 该字符串是否绕过 `parse_config`；是否在 Python default 内而非 config 文本 |
| checkpoint shape mismatch | RQ-VAE geometry、C/K、HF vocab resize、base revision |
| trie 大量 skipped item | tokenizer 没有对应 `<Cc_code>`；K 或 sem_ids artifact 不一致 |
| 读取到旧 embedding | encoder basename 相同但内容变了；缓存未 force regenerate |
| 本地找 `/root/workspace/models_hub` | `base.gin` macro 未覆盖 |
| 从非根目录运行 include 失败 | 回仓库根目录运行，或使用一致的绝对 config/path |
| 结果 protocol 为 `unknown`/错误 split | 入口未先调用 `parse_config()`，或 CLI 与 Dataset binding 矛盾 |

新增配置的完整步骤见[扩展指南](extension-guide.md)。
