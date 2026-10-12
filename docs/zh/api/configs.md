# Gin 配置与 CLI API

## 配置模型

当前项目使用 [Gin-config](https://github.com/google/gin-config)，没有 `DatasetConfig`、`TextEncodingConfig`、`TrainingConfig` 配置层级，也没有 YAML/JSON 配置 loader。可配置对象直接是带 `@gin.configurable` 的函数或类。

标准执行链：

```text
python trainer.py CONFIG.gin --split SPLIT --gin BINDING ...
  -> parse_config()
  -> 文本替换 {split}
  -> gin.parse_config(config_content)
  -> gin.parse_config(cli_bindings)
  -> 调用 @gin.configurable train 函数
```

## `parse_config`

源文件：[`genrec/modules/utils.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/utils.py)

```python
parse_config() -> argparse.Namespace
```

位置与选项：

```text
config_path             必填 Gin 文件路径
--split SPLIT           默认 beauty
--gin BINDING           可重复
```

行为：

1. 读取整个 `config_path`；
2. 对配置文本做全局字符串替换：`{split}` → `args.split`；
3. 调用 `gin.parse_config`；
4. 对每个 `--gin` binding 做同样的 `{split}` 替换，再解析；
5. 把 `split`、`config_path` 放入进程级 `_RUN_CONTEXT`；
6. 返回 `argparse.Namespace(config_path, split, gin)`。

```python
get_run_split(default: str = "unknown") -> str
```

返回最近一次 `parse_config()` 解析的 split；在未调用时返回 `default`。它是进程全局上下文，不适合同一进程并发运行两份不同配置。

文件不存在会抛 `FileNotFoundError`；Gin 语法、未知 selector、重复/非法 binding 由 Gin 抛异常。函数不捕获这些异常，也不自动验证路径、维度或 checkpoint 兼容性。

## 命令示例

```bash
# 基本训练
python genrec/trainers/sasrec_trainer.py \
  config/sasrec/amazon.gin --split beauty

# 多个 override 必须重复 --gin
python genrec/trainers/hstu_trainer.py \
  config/hstu/amazon.gin --split sports \
  --gin 'train.epochs=100' \
  --gin 'train.use_temporal_bias=False'

# 带列表/字符串的 binding；外层用单引号保护 shell
python genrec/trainers/onerec_sft_trainer.py \
  config/onerec/amazon/sft.gin --split beauty \
  --gin 'AmazonOneRecSFTDataset.enabled_tasks=["sidsft","title2sid"]'
```

`{split}` 是纯文本替换，不是 Python format 安全解析。若普通 prompt/正则文本本来包含同名花括号，也会被替换。

## 配置中的三类绑定

### 函数参数

```gin
train.epochs = 200
train.batch_size = 256
```

selector 必须匹配当前进程已注册的 configurable。很多 trainer 的入口都叫 `train`；配置只能与相应 trainer 脚本一起使用，不能把 `config/tiger/...` 交给 SASRec trainer。

OneRec/GRAM/RPG 使用明确函数名：

```gin
train_sft.learning_rate = 8e-5
train_grpo.num_generations = 8
train_clm.max_length = 256
train_rpg.n_codebook = 32
train_gram.num_beams = 50
fit_rqkmeans.codebook_size = 256
```

### 类参数

```gin
AmazonSeqDataset.split = "{split}"
AmazonSeqDataset.evaluation_protocol = "leave_one_out"
```

只有 trainer 没有显式传同名参数时，类 binding 才决定值。

### 对象引用与宏

```gin
include "config/base.gin"
import genrec.data.amazon

train.dataset = @AmazonSeqDataset
AmazonSeqDataset.encoder_model_name = %MODEL_HUB_SENTENCE_T5_XL
```

- `@Name` 传 configurable/callable 引用；对应模块必须先 import。
- `%NAME` 引用 Gin macro。
- `include` 路径按启动工作目录解析，项目命令假设从仓库根目录运行。

## `config/base.gin`

源文件：[`config/base.gin`](https://github.com/phonism/genrec/blob/main/config/base.gin)

当前定义五个固定路径宏：

```gin
MODEL_HUB_SENTENCE_T5_BASE
MODEL_HUB_SENTENCE_T5_XL
MODEL_HUB_QWEN3_EMBEDDING_8B
MODEL_HUB_QWEN3_1_7B
MODEL_HUB_QWEN3_4B
```

文件中的实际值是 `/root/workspace/models_hub/...` 字符串。`parse_config()` 没有读取 `MODEL_HUB` 环境变量或展开 `${MODEL_HUB}` 的逻辑；换机器时应通过 CLI 重绑 macro 或维护本地配置，而不能只设置环境变量后假定自动生效。

示例：

```bash
--gin 'MODEL_HUB_QWEN3_1_7B="/data/models/Qwen3-1.7B"'
```

## 配置文件索引

| 路径 | 入口 selector | 用途 |
|---|---|---|
| [`config/sasrec/amazon.gin`](https://github.com/phonism/genrec/blob/main/config/sasrec/amazon.gin) | `train` | Amazon 2014 SASRec |
| [`config/sasrec/amazon2023.gin`](https://github.com/phonism/genrec/blob/main/config/sasrec/amazon2023.gin) | `train` | Amazon 2023，经预处理后的 2014-compatible raw schema |
| [`config/hstu/`](https://github.com/phonism/genrec/tree/main/config/hstu) | `train` | CE / sampled-softmax / Amazon 2023 compatible raw schema |
| [`config/tiger/amazon/rqvae.gin`](https://github.com/phonism/genrec/blob/main/config/tiger/amazon/rqvae.gin) | `train` | TIGER 上游 RQVAE |
| [`config/tiger/amazon/tiger.gin`](https://github.com/phonism/genrec/blob/main/config/tiger/amazon/tiger.gin) | `train` | TIGER 推荐器 |
| [`config/lcrec/`](https://github.com/phonism/genrec/tree/main/config/lcrec) | `train` | RQVAE 与 LCRec |
| [`config/cobra/`](https://github.com/phonism/genrec/tree/main/config/cobra) | `train` | RQVAE、sparse-only、mixed variants |
| [`config/onerec/amazon/`](https://github.com/phonism/genrec/tree/main/config/onerec/amazon) | `fit_rqkmeans` / `train_sft` / `train_grpo` / `train_clm` | OneRec 三阶段与 CLM |
| [`config/rpg/`](https://github.com/phonism/genrec/tree/main/config/rpg) | `train_rpg` | Beauty/Sports/Toys 数据集特定配置 |
| [`config/notellm/amazon.gin`](https://github.com/phonism/genrec/blob/main/config/notellm/amazon.gin) | `train` | NoteLLM 对称 GCL + 可选 CSFT |
| [`config/gram/amazon/gram.gin`](https://github.com/phonism/genrec/blob/main/config/gram/amazon/gram.gin) | `train_gram` | GRAM |

!!! note "Amazon 2023 baseline 的实际路由"
    [`sasrec_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/sasrec_trainer.py) 与 [`hstu_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/hstu_trainer.py) 没有构造 `genrec.data.amazon2023.Amazon2023SASRecDataset`，而是继续使用 `AmazonSASRecDataset` / `AmazonHSTUDataset`。[`prepare_amazon2023()`](processors.md#amazon-2023-download-k-core) 会写出 Amazon 2014-compatible gzip；`DATASET_CONFIGS` 当前登记 `books`、`arts_crafts_and_sewing`。配置运行前必须先生成这些 raw 文件，新增 category 还须增加文件名映射。

## Enum binding

[`QuantizeForwardMode`](https://github.com/phonism/genrec/blob/main/genrec/models/rqvae.py) 用 `@gin.constants_from_enum` 注册，可在 RQVAE 配置中写：

```gin
train.vae_codebook_mode = %genrec.models.rqvae.QuantizeForwardMode.GUMBEL_SOFTMAX
train.vae_codebook_last_layer_mode = %genrec.models.rqvae.QuantizeForwardMode.SINKHORN
```

合法值为 `GUMBEL_SOFTMAX`、`STE`、`ROTATION_TRICK`、`SINKHORN`。`QuantizeDistance` 没有同样的 Gin constants decorator。

## 常见错误

1. **`No configurable matching ...`**：缺少 `import genrec.data...`，或把配置交给了错误 trainer。
2. **override 没生效**：selector 写错，例如 `train` 与 `train_sft` 混用；查函数装饰器和配置原绑定。
3. **路径仍指向旧 split**：某个字符串没有 `{split}` 或被硬编码；用 `rg` 检查 config。
4. **列表被 shell 拆开**：外层用单引号包住整个 binding。
5. **相同维度 checkpoint 能加载但语义错误**：Gin 只配置形状，不验证 encoder/codebook provenance。
6. **尝试 `load_config_from_file()`**：不存在该 API；配置只在 CLI `parse_config()` 中解析。
7. **split 与输出目录不一致**：检查解析后的配置与 `results.json`。NoteLLM 已同时绑定 `train.split` 和 `AmazonNoteLLMDataset.split`，标准 `--split sports` 会更新两处。

## 相关文档

- [训练入口](trainers.md)
- [数据类注入](dataset-factory.md)
- [架构：配置](../architecture/configuration.md)
