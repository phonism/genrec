# 数据类选择与注入

## 先澄清：没有 `DatasetFactory`

当前仓库没有 `genrec.data.dataset_factory`、`DatasetFactory`、`DatasetRegistry`、`DatasetManager` 或 `ConfigBuilder`。数据类通过以下两种方式选择：

1. trainer 源码直接构造固定 dataset；
2. `@gin.configurable` trainer 接收 `dataset` 类对象，由 `.gin` 的 `@ClassName` 绑定。

文件名保留用于旧导航兼容，但本页没有可导入的 factory API。

## 直接构造

```python
from genrec.data.amazon_sasrec import AmazonSASRecDataset

train_ds = AmazonSASRecDataset(
    root="dataset/amazon",
    split="beauty",
    train_test_split="train",
    max_seq_len=50,
    evaluation_protocol="leave_one_out",
)
```

`genrec.data.__init__` 只重导出 schema 和 `cycle`，不会自动注册或导出 dataset。应从具体模块导入。

## Gin 类引用

配置必须先 import 类所在模块，之后才能用 `@ClassName`：

```gin
import genrec.data.amazon

train.dataset = @AmazonSeqDataset
AmazonSeqDataset.split = "{split}"
AmazonSeqDataset.encoder_model_name = %MODEL_HUB_SENTENCE_T5_XL
AmazonSeqDataset.evaluation_protocol = "leave_one_out"
```

trainer 收到的是类对象，然后显式调用：

```python
train_dataset = dataset(
    root=dataset_folder,
    train_test_split="train",
    max_seq_len=max_seq_len,
    pretrained_rqvae_path=pretrained_rqvae_path,
)
```

Gin 对类的其他绑定在构造时注入；trainer 显式传入的 keyword 参数优先。没有接口检查，构造参数不兼容时会直接抛 Python `TypeError`。

## Trainer 到 dataset 的真实映射

| Trainer | 选择方式 | 默认/配置类 | 必需样本 schema |
|---|---|---|---|
| RQ-VAE | `train(dataset=...)`，代码有默认 | `P5AmazonReviewsItemDataset`；下游配置可换 `AmazonItemDataset` | 单个 `list[float]` embedding |
| TIGER | Gin 必须注入，代码默认 `None` | `AmazonSeqDataset` | `SeqData` |
| LCRec | Gin 必须注入 | `AmazonLCRecDataset` | task/prompt/response dict |
| COBRA | Gin 必须注入 | `AmazonCobraDataset` | sparse IDs + text token tensors |
| OneRec SFT | Gin 必须注入 | `AmazonOneRecSFTDataset` | task/prompt/response + target SID/item |
| OneRec GRPO | `rl_dataset` 与 `eval_dataset` 分开注入 | `AmazonOneRecRLDataset` / `AmazonOneRecSFTDataset` | RL prompt / SFT eval dict |
| OneRec CLM | `dataset` 可注入；为空时 fallback | `AmazonOneRecCLMDataset` | `item_seq` dict |
| SASRec | trainer 内固定 | `AmazonSASRecDataset` | history/target dict |
| HSTU | `dataset_type` 分支 | `AmazonHSTUDataset` 或 `MovieLens1MHSTUDataset` | history/timestamps/target |
| RPG | trainer 内固定 | `AmazonRPGDataset` | input item IDs / labels |
| NoteLLM | trainer 内固定 | `AmazonNoteLLMDataset` | query/target item IDs |
| GRAM | `train_gram(dataset=...)`，为空时 fallback | `AmazonGRAMDataset` | input passages / lexical output |

“配置文件存在 dataset 绑定”不代表可替换为任意 PyTorch Dataset；collator 依赖具体字段，trainer 还可能读取 `num_items`、`sem_ids_list`、`sequences`、`item_id2tokens` 或 `all_items` 等属性。

当前 SASRec/HSTU 的 Amazon 2023 Gin 没有改变上述固定选择：`Amazon2023SASRecDataset` 并未接入这两个 trainer；它们读取预处理器产出的 2014-compatible raw schema。详见 [Trainer API 的路由说明](trainers.md#discriminative-baseline-signatures)。

## 配置路径索引

| 数据路径 | 示例配置 |
|---|---|
| Amazon 2014 item embedding | [`config/tiger/amazon/rqvae.gin`](https://github.com/phonism/genrec/blob/main/config/tiger/amazon/rqvae.gin) |
| Amazon 2014 RQVAE SID sequence | [`config/tiger/amazon/tiger.gin`](https://github.com/phonism/genrec/blob/main/config/tiger/amazon/tiger.gin) |
| LCRec multitask | [`config/lcrec/amazon/lcrec.gin`](https://github.com/phonism/genrec/blob/main/config/lcrec/amazon/lcrec.gin) |
| OneRec RQ-KMeans/SFT/GRPO | [`config/onerec/amazon/`](https://github.com/phonism/genrec/tree/main/config/onerec/amazon) |
| GRAM 预处理文件 | [`config/gram/amazon/gram.gin`](https://github.com/phonism/genrec/blob/main/config/gram/amazon/gram.gin) |

## 自定义数据类接入

假设为 TIGER 提供一个自定义类：

```python
# my_project/data.py
import gin
from torch.utils.data import Dataset
from genrec.data.schemas import SeqData

@gin.configurable
class MyTigerDataset(Dataset):
    def __init__(self, root, train_test_split, max_seq_len,
                 pretrained_rqvae_path, split="demo"):
        self.evaluation_protocol = "leave_one_out"
        self.sem_ids_list = [[0, 0, 0], [1, 2, 3]]
        self.sequences = [[0, 1, 1]]
        self.rows = [SeqData(0, [0, 0, 0], [1, 2, 3], 1)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]
```

Gin：

```gin
import my_project.data
train.dataset = @MyTigerDataset
MyTigerDataset.split = "demo"
```

除 `__getitem__` 外，TIGER trainer 会读取：

- `evaluation_protocol` 写入 run config；
- `sem_ids_list` 构造 SID→item mapper；
- `sequences` 统计训练前缀 popularity。

缺任一属性都会在训练启动时抛 `AttributeError`。这正是当前 duck-typed 接口，仓库没有 registry 帮你提前校验。

## 如何检查实际绑定

```bash
# 查看配置注入了什么类
rg -n 'dataset\s*=\s*@|rl_dataset\s*=|eval_dataset\s*=' config

# 查看 trainer 构造时显式传了哪些参数/读取哪些属性
rg -n 'dataset\(|_dataset\.|_ds\.' genrec/trainers
```

Gin 不提供本项目专用 `list_datasets()`。可用 `rg '^class .*Dataset' genrec/data` 静态枚举，或参考[数据集 API](datasets.md)。

## 历史 API 迁移

| 旧文档调用 | 当前写法 |
|---|---|
| `DatasetFactory.list_datasets()` | `rg '^class .*Dataset' genrec/data` / [数据集表](datasets.md) |
| `create_base_dataset(name, **kwargs)` | 从具体模块 import 后直接构造 |
| `create_item_dataset(...)` | `AmazonItemDataset(...)` 等具体类 |
| `create_sequence_dataset(...)` | 对应模型 adapter；没有通用 wrapper |
| `register_dataset(...)` | 让 Gin config import 模块并绑定 `@ClassName` |
| `create_dataset_from_config(yaml)` | trainer 的 Gin config + `parse_config()` |

## 相关文档

- [数据集 API](datasets.md)
- [数据集基础契约](base-dataset.md)
- [Gin 与 CLI](configs.md)
- [训练入口](trainers.md)
