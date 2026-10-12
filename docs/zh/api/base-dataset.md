# 数据集基础契约

## 先澄清：没有 `BaseRecommenderDataset`

当前仓库不存在 `genrec.data.base_dataset`，也没有 `BaseRecommenderDataset`、通用 `ItemDataset` 或通用 `SequenceDataset`。所有可训练数据类直接继承 `torch.utils.data.Dataset`，由具体 trainer 选择相应 collate 函数。

因此自定义数据集的最低契约是 PyTorch 的 duck typing：

```python
class MyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int: ...
    def __getitem__(self, index): ...
```

`__getitem__()` 的返回 schema 必须匹配目标 trainer，而不是匹配一个不存在的基类。现有数据类与 schema 对照见[数据集 API](datasets.md)。

## 共享协议函数 { #shared-protocol-functions }

源文件：[`genrec/data/evaluation_protocol.py`](https://github.com/phonism/genrec/blob/main/genrec/data/evaluation_protocol.py)

```python
TimedInteraction = tuple[int, int]       # (timestamp, item_id)
TimedUserSequence = tuple[str, list[TimedInteraction]]

select_protocol_sequences(
    user_sequences: dict[str, list[TimedInteraction]],
    split: str,
    protocol: str = "leave_one_out",
    min_seq_len: int = 5,
    temporal_train_ratio: float = 0.8,
    temporal_valid_ratio: float = 0.1,
) -> tuple[list[TimedUserSequence], ProtocolInfo]
```

### 输入与返回

- `user_sequences`：未要求预排序；函数内部按 timestamp 升序排列。
- `split`：只能是 `train`、`valid` 或 `test`。
- `protocol`：只能是 `leave_one_out` 或 `global_temporal`。
- 返回的第一项不是最终样本，而是给现有 LOO sample builder 消费的用户序列。
- 返回的第二项为不可变 dataclass：

```python
@dataclass(frozen=True)
class ProtocolInfo:
    protocol: str
    train_cutoff: int = 0
    valid_cutoff: int = 0
    users_before: int = 0
    users_after: int = 0
```

### 协议行为

`leave_one_out` 只过滤长度小于 `min_seq_len` 的用户并返回排序序列；各 dataset 再把最后两项作为 valid/test target。

`global_temporal` 在所有通过长度过滤的 interaction timestamp 上求全局 cutoff。为兼容现有“训练删最后两项、验证删最后一项”的 builder，函数会在返回序列尾部添加随后必被删除的占位 interaction：

- train：cutoff 之前的前缀 + 两个尾部副本；
- valid：历史 + valid target + 一个副本；
- test：历史 + test target。

调用者不应直接把这些占位序列当用户真实行为展示。

### 明确异常

以下情况抛 `ValueError`：

- 未知 `split` 或 `protocol`；
- train/valid ratio 不在 `(0,1)`；
- 两个 ratio 之和大于等于 1；
- 无 timestamp 可求 cutoff；
- cutoff 落在同一个 timestamp 或没有 test-time interaction；
- 指定 split 没有可用用户。

GRAM 与 Amazon 2023 的部分专用 adapter 没有调用该函数；是否支持某协议必须查具体 dataset，不能从本函数存在推断。

## 共享样本类型 { #shared-sample-types }

源文件：[`genrec/data/schemas.py`](https://github.com/phonism/genrec/blob/main/genrec/data/schemas.py)

### `SeqData`

```python
class SeqData(NamedTuple):
    user_id: int
    item_ids: list[int]
    target_ids: list[int]
    target_item: int = -1
```

TIGER 路径使用它：

- `item_ids` 是按 item 展平的 raw semantic codes，长度通常为 `history_items × C`；
- `target_ids` 是一个 target 的 raw codes，长度为 `C`；
- `target_item` 是碰撞感知 item-level 评测所需的原 catalog item ID。

旧 dataset 可省略 `target_item`，此时为 `-1`；当前 TIGER evaluator 遇到负 target item 会抛 `ValueError`，所以它只保证源代码兼容，不保证可正式评测。

### `SeqBatch`

```python
class SeqBatch(NamedTuple):
    user_ids: Tensor
    ids: Tensor
    ids_fut: Tensor
    x: Tensor
    x_fut: Tensor
    seq_mask: Tensor
```

### `TokenizedSeqBatch`

```python
class TokenizedSeqBatch(NamedTuple):
    user_ids: Tensor
    sem_ids: Tensor
    sem_ids_fut: Tensor
    seq_mask: Tensor
    token_type_ids: Tensor
    token_type_ids_fut: Tensor
```

后两个类型只是字段容器，没有构造时的 dtype/shape 验证。`SeqBatch` 仅在旧 P5 管线被实际引用，`TokenizedSeqBatch` 目前只作为 debug metric 的类型约定；新代码不要把它们视为所有 trainer 的统一 batch 类型。

常量 `FUT_SUFFIX = "_fut"` 用于旧 P5 preprocessing 字段名。

## DataLoader 循环工具

源文件：[`genrec/data/utils.py`](https://github.com/phonism/genrec/blob/main/genrec/data/utils.py)

```python
cycle(dataloader)
```

返回一个无限 generator，不断迭代给定 dataloader。RQ-VAE 的 iteration-based trainer 用它。若 dataloader 为空，generator 会持续启动空循环而不 yield；调用前应确保 `len(dataloader) > 0`。

## 自定义数据集的正确接入方式

以 SASRec 为例，返回值应匹配真实 collator：

```python
from torch.utils.data import Dataset
from genrec.data.amazon_sasrec import sasrec_collate_fn

class MySASRecDataset(Dataset):
    def __init__(self, train_test_split="train"):
        self.samples = [{"history": [1, 2], "target": 3}]
        self.num_items = 100

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
```

这段类可以直接配合 `sasrec_collate_fn` 写自定义循环；但当前 `sasrec_trainer.train()` 在函数内部固定构造 `AmazonSASRecDataset`，没有 `dataset` 参数，不能用 `train.dataset=@MySASRecDataset` 注入。要复用现成 trainer，需先给它增加显式 dataset 参数并测试三种 split。支持 Gin 注入的 trainer 见[数据类选择与注入](dataset-factory.md)。

## 不存在的历史 API 迁移表

| 旧文档名 | 当前替代 |
|---|---|
| `BaseRecommenderDataset.load_dataset()` | 具体 dataset 的 `__init__()` 内下载/读取 |
| `get_items()` / `get_interactions()` | 具体类的 `item_*`、`sequences` 等属性；无统一命名 |
| 通用 `ItemDataset` | `AmazonItemDataset`、`Amazon2023ItemDataset`、`P5AmazonReviewsItemDataset` |
| 通用 `SequenceDataset` | 对应模型 adapter，如 `AmazonSASRecDataset`、`AmazonSeqDataset` |
| `CacheManager` | 各 dataset 自己的文件 cache/lock；无共享类 |
| `validate_dataset()` | 专用 tests/scripts 与构造时检查；无共享函数 |

## 相关文档

- [全部数据类与输出 schema](datasets.md)
- [数据类选择与 Gin 注入](dataset-factory.md)
- [预处理实际入口](processors.md)
- [架构：数据流水线](../architecture/data-pipeline.md)
