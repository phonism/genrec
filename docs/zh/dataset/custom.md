# 添加自定义数据集

当前仓库没有 `BaseRecommenderDataset`、`DatasetFactory` 或 dataclass 配置系统。添加数据集的正确方式是：选一个任务 schema 最接近的现有 adapter，复用协议与 collate 约定，再通过 Gin 把 trainer 的 `dataset` 参数绑定到新类。

## 先选择接入层

| 目标 | 推荐参考 |
|---|---|
| 单 item embedding / tokenizer | `AmazonItemDataset` |
| item-ID next-item baseline | `AmazonSASRecDataset` |
| 带 timestamp HSTU | `AmazonHSTUDataset` |
| SID encoder–decoder | `AmazonSeqDataset` |
| LLM prompt/SFT/RL | `AmazonOneRecSFTDataset`, `AmazonOneRecRLDataset` |
| parallel OPQ | `AmazonRPGDataset` |
| dense item-to-item retrieval | `AmazonNoteLLMDataset` |
| lexical ID + multi-passage | `AmazonGRAMDataset` |

不要先造一个抽象基类再强迫所有模型适配。模型之间的 ID base、padding、文本输入和训练样本增强确实不同，隐藏这些差异会让评测静默错位。

## 最小原始 schema

推荐先规范为两张逻辑表：

```text
interactions: user_key, item_key, timestamp[, rating/action]
items:        item_key, title[, description, brand, category, ...]
```

硬要求：

1. `(user,item,timestamp)` 可稳定解析；
2. 同一用户按 timestamp 排序时有确定 tie-break；
3. item mapping 在 train/valid/test 与 embedding catalog 间一致；
4. 所有测试 target 都能映射到 catalog；
5. 数据版本、过滤与 checksum 可记录。

## 示例：item-ID 序列 adapter

下面只展示接口骨架；切分逻辑复用仓库函数，而不是自己再实现一套 LOO：

```python
from collections import defaultdict
from torch.utils.data import Dataset

from genrec.data.evaluation_protocol import select_protocol_sequences


class MySASRecDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        train_test_split="train",
        max_seq_len=50,
        min_seq_len=5,
        evaluation_protocol="leave_one_out",
        temporal_train_ratio=0.8,
        temporal_valid_ratio=0.1,
    ):
        rows = load_my_interactions(root, split)

        # SASRec/HSTU 约定：0 padding，真实 item 从 1 开始。
        self.item_id_mapping = {}
        user_sequences = defaultdict(list)
        for row_index, row in enumerate(rows):
            if row.item_key not in self.item_id_mapping:
                self.item_id_mapping[row.item_key] = len(self.item_id_mapping) + 1
            item_id = self.item_id_mapping[row.item_key]
            # 如 timestamp 可能相同，应在 load 阶段加入稳定 tie-break。
            user_sequences[row.user_key].append((row.timestamp, item_id))

        selected, self.protocol_info = select_protocol_sequences(
            user_sequences,
            train_test_split,
            protocol=evaluation_protocol,
            min_seq_len=min_seq_len,
            temporal_train_ratio=temporal_train_ratio,
            temporal_valid_ratio=temporal_valid_ratio,
        )
        self.evaluation_protocol = evaluation_protocol
        self.num_items = len(self.item_id_mapping)
        self.sequences = [[item for _, item in seq] for _, seq in selected]
        self.samples = build_samples_like_amazon_sasrec(
            self.sequences, train_test_split, max_seq_len
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]  # {"history": [...], "target": int}
```

如果返回 schema 完全相同，可直接复用：

```python
from genrec.data.amazon_sasrec import (
    sasrec_collate_fn,
    sasrec_eval_collate_fn,
)
```

## 绑定到 trainer

有些 trainer（TIGER、OneRec、GRAM）把 dataset class 暴露成 Gin 参数，可直接绑定：

```gin
import my_package.my_dataset

train.dataset = @MyTigerDataset
MyTigerDataset.root = "dataset/my_data"
MyTigerDataset.split = "category_a"
```

SASRec/HSTU trainer 当前在函数内部直接写死 `AmazonSASRecDataset` / `AmazonHSTUDataset`，若要接新数据集，需要把 `dataset` class 变成可配置参数，或新增同结构 trainer。修改时同时覆盖 train、valid、test 三处构造，不要只替换 train。

## item tokenizer adapter

RQ-VAE/RQ-KMeans 需要 `__getitem__` 返回一条固定维度 float vector。自定义实现至少暴露：

```text
embeddings: np.ndarray [N,D]
dim: int
__len__ -> N
__getitem__(idx) -> List[float]
```

同时保存稳定的 `item_key ↔ row_index` mapping。只保存 `[N,D]` 数组而丢掉 mapping，会使 SID 无法安全回接用户序列。

## 生成式 adapter 的额外约束

SID 模型必须保留真实 item target。推荐沿用 `SeqData`：

```python
from genrec.data.schemas import SeqData

return SeqData(
    user_id=user_id,
    item_ids=flat_history_codes,
    target_ids=target_sid,
    target_item=target_item_id,
)
```

评测时用 `SemanticIdItemMapper` 从**训练交互**构建 SID→item resolver。不要用 valid/test target popularity，也不要因为 adapter 已经返回 SID 就删除 `target_item`。

## 协议测试

至少添加以下测试：

```python
def test_item_mapping_is_identical_across_splits(): ...
def test_zero_is_reserved_only_when_adapter_requires_padding(): ...
def test_valid_and_test_targets_follow_protocol(): ...
def test_global_temporal_has_no_future_history(): ...
def test_every_target_exists_in_catalog(): ...
def test_collate_shapes_and_padding_direction(): ...
def test_sid_rows_align_with_item_ids(): ...
```

参考现有：

- `tests/test_evaluation_protocol.py`；
- `tests/test_honest_metrics.py`；
- `tests/test_notellm_full_catalog.py`。

## 数据 provenance

为每次数据准备写 manifest，至少包含：

```json
{
  "source": "provider / URL / dataset revision",
  "raw_sha256": {"interactions": "...", "items": "..."},
  "filter": "iterative 5-core",
  "mapping": "first-seen after deterministic input ordering",
  "users": 0,
  "items": 0,
  "interactions": 0,
  "timestamp_unit": "seconds",
  "prepared_at": "ISO-8601"
}
```

不要提交有许可证限制的原始数据；提交准备脚本、schema、checksum 和小型合成 fixture。

## 完成定义

- [ ] train/valid/test 都能构造且统计合理；
- [ ] item mapping 与 embedding/SID catalog 同源；
- [ ] LOO 与 global temporal 至少一个有测试；
- [ ] collate 的 padding、mask、target shape 有测试；
- [ ] full-catalog item-level 评测可运行；
- [ ] Gin config、命令和输出目录已记录；
- [ ] 数据来源、许可与 preprocessing manifest 已写；
- [ ] 模型页的“支持数据集”状态已更新。

完整代码接入顺序见[扩展指南](../architecture/extension-guide.md)。
