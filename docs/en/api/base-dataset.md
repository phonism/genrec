# Sequence data contracts

## There is no `BaseRecommenderDataset`

The repository has no `genrec.data.base_dataset`, `BaseRecommenderDataset`, generic `ItemDataset`, or generic `SequenceDataset`. Trainable datasets inherit directly from `torch.utils.data.Dataset`; each trainer selects its own collator.

The minimum contract is therefore ordinary PyTorch duck typing:

```python
class MyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int: ...
    def __getitem__(self, index): ...
```

The returned item must match the target trainer, not a nonexistent base class. See [Datasets](datasets.md) for the concrete schemas.

## Evaluation protocol selector

Source: [`genrec/data/evaluation_protocol.py`](https://github.com/phonism/genrec/blob/main/genrec/data/evaluation_protocol.py)

```python
TimedInteraction = tuple[int, int]  # (timestamp, item_id)
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

`ProtocolInfo` is a frozen dataclass with `protocol`, `train_cutoff`, `valid_cutoff`, `users_before`, and `users_after` fields. Input interactions are sorted by timestamp.

- `leave_one_out` filters users shorter than `min_seq_len`; each adapter later reserves the last two interactions for validation/test.
- `global_temporal` computes catalog-wide timestamp cutoffs. It appends placeholder tail interactions so the existing leave-one-out sample builders remove the correct positions. Do not present those returned sequences as literal user histories.

Unknown split/protocol, invalid ratios, unusable cutoffs, or an empty selected split raise `ValueError`. Not every adapter calls this function; support must be checked per dataset.

## Shared named tuples

Source: [`genrec/data/schemas.py`](https://github.com/phonism/genrec/blob/main/genrec/data/schemas.py)

```python
class SeqData(NamedTuple):
    user_id: int
    item_ids: list[int]
    target_ids: list[int]
    target_item: int = -1

class SeqBatch(NamedTuple):
    user_ids: Tensor
    ids: Tensor
    ids_fut: Tensor
    x: Tensor
    x_fut: Tensor
    seq_mask: Tensor

class TokenizedSeqBatch(NamedTuple):
    user_ids: Tensor
    sem_ids: Tensor
    sem_ids_fut: Tensor
    seq_mask: Tensor
    token_type_ids: Tensor
    token_type_ids_fut: Tensor
```

TIGER uses `SeqData`: `item_ids` is a history flattened over semantic-code positions, `target_ids` contains one raw SID tuple, and `target_item` preserves the catalog item for collision-aware evaluation. A negative `target_item` remains source-compatible with older datasets, but the current evaluator rejects it.

`SeqBatch` belongs to the older P5 path; `TokenizedSeqBatch` is currently a debug-metric type. Neither validates dtype or shape, and neither is a universal trainer batch type. `FUT_SUFFIX = "_fut"` is the legacy P5 field suffix.

## Infinite loader helper

Source: [`genrec/data/utils.py`](https://github.com/phonism/genrec/blob/main/genrec/data/utils.py)

```python
cycle(dataloader)
```

It returns an infinite generator and is used by the iteration-based RQ-VAE trainer. An empty loader never yields, so validate it before calling `next()`.

## Custom adapter example

```python
from torch.utils.data import Dataset

class MySASRecDataset(Dataset):
    def __init__(self):
        self.samples = [{"history": [1, 2], "target": 3}]
        self.num_items = 100

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
```

This schema works with `sasrec_collate_fn` in a custom loop. The current SASRec trainer constructs `AmazonSASRecDataset` internally and cannot inject this class through Gin without a trainer change.

## Historical names and current replacements

| Old documentation | Current reality |
|---|---|
| `BaseRecommenderDataset.load_dataset()` | Concrete dataset initialization performs loading |
| `get_items()` / `get_interactions()` | Adapter-specific attributes; no common names |
| generic `ItemDataset` / `SequenceDataset` | Model-specific classes listed in [Datasets](datasets.md) |
| `CacheManager` | Per-adapter files, locks, and cache keys |
| `validate_dataset()` | Constructor checks plus model-specific tests/scripts |

See [data routing](dataset-factory.md), [processing](processors.md), and the [data pipeline](../architecture/data-pipeline.md).
