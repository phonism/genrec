# Data routing and dataset injection

## There is no `DatasetFactory`

The repository has no `genrec.data.dataset_factory`, `DatasetFactory`, `DatasetRegistry`, `DatasetManager`, or `ConfigBuilder`. A trainer either constructs a concrete dataset in source code or accepts a dataset class through Gin. This filename remains only for old navigation compatibility.

Direct construction:

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

Gin injection:

```gin
import genrec.data.amazon
train.dataset = @AmazonSeqDataset
AmazonSeqDataset.split = "{split}"
```

The trainer receives a class and calls it with its own explicit keywords. An incompatible constructor raises ordinary `TypeError`; there is no registry-level validation.

## Actual trainer routing

| Trainer | Selection | Expected class / schema |
|---|---|---|
| RQ-VAE | injectable `dataset`, with default | item embedding vector; usually `P5AmazonReviewsItemDataset` or `AmazonItemDataset` |
| TIGER | required Gin injection | `AmazonSeqDataset` returning `SeqData` |
| LCRec | required Gin injection | `AmazonLCRecDataset` task/prompt/response dictionaries |
| COBRA | required Gin injection | `AmazonCobraDataset` sparse IDs plus text tensors |
| OneRec SFT | required Gin injection | `AmazonOneRecSFTDataset` |
| OneRec GRPO | separate `rl_dataset` and `eval_dataset` | RL prompt and SFT-eval dictionaries |
| OneRec CLM | injectable, with fallback | `AmazonOneRecCLMDataset` item-sequence dictionaries |
| SASRec | fixed in trainer | `AmazonSASRecDataset` history/target dictionaries |
| HSTU | `dataset_type` branch | Amazon HSTU or MovieLens-1M HSTU dictionaries |
| RPG | fixed in trainer | `AmazonRPGDataset` item IDs and labels |
| NoteLLM | fixed in trainer | `AmazonNoteLLMDataset` query/target IDs |
| GRAM | injectable, with fallback | `AmazonGRAMDataset` passages and lexical output |

A compatible `__getitem__` is not enough. Trainers also read adapter attributes such as `num_items`, `sem_ids_list`, `sequences`, `item_id2tokens`, `evaluation_protocol`, or `all_items`.

The Amazon 2023 baseline configs do not route SASRec/HSTU through `Amazon2023SASRecDataset`; they use the converted 2014-compatible raw files.

## Example: custom TIGER dataset

```python
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

```gin
import my_project.data
train.dataset = @MyTigerDataset
MyTigerDataset.split = "demo"
```

TIGER reads all three extra attributes shown above to record the protocol and build a training-only SID-to-item resolver. Missing attributes raise `AttributeError`.

## Migration from fictional factory APIs

| Old call | Current approach |
|---|---|
| `DatasetFactory.list_datasets()` | `rg '^class .*Dataset' genrec/data` or [the dataset table](datasets.md) |
| `create_*_dataset(...)` | Import and construct the concrete class |
| `register_dataset(...)` | Import its module in Gin and bind `@ClassName` |
| `create_dataset_from_config(yaml)` | Use the trainer's Gin config and `parse_config()` |

See [Datasets](datasets.md), [sequence contracts](base-dataset.md), [Gin](configs.md), and [Trainers](trainers.md).
