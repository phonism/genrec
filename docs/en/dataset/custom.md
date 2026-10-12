# Adding a custom dataset

The repository has no `BaseRecommenderDataset`, dataset factory, processor
registry, or dataclass configuration layer. Extend the closest concrete adapter,
reuse the shared split protocol and collate contract, and bind the new class to a
trainer through Gin when that trainer exposes a `dataset` argument.

## Choose the closest integration point

| Goal | Best reference |
|---|---|
| One embedding per item / tokenizer input | `AmazonItemDataset` |
| Item-ID next-item baseline | `AmazonSASRecDataset` |
| Timestamp-aware HSTU | `AmazonHSTUDataset` |
| SID encoder-decoder | `AmazonSeqDataset` |
| LLM prompt, SFT, or RL | `AmazonOneRecSFTDataset`, `AmazonOneRecRLDataset` |
| Parallel OPQ prediction | `AmazonRPGDataset` |
| Dense item-to-item retrieval | `AmazonNoteLLMDataset` |
| Lexical IDs and multiple passages | `AmazonGRAMDataset` |

Do not hide genuine differences in ID bases, padding, text inputs, and sample
augmentation behind a new universal abstraction.

## Minimal raw schema

Normalize the source into two logical tables:

```text
interactions: user_key, item_key, timestamp[, rating/action]
items:        item_key, title[, description, brand, category, ...]
```

Required invariants:

1. `(user, item, timestamp)` parses deterministically.
2. Equal timestamps have a stable tie-break.
3. The item mapping is identical across splits and the embedding catalog.
4. Every validation/test target maps to the catalog.
5. Source revision, filtering, and checksums are recorded.

## Item-ID sequence skeleton

Reuse `select_protocol_sequences` rather than implementing another leave-one-out
split. A SASRec-like adapter reserves 0 for padding and maps real items from 1:

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
        self.item_id_mapping = {}
        user_sequences = defaultdict(list)
        for row in rows:
            if row.item_key not in self.item_id_mapping:
                self.item_id_mapping[row.item_key] = len(self.item_id_mapping) + 1
            item_id = self.item_id_mapping[row.item_key]
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

If the returned schema is identical, reuse `sasrec_collate_fn` and
`sasrec_eval_collate_fn` from `genrec.data.amazon_sasrec`.

## Trainer binding

Trainers such as TIGER, OneRec, and GRAM expose the dataset class as a Gin
parameter:

```gin
import my_package.my_dataset

train.dataset = @MyTigerDataset
MyTigerDataset.root = "dataset/my_data"
MyTigerDataset.split = "category_a"
```

The current SASRec and HSTU trainers instantiate `AmazonSASRecDataset` and
`AmazonHSTUDataset` inside `train`. Supporting another adapter therefore
requires adding a configurable dataset parameter or a structurally equivalent
trainer. Update train, validation, and test construction together.

## Item-tokenizer adapters

RQ-VAE and RQ-KMeans expect a fixed-width float vector from each item. A custom
adapter should expose:

```text
embeddings: np.ndarray [N, D]
dim: int
__len__() -> N
__getitem__(index) -> List[float]
```

Persist the `item_key <-> row_index` mapping with the array. An anonymous
`[N, D]` matrix cannot be safely joined back to user sequences.

## Extra requirements for generative adapters

Keep the real target item even when the model predicts a semantic ID:

```python
from genrec.data.schemas import SeqData

return SeqData(
    user_id=user_id,
    item_ids=flat_history_codes,
    target_ids=target_sid,
    target_item=target_item_id,
)
```

At evaluation time, construct `SemanticIdItemMapper` from **training
interactions only**, resolve each SID to one item, remove duplicate items, and
score against `target_item`. Validation/test target popularity must not enter
the resolver.

## Tests and provenance

At minimum, test:

- identical item mappings across splits;
- padding and ID-base conventions;
- leave-one-out or global-temporal target boundaries;
- catalog coverage of every target;
- collate shapes, masks, and padding direction;
- embedding/SID row alignment;
- full-catalog item-level evaluation.

Useful references are `tests/test_evaluation_protocol.py`,
`tests/test_honest_metrics.py`, and `tests/test_notellm_full_catalog.py`.

Store a manifest with the source URL/revision, raw hashes, filtering rule,
mapping rule, user/item/interaction counts, timestamp unit, and preparation
time. Commit preparation code, schemas, checksums, and a small synthetic fixture;
do not commit restricted raw data.

Before considering the adapter complete, document its Gin binding, command,
output directory, data license, and supported-model status.
