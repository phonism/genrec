# Amazon 2014 and the P5 compatibility layer

Older documentation called the main data path “P5 Amazon.” The current code has
two distinct implementations: model-specific Amazon 2014 adapters used by
benchmarks, and a legacy PyTorch Geometric/P5 pipeline in `p5_amazon.py`.

## Current path: Amazon Reviews 2014 5-core

Expected raw files are:

```text
dataset/amazon/raw/<split>/reviews_<Category>_5.json.gz
dataset/amazon/raw/<split>/meta_<Category>.json.gz
```

The current `DATASET_CONFIGS` entries are:

| `--split` | Review file | Metadata file |
|---|---|---|
| `beauty` | `reviews_Beauty_5.json.gz` | `meta_Beauty.json.gz` |
| `sports` | `reviews_Sports_and_Outdoors_5.json.gz` | `meta_Sports_and_Outdoors.json.gz` |
| `toys` | `reviews_Toys_and_Games_5.json.gz` | `meta_Toys_and_Games.json.gz` |
| `clothing` | `reviews_Clothing_Shoes_and_Jewelry_5.json.gz` | `meta_Clothing_Shoes_and_Jewelry.json.gz` |
| `home` | `reviews_Home_and_Kitchen_5.json.gz` | `meta_Home_and_Kitchen.json.gz` |

The `books` and `arts_crafts_and_sewing` entries read compatible files produced
by the Amazon 2023 converter; they are not Stanford 2014 categories with the
same names.

## Fields used by current adapters

| Source | Field | Use |
|---|---|---|
| Review | `asin` | Item key |
| Review | `reviewerID` | User key |
| Review | `unixReviewTime` | Sequence ordering and temporal splits |
| Metadata | `asin` | Alignment with the item mapping |
| Metadata | `title`, `price`, `salesRank`, `brand`, `categories` | Text passed to `AmazonItemDataset`'s encoder |

Ratings are not supervision targets for current next-item adapters. Items that
lack usable metadata can disappear from the embedding table, so always check
embedding rows against the sequence mapping.

## Download and validate

```bash
python scripts/prepare_amazon2014.py \
  --splits beauty sports toys \
  --kinds reviews meta
```

The script serializes concurrent downloads, writes temporary files, reads to
gzip EOF to validate the stream, and records SHA-256 hashes. `AmazonItemDataset`
can also download missing files, but the explicit preparation script gives
clearer provenance on shared machines.

## Text embeddings

`AmazonItemDataset` formats available metadata into a text record containing
title, price, sales rank, brand, and categories. SentenceTransformer encodes the
records in batches and writes parquet rows with:

| Column | Meaning |
|---|---|
| `ItemID` | Integer ID in the current mapping |
| `embedding` | Vector represented as a Python list |

The encoder basename is part of the cache name, for example
`item_emb_sentence-t5-xl.parquet`. TIGER, LCRec, COBRA, and RPG commonly use
Sentence-T5; OneRec's RQ-KMeans path uses Qwen3-Embedding-8B.

## User sequences are adapter-specific

All adapters rebuild mappings from reviews and sort by timestamp, but their
training samples differ:

- SASRec and HSTU keep one training sequence per user and supervise multiple
  shifted positions in the collate step;
- TIGER and LCRec commonly build sliding prefixes;
- OneRec creates natural-language/SID tasks;
- RPG uses 1-based IDs and right padding;
- NoteLLM creates adjacent item-to-item retrieval pairs.

Do not pass integer sequences serialized by one adapter to another adapter.

## Legacy `p5_amazon.py`

The classes that actually exist are:

| Class | Purpose |
|---|---|
| `AmazonReviews` | PyG `InMemoryDataset` that builds a heterogeneous user-item graph and histories |
| `P5AmazonReviewsItemDataset` | Reads item embeddings from the PyG processed artifact |
| `P5AmazonReviewsSeqDataset` | Loads RQ-VAE and converts item histories into `SeqData` SIDs |

Names such as `P5AmazonItemDataset` and `P5AmazonSequenceDataset` do not exist in
this repository. The legacy module imports `torch_geometric`, has its own
processed artifacts and historical defaults, and is not bound by current Gin
configs. Use it only when reproducing an old checkpoint, with its data version
recorded explicitly.

## Minimal examples

Item tokenizer input:

```python
from genrec.data.amazon import AmazonItemDataset

items = AmazonItemDataset(
    root="dataset/amazon",
    split="beauty",
    train_test_split="all",
    encoder_model_name="/data/models/sentence-t5-xl",
)
print(len(items), items.dim)
```

SASRec evaluation input:

```python
from genrec.data.amazon_sasrec import AmazonSASRecDataset

test = AmazonSASRecDataset(
    root="dataset/amazon",
    split="beauty",
    train_test_split="test",
    max_seq_len=50,
)
print(len(test), test.num_items, test[0])
```

Construction may read large raw files or encode metadata; these snippets assume
that caches have already been prepared.

Data sources retain their own licenses. The repository's MIT license does not
relicense Amazon data or pretrained models. See the
[Amazon Product Data page](https://jmcauley.ucsd.edu/data/amazon/) and the
[P5 paper](https://arxiv.org/abs/2203.13366).
