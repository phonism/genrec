# Dataset overview

GenRec does not have a universal dataset base class. Each model adapter implements
`torch.utils.data.Dataset` directly. The shared pieces are the raw Amazon files,
evaluation-protocol helpers, and a small set of schemas. Start from the concrete
dataset class constructed by a trainer.

## Data sources

| Source | Entry point | Main use | Status |
|---|---|---|---|
| Amazon Reviews 2014 5-core | `genrec/data/amazon.py`, `amazon_<model>.py` | Main Beauty/Sports/Toys/Home/Clothing benchmarks | Core |
| Amazon Reviews 2023 | `genrec/data/amazon2023.py` | Download JSONL, iterative k-core, compatible gzip export | Preprocessing and selected configs |
| MovieLens 1M | `genrec/data/ml1m_hstu.py` | Small HSTU parity and ablation runs | HSTU only |
| GRAM-formatted Amazon | `genrec/data/amazon_gram.py` | Lexical IDs, item text, CF-neighbor passages | Separate pipeline |
| Legacy P5/PyG pipeline | `genrec/data/p5_amazon.py` | Older graph-backed item/sequence experiments | Compatibility only |

## Model-to-adapter map

| Model or stage | Dataset | ID convention | Important output |
|---|---|---|---|
| SASRec | `AmazonSASRecDataset` | Real items start at 1; 0 pads | History and target |
| HSTU | `AmazonHSTUDataset` | Real items start at 1; 0 pads | History, timestamps, target |
| RQ-VAE / RQ-KMeans | `AmazonItemDataset` | Parquet row aligned with sequence mapping | Continuous item embedding |
| TIGER | `AmazonSeqDataset` | Item IDs are 0-based; SIDs are encoded separately | `SeqData` and raw `target_item` |
| LCRec | `AmazonLCRecDataset` | 0-based | Multi-task prompts/responses and SID catalog |
| OneRec SFT / GRPO | `AmazonOneRecSFTDataset`, `AmazonOneRecRLDataset` | 0-based | Four SFT tasks or RL prompts |
| OneRec CLM | `AmazonOneRecCLMDataset` | 0-based | One flat token stream per user |
| COBRA | `AmazonCobraDataset` | 0-based | Sparse codes plus dense text tokens |
| RPG | `AmazonRPGDataset` | Real items start at 1; 0 pads | Item sequence and OPQ tokens |
| NoteLLM | `AmazonNoteLLMDataset` | Inherits SASRec's 1-based IDs | Item-to-item query/target and text |
| GRAM | `AmazonGRAMDataset` | Raw string ID plus lexical ID | User passage and per-item passages |

!!! danger "Do not reuse integer item IDs across adapters"
    The same ASIN may be 0-based in one adapter and 1-based in another, and
    mappings depend on source order. A checkpoint, SID array, and prediction
    dump must use the mapping that created them.

## Amazon 2014 layout

```text
dataset/amazon/
├── raw/beauty/
│   ├── reviews_Beauty_5.json.gz
│   └── meta_Beauty.json.gz
└── processed/beauty/
    ├── item_emb_sentence-t5-xl.parquet
    └── item_emb_Qwen3-Embedding-8B.parquet
```

Prepare the explicitly supported Beauty, Sports, and Toys downloads with:

```bash
python scripts/prepare_amazon2014.py \
  --splits beauty sports toys --kinds reviews meta
```

Other entries in `DATASET_CONFIGS` can be prepared manually using the expected
file names, or downloaded through `AmazonItemDataset`. Preserve a checksum
manifest for reproducible runs.

## Amazon 2023 conversion

```bash
python -m genrec.data.amazon2023 --list_categories
python -m genrec.data.amazon2023 \
  --category Arts_Crafts_and_Sewing \
  --k_core 5 \
  --output_dir dataset/amazon2023
```

The converter downloads review and metadata JSONL, iteratively filters users and
items until both satisfy k-core, then writes gzip files compatible with the main
Amazon adapters. `--max_reviews` is for smoke tests only; never compare a
truncated run with a full-data benchmark.

## Evaluation protocols

`genrec.data.evaluation_protocol.select_protocol_sequences` implements:

- `leave_one_out`: the last item is test and the penultimate item is validation;
- `global_temporal`: global timestamp quantiles define train/validation cutoffs,
  so later interactions cannot enter an earlier history.

Adapters then turn selected sequences into sliding windows or one sample per
user. Sharing a split protocol does not imply identical training augmentation.
The global-temporal path is still transductive at the catalog level: metadata,
k-core filtering, and SID construction may observe the complete item set.

## Embedding caches

`AmazonItemDataset` formats metadata text, encodes it with SentenceTransformer,
and stores a parquet cache. Generation uses an exclusive `.lock`, a
`.part.<pid>` temporary file, and atomic `os.replace`. RPG has its own
embedding/OPQ caches, while OneRec uses Qwen3-Embedding-8B. Equal vector
dimensions do not make caches interchangeable.

## Minimum data checks

1. Record raw-file checksums and counts.
2. Verify user/item counts and the k-core condition.
3. Check target timestamp boundaries for every split.
4. Match adapter item mappings to embedding/SID row counts.
5. Record train/validation/test sample counts.
6. Measure SID collisions and codebook health.
7. Record encoder name, revision, dimension, and cache hash.

Useful commands:

```bash
python scripts/verify_dataset.py --skip_rqvae
python scripts/prepare_item_embeddings.py --help
```

`scripts/diagnose_dataset.py` and `scripts/check_data_leakage.py` contain
historical hard-coded paths and are not portable general entry points.

See also [Amazon 2014 and the P5 compatibility layer](p5-amazon.md) and
[adding a custom dataset](custom.md).
