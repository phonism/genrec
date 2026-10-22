# GenRec

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

[English](README.md) | [简体中文](README.zh-CN.md)

A Model Zoo for Generative Recommendation.

## Benchmark Results

The Amazon-2014 item-level matrix is complete: **162 cells, 54 three-seed rows**
across Beauty/Sports/Toys and leave-one-out/global-temporal protocols.
[BENCHMARKS.md](BENCHMARKS.md) reports mean ± population standard deviation,
recipe boundaries and provenance; [per-seed CSV](results/matrix_20261022.csv)
contains exact source artifact paths and checksums.

- All headline metrics are item-level; SID models use training-only resolution
  and item deduplication. Catalog/metadata filtering remains transductive.
- Tables below are leave-one-out means over seeds 42/43/44. Sequence history
  is 50 except LCRec's paper recipe (20); these are not length-matched runs.
  Temporal rows and standard deviations are in BENCHMARKS.md.
- OneRec uses 3×256 codes. Its 4-code Beauty ablation is reported separately.
- OneRec + GRPO is validation-selected, including the SFT baseline: **16/18
  cells fall back to SFT**. Random evaluation prompt templates also cause
  baseline drift, so small table differences are not isolated RL gains.
- NoteLLM is a GCL+CSFT I2I adaptation that excludes the query item. Completed
  repository runs do not imply reproduction of every original paper setting.
- Archived TIGER/COBRA paper-alignment audits, old SID metrics and legacy runs
  are kept separate in BENCHMARKS.md.

### Amazon 2014 Beauty

| Method | R@5 | R@10 | N@5 | N@10 |
| --- | --- | --- | --- | --- |
| SASRec (CE) | 0.0542 | 0.0858 | 0.0317 | 0.0420 |
| HSTU (CE) | 0.0461 | 0.0727 | 0.0306 | 0.0392 |
| TIGER (matrix recipe) | 0.0361 | 0.0568 | 0.0234 | 0.0301 |
| LCRec (4B, seq20, gb128) | 0.0509 | 0.0729 | 0.0360 | 0.0431 |
| COBRA (optimized light) | 0.0290 | 0.0454 | 0.0187 | 0.0240 |
| RPG (direct) | 0.0506 | 0.0732 | 0.0354 | 0.0427 |
| NoteLLM (GCL+CSFT, I2I) | 0.0502 | 0.0693 | 0.0362 | 0.0423 |
| OneRec SFT (3 SID) | 0.0433 | 0.0658 | 0.0291 | 0.0364 |
| OneRec + GRPO (selected) | 0.0437 | 0.0661 | 0.0293 | 0.0365 |

### Amazon 2014 Sports

| Method | R@5 | R@10 | N@5 | N@10 |
| --- | --- | --- | --- | --- |
| SASRec (CE) | 0.0313 | 0.0497 | 0.0182 | 0.0241 |
| HSTU (CE) | 0.0234 | 0.0384 | 0.0152 | 0.0200 |
| TIGER (matrix recipe) | 0.0207 | 0.0342 | 0.0131 | 0.0174 |
| LCRec (4B, seq20, gb128) | 0.0260 | 0.0399 | 0.0178 | 0.0223 |
| COBRA (optimized light) | 0.0179 | 0.0290 | 0.0117 | 0.0153 |
| RPG (direct) | 0.0290 | 0.0426 | 0.0200 | 0.0244 |
| NoteLLM (GCL+CSFT, I2I) | 0.0237 | 0.0351 | 0.0161 | 0.0197 |
| OneRec SFT (3 SID) | 0.0222 | 0.0341 | 0.0142 | 0.0180 |
| OneRec + GRPO (selected) | 0.0220 | 0.0341 | 0.0141 | 0.0180 |

### Amazon 2014 Toys

| Method | R@5 | R@10 | N@5 | N@10 |
| --- | --- | --- | --- | --- |
| SASRec (CE) | 0.0610 | 0.0917 | 0.0342 | 0.0441 |
| HSTU (CE) | 0.0438 | 0.0686 | 0.0287 | 0.0366 |
| TIGER (matrix recipe) | 0.0288 | 0.0477 | 0.0181 | 0.0242 |
| LCRec (4B, seq20, gb128) | 0.0504 | 0.0705 | 0.0356 | 0.0420 |
| COBRA (optimized light) | 0.0252 | 0.0412 | 0.0151 | 0.0202 |
| RPG (direct) | 0.0569 | 0.0820 | 0.0391 | 0.0472 |
| NoteLLM (GCL+CSFT, I2I) | 0.0674 | 0.0917 | 0.0460 | 0.0539 |
| OneRec SFT (3 SID) | 0.0386 | 0.0591 | 0.0256 | 0.0321 |
| OneRec + GRPO (selected) | 0.0393 | 0.0604 | 0.0255 | 0.0323 |

Supplemental Beauty LOO: HSTU corrected Meta-local sampled softmax R@10 0.0754 ± 0.0021; OneRec 4-code suffix ablation R@10 0.0727 ± 0.0013.

### Amazon 2014 Home (legacy snapshot; outside this matrix)

A separate **54-cell Home extension** (same nine pipelines, two protocols and
three seeds) is being prepared with the [Home preset](config/matrix/amazon2014_home_20261022.json).
The values below remain historical until the new item-level runs are complete.
RPG transfers the Beauty recipe without Home-specific test tuning.

| Methods | R@5 | R@10 | N@5 | N@10 |
|---------|-----|------|-----|------|
| SASRec (CE) | 0.0177 | 0.0277 | 0.0106 | 0.0138 |
| SASRec (BCE) | 0.0081 | 0.0143 | 0.0046 | 0.0066 |
| HSTU (CE) | 0.0129 | 0.0208 | 0.0084 | 0.0109 |
| HSTU (SS) | 0.0123 | 0.0193 | 0.0079 | 0.0102 |
| TIGER † | - | - | - | - |
| LCRec † | - | - | - | - |

† Home TIGER/LCRec remain outside the verified snapshot; no legacy SID headline is used.

## Features

- **Multiple Models**: 9 recommenders: SASRec, HSTU, TIGER, LCRec, COBRA, RPG, NoteLLM, OneRec, and GRAM; RQ-VAE/RQ-KMeans tokenizer components
- **Multiple Datasets**: Amazon 2014 (Beauty, Sports, Toys, Clothing) and Amazon 2023 (32 categories)
- **Modular Design**: Clean separation of models, data, and training logic
- **Flexible Configuration**: Gin-config based experiment management
- **Easy Extension**: Add custom datasets and models with minimal code
- **Reproducible**: Consistent evaluation metrics (Recall@K, NDCG@K) with W&B logging

## Models

| Model | Type | Description |
|-------|------|-------------|
| **SASRec** | Baseline | Self-Attentive Sequential Recommendation |
| **HSTU** | Baseline | Hierarchical Sequential Transduction Unit with temporal bias |
| **RQVAE / RQ-KMeans** | Tokenizer | Item embeddings to semantic IDs; not standalone recommenders |
| **TIGER** | Generative | Generative Retrieval with trie-based constrained decoding |
| **LCRec** | Generative | LLM-based recommendation with collaborative semantics |
| **COBRA** | Generative | Cascaded sparse-dense representations |
| **RPG** | Generative | Parallel generation with OPQ semantic IDs and graph-constrained decoding |
| **NoteLLM** | LLM retrieval | GCL+CSFT item-to-item adaptation |
| **OneRec** | Generative | Qwen + semantic IDs; SFT/GRPO and alternative CLM |
| **GRAM** | Generative | Lexical IDs with multi-granularity fusion (outside the matrix) |

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/phonism/genrec.git
cd genrec
pip install -e .
```

### Full Installation (with Triton, TorchRec, etc.)

```bash
pip install -e ".[full]"
```

### Dependencies Only

```bash
pip install -r requirements.txt
```

## Quick Start

### Reproduce the published matrix

Use the dated preset for the tables above. Generic configs elsewhere in this
repository include older recipes; they are not interchangeable with this snapshot.
The preset maps each CSV recipe to its Gin file, disables W&B login, separates
protocol/seed outputs, and makes GRPO load the matching SFT checkpoint.
Run commands from the repository root with one GPU available per cell. On shared
hosts, follow the project's GPU lock/scheduling rules before executing commands.

For the full matrix, install the RPG and LoRA dependencies: `pip install -e ".[rpg]" "peft>=0.10.0"`.

Populate `models_hub/` with `sentence-t5-xl`, `Qwen3-Embedding-8B`,
`Qwen3-1.7B`, and `Qwen3-4B` as needed (see `config/base.gin`). Prepare
**one seed-42 tokenizer per domain**, shared by recommender seeds 42/43/44:

```bash
# Use a clean committed checkout; this archive records the code used by new runs.
mkdir -p out
GENREC_REPRO_REV=$(git rev-parse HEAD)
export GENREC_SOURCE_BUNDLE_NAME="genrec-source-${GENREC_REPRO_REV}.tar.gz"
git archive --format=tar.gz --output="out/${GENREC_SOURCE_BUNDLE_NAME}" HEAD
export GENREC_SOURCE_BUNDLE_SHA256="$(python -c 'import hashlib,os,pathlib; print(hashlib.sha256(pathlib.Path("out", os.environ["GENREC_SOURCE_BUNDLE_NAME"]).read_bytes()).hexdigest())')"

for split in beauty sports toys; do
  python -m genrec.trainers.rqvae_trainer config/tiger/amazon/rqvae.gin \
    --split "$split" --gin 'train.seed=42' --gin 'train.wandb_logging=False'
  python -m genrec.trainers.rqvae_trainer config/lcrec/amazon/rqvae_paper_recipe.gin \
    --split "$split" --gin 'train.seed=42'
  python -m genrec.trainers.rqvae_trainer config/cobra/amazon/rqvae.gin \
    --split "$split" --gin 'train.seed=42' --gin 'train.wandb_logging=False'
  python -m genrec.trainers.rqkmeans_trainer config/onerec/amazon/rqkmeans.gin \
    --split "$split" --gin 'fit_rqkmeans.seed=42'
done

# 81 cells per protocol. Review first; remove --dry-run to train serially.
for protocol in leave_one_out global_temporal; do
  python scripts/run_matrix.py --preset amazon2014_20261022 \
    --models sasrec hstu tiger lcrec cobra rpg notellm onerec_sft onerec_grpo \
    --splits beauty sports toys --seeds 42 43 44 \
    --evaluation-protocol "$protocol" \
    --save-dir-root 'out/reproduce_20261022/{protocol}/{model}/{split}/seed{seed}' \
    --dry-run
done

# Check published numbers without training dependencies or raw artifacts.
python scripts/check_benchmarks.py
```

LCRec uses `rqvae_paper_recipe/checkpoint_best_health.pt`; TIGER and COBRA
use their respective `rqvae/checkpoint_epoch_4999.pt`; OneRec uses
`rqkmeans/rqkmeans.pt`. The runner checks these before starting each dependent
cell. RPG prepares OPQ in its dataset adapter. The preset targets the main
3-code OneRec rows; 4-code and sampled-softmax ablations remain separate.

This reproduces the documented recipes, not identical scores across hardware
or regenerated tokenizers. Checkpoints, cached embeddings, and raw result files
are not bundled with Git; the CSV retains original result paths and SHA256s.
OneRec/LCRec evaluation prompt sampling and the historical mixed source versions
are documented in [BENCHMARKS.md](BENCHMARKS.md).

### Train Baseline Models

```bash
# SASRec on Amazon 2014
python genrec/trainers/sasrec_trainer.py config/sasrec/amazon.gin --split beauty

# HSTU on Amazon 2014
python genrec/trainers/hstu_trainer.py config/hstu/amazon.gin --split beauty

# SASRec on Amazon 2023
python genrec/trainers/sasrec_trainer.py config/sasrec/amazon2023.gin

# HSTU on Amazon 2023
python genrec/trainers/hstu_trainer.py config/hstu/amazon2023.gin
```

### Train RQVAE (Semantic ID Generator)

```bash
# For TIGER pipeline
python genrec/trainers/rqvae_trainer.py config/tiger/amazon/rqvae.gin --split beauty

# For the matrix LCRec pipeline (four codebooks, USM)
python genrec/trainers/rqvae_trainer.py config/lcrec/amazon/rqvae_paper_recipe.gin --split beauty

# For COBRA pipeline
python genrec/trainers/rqvae_trainer.py config/cobra/amazon/rqvae.gin --split beauty
```

### Train TIGER (Generative Retrieval)

```bash
# Requires pretrained RQVAE checkpoint
python genrec/trainers/tiger_trainer.py config/tiger/amazon/tiger.gin --split beauty
```

### Train LCRec (LLM-based)

```bash
# Matrix recipe: Qwen3-4B, history 20, effective batch 128
python genrec/trainers/lcrec_trainer.py config/lcrec/amazon/lcrec_paper_recipe.gin --split beauty

# On Amazon 2023
python genrec/trainers/lcrec_trainer.py config/lcrec/amazon2023/lcrec.gin
```

### Train COBRA

```bash
# Requires pretrained RQVAE checkpoint
python genrec/trainers/cobra_trainer.py config/cobra/amazon/cobra_optimized.gin --split beauty
```

### Train RPG (Parallel Generation)

```bash
# RPG on Amazon 2014
python -m genrec.trainers.rpg_trainer config/rpg/amazon.gin --split beauty

# Multi-GPU
bash scripts/train_rpg.sh beauty 4
```

### Train NoteLLM

```bash
python -m genrec.trainers.notellm_trainer config/notellm/amazon.gin --split beauty
```

### Train OneRec (RQ-KMeans → SFT → GRPO)

Trains the main **OneRec SFT (3 SID)** recipe and its validation-selected GRPO stage.
Requires `Qwen3-Embedding-8B` + `Qwen3-1.7B` in `./models_hub/`.

```bash
# Stage 1 — RQ-KMeans quantizer (K=256, 3 codebooks). Item embeddings are generated
# automatically from Qwen3-Embedding-8B on first run.
python genrec/trainers/rqkmeans_trainer.py config/onerec/amazon/rqkmeans.gin --split beauty

# Stage 2 — SFT (Qwen3-1.7B, full-param), one GPU.
python -m genrec.trainers.onerec_sft_trainer config/onerec/amazon/sft.gin --split beauty

# Stage 3 — GRPO; standalone defaults load the SFT checkpoint_best above.
bash scripts/train_onerec_grpo.sh beauty 1
```

> **Recipe** (`sft.gin`): `lr=8e-5, weight_decay=0.05,
> attention_dropout=0.1, label_smoothing=0.0, early_stopping_patience=2`. Use K=256 (not
> 8192) on small catalogs — larger codebooks collapse on shallow residuals.

## Configuration

### Dataset Selection

```bash
# Amazon 2014 datasets (via --split)
--split beauty    # Beauty
--split sports    # Sports and Outdoors
--split toys      # Toys and Games
--split clothing  # Clothing, Shoes and Jewelry

# Amazon 2023 datasets use dedicated config files
config/sasrec/amazon2023.gin
config/hstu/amazon2023.gin
config/lcrec/amazon2023/lcrec.gin
```

### Parameter Override

```bash
--gin "param=value"
```

### Examples

```bash
# Change epochs and batch size
python genrec/trainers/tiger_trainer.py config/tiger/amazon/tiger.gin \
    --split beauty \
    --gin "train.epochs=200" \
    --gin "train.batch_size=128"

# Custom model path for LCRec
python genrec/trainers/lcrec_trainer.py config/lcrec/amazon/lcrec_paper_recipe.gin \
    --split beauty \
    --gin "MODEL_HUB_QWEN3_4B='/path/to/model'"
```

## Project Structure

```
genrec/
├── genrec/
│   ├── models/          # Model implementations
│   │   ├── sasrec.py        # SASRec
│   │   ├── hstu.py          # HSTU
│   │   ├── rqvae.py         # RQVAE
│   │   ├── tiger.py         # TIGER
│   │   ├── lcrec.py         # LCRec
│   │   ├── cobra.py         # COBRA
│   │   ├── rpg.py           # RPG
│   │   └── notellm.py       # NoteLLM
│   ├── trainers/        # Training scripts
│   │   ├── sasrec_trainer.py
│   │   ├── hstu_trainer.py
│   │   ├── rqvae_trainer.py
│   │   ├── tiger_trainer.py
│   │   ├── lcrec_trainer.py
│   │   ├── cobra_trainer.py
│   │   ├── rpg_trainer.py
│   │   └── trainer_utils.py
│   ├── modules/         # Reusable components
│   │   ├── transformer.py   # Transformer blocks
│   │   ├── embedding.py     # Embedding layers
│   │   ├── encoder.py       # Encoder modules
│   │   ├── metrics.py       # Recall@K, NDCG@K
│   │   ├── loss.py          # Loss functions
│   │   ├── scheduler.py     # LR schedulers
│   │   ├── kmeans.py        # K-means for RQVAE init
│   │   ├── gumbel.py        # Gumbel softmax
│   │   └── normalize.py     # Normalization layers
│   └── data/            # Dataset implementations
│       ├── amazon.py        # Amazon 2014 datasets
│       ├── amazon2023.py    # Amazon 2023 datasets (32 categories)
│       ├── amazon_sasrec.py # SASRec-specific data
│       ├── amazon_hstu.py   # HSTU-specific data
│       ├── amazon_lcrec.py  # LCRec-specific data
│       ├── amazon_cobra.py  # COBRA-specific data
│       ├── amazon_rpg.py    # RPG-specific data (OPQ tokenization)
│       └── p5_amazon.py     # P5-format data
├── config/              # Gin configuration files
│   ├── base.gin             # Base config
│   ├── sasrec/              # SASRec configs
│   ├── hstu/                # HSTU configs
│   ├── tiger/               # TIGER configs (amazon/)
│   ├── lcrec/               # LCRec configs (amazon/, amazon2023/)
│   ├── cobra/               # COBRA configs
│   └── rpg/                 # RPG configs
├── scripts/             # Utility scripts
├── docs/                # Documentation (English & Chinese)
├── assets/              # Media assets
└── reference/           # Reference implementations
```

## Documentation

Full documentation is available at [https://phonism.github.io/genrec](https://phonism.github.io/genrec)

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/en/contributing.md) for details.

## Citation

If you find this project useful, please cite:

```bibtex
@software{genrec2025,
  title = {GenRec: A Model Zoo for Generative Recommendation},
  author = {Qi Lu},
  year = {2025},
  url = {https://github.com/phonism/genrec}
}
```

## References

- [SASRec](https://arxiv.org/abs/1808.09781): Self-Attentive Sequential Recommendation
- [HSTU](https://arxiv.org/abs/2402.17152): Hierarchical Sequential Transduction Units
- [TIGER](https://arxiv.org/abs/2305.05065): Recommender Systems with Generative Retrieval
- [RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender) by Edoardo Botta
- [LC-Rec](https://arxiv.org/abs/2311.09049): LLM-based Collaborative Recommendation
- [COBRA](https://arxiv.org/abs/2503.02453): Cascaded Sparse-Dense Representations
- [RPG](https://arxiv.org/abs/2506.05781): Recommendation with Parallel Generation ([official code](https://github.com/facebookresearch/RPG_KDD2025))
- [NoteLLM](https://arxiv.org/abs/2403.01744): A Retrievable LLM for Note Recommendation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
