# GenRec

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A Model Zoo for Generative Recommendation.

## Benchmark Results

### Evaluation Protocol

Following [TIGER](https://arxiv.org/abs/2305.05065), [LC-Rec](https://arxiv.org/abs/2311.09049), and [OpenOneRec](https://arxiv.org/abs/2512.24762):

- **Dataset**: Amazon 2014 with 5-core filtering (users and items with < 5 interactions removed)
- **Split**: Leave-one-out (last item for test, second-to-last for validation, rest for training)
- **Ranking**: Full-item-set ranking over all items (no negative sampling)
- **Max sequence length**: 50 for all models
- **Metrics**: Recall@K and NDCG@K (K=5, 10)
- **HSTU**: Tested with both full-vocabulary cross-entropy (CE) and sampled softmax (SS, 128 negatives, temp=0.05, L2 norm)

### Amazon 2014 Beauty

| Methods | R@5 | R@10 | N@5 | N@10 |
|---------|-----|------|-----|------|
| SASRec (CE) | 0.0538 | 0.0851 | 0.0320 | 0.0421 |
| SASRec (BCE) | 0.0258 | 0.0503 | 0.0137 | 0.0216 |
| HSTU (CE) | 0.0568 | 0.0859 | 0.0347 | 0.0441 |
| HSTU (SS) | 0.0414 | 0.0727 | 0.0235 | 0.0335 |
| TIGER | 0.0419 | 0.0644 | 0.0282 | 0.0354 |
| LCRec | 0.0481 | 0.0704 | 0.0331 | 0.0403 |
| RPG (sentence-t5-xl) | 0.0525 | 0.0744 | 0.0363 | 0.0433 |
| RPG (text-emb-3-large) | 0.0531 | 0.0780 | 0.0370 | 0.0450 |
| RPG (paper, text-emb-3-large) | 0.0569 | 0.0809 | - | 0.0464 |
| **OneRec-SFT (1.7B)** | **0.0612** | **0.0925** | **0.0400** | **0.0501** |

### Amazon 2014 Sports

| Methods | R@5 | R@10 | N@5 | N@10 |
|---------|-----|------|-----|------|
| SASRec (CE) | 0.0321 | 0.0495 | 0.0191 | 0.0248 |
| SASRec (BCE) | 0.0156 | 0.0291 | 0.0085 | 0.0128 |
| HSTU (CE) | 0.0283 | 0.0439 | 0.0182 | 0.0232 |
| HSTU (SS) | 0.0246 | 0.0393 | 0.0143 | 0.0191 |
| TIGER | 0.0236 | 0.0377 | 0.0150 | 0.0195 |
| LCRec | 0.0238 | 0.0360 | 0.0159 | 0.0198 |
| **OneRec-SFT (1.7B)** | **0.0403** | **0.0596** | **0.0264** | **0.0325** |

### Amazon 2014 Toys

| Methods | R@5 | R@10 | N@5 | N@10 |
|---------|-----|------|-----|------|
| SASRec (CE) | 0.0613 | 0.0922 | 0.0348 | 0.0448 |
| SASRec (BCE) | 0.0353 | 0.0594 | 0.0186 | 0.0264 |
| HSTU (CE) | 0.0611 | 0.0914 | 0.0363 | 0.0461 |
| HSTU (SS) | 0.0494 | 0.0795 | 0.0277 | 0.0375 |
| TIGER | 0.0340 | 0.0521 | 0.0214 | 0.0272 |
| LCRec | 0.0433 | 0.0614 | 0.0310 | 0.0368 |
| **OneRec-SFT (1.7B)** | **0.0637** | **0.0946** | **0.0440** | **0.0541** |

### Amazon 2014 Home

| Methods | R@5 | R@10 | N@5 | N@10 |
|---------|-----|------|-----|------|
| SASRec (CE) | 0.0177 | 0.0277 | 0.0106 | 0.0138 |
| SASRec (BCE) | 0.0081 | 0.0143 | 0.0046 | 0.0066 |
| HSTU (CE) | 0.0129 | 0.0208 | 0.0084 | 0.0109 |
| HSTU (SS) | 0.0123 | 0.0193 | 0.0079 | 0.0102 |
| TIGER | 0.0145 | 0.0231 | 0.0096 | 0.0123 |
| LCRec | 0.0163 | 0.0234 | 0.0110 | 0.0133 |
| **OneRec-SFT (1.7B)** | **0.0238** | **0.0348** | **0.0157** | **0.0193** |

## Features

- **Multiple Models**: Implementations of SASRec, HSTU, RQVAE, TIGER, LCRec, COBRA, RPG, and NoteLLM
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
| **RQVAE** | Generative | Residual Quantized VAE for semantic ID generation |
| **TIGER** | Generative | Generative Retrieval with trie-based constrained decoding |
| **LCRec** | Generative | LLM-based recommendation with collaborative semantics |
| **COBRA** | Generative | Cascaded sparse-dense representations |
| **RPG** | Generative | Parallel generation with OPQ semantic IDs and graph-constrained decoding |
| **NoteLLM** | Generative | Retrievable LLM for note recommendation (experimental) |

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

# For LCRec pipeline
python genrec/trainers/rqvae_trainer.py config/lcrec/amazon/rqvae.gin --split beauty

# For COBRA pipeline
python genrec/trainers/rqvae_trainer.py config/cobra/amazon/rqvae.gin --split beauty
```

### Train TIGER (Generative Retrieval)

```bash
# Requires pretrained RQVAE checkpoint
python genrec/trainers/tiger_trainer.py config/tiger/amazon/tiger.gin --split beauty

# On Amazon 2023
python genrec/trainers/tiger_trainer.py config/tiger/amazon2023/tiger.gin
```

### Train LCRec (LLM-based)

```bash
# Requires pretrained RQVAE checkpoint
python genrec/trainers/lcrec_trainer.py config/lcrec/amazon/lcrec.gin --split beauty

# On Amazon 2023
python genrec/trainers/lcrec_trainer.py config/lcrec/amazon2023/lcrec.gin
```

### Train COBRA

```bash
# Requires pretrained RQVAE checkpoint
python genrec/trainers/cobra_trainer.py config/cobra/amazon/cobra.gin --split beauty
```

### Train RPG (Parallel Generation)

```bash
# RPG on Amazon 2014
python -m genrec.trainers.rpg_trainer config/rpg/amazon.gin --split beauty

# Multi-GPU
bash scripts/train_rpg.sh beauty 4
```

### Train OneRec (RQ-KMeans → SFT)

Reproduces the **OneRec-SFT (1.7B)** rows in the benchmark tables (best R@10 on all four
Amazon 2014 splits). Requires `Qwen3-Embedding-8B` + `Qwen3-1.7B` in `./models_hub/`.

```bash
# Stage 1 — RQ-KMeans quantizer (K=256, 3 codebooks). Item embeddings are generated
# automatically from Qwen3-Embedding-8B on first run.
python genrec/trainers/rqkmeans_trainer.py config/onerec/amazon/rqkmeans.gin --split beauty

# Stage 2 — SFT (Qwen3-1.7B, full-param), 4-GPU. The v2b recipe (no label smoothing) is
# the current best and produces the benchmark numbers above.
accelerate launch --config_file config/accelerate_4gpu.yaml \
    -m genrec.trainers.onerec_sft_trainer config/onerec/amazon/sft.gin --split beauty
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
config/tiger/amazon2023/tiger.gin
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
python genrec/trainers/lcrec_trainer.py config/lcrec/amazon/lcrec.gin \
    --split beauty \
    --gin "MODEL_HUB_QWEN3_1_7B='/path/to/model'"
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
│   ├── tiger/               # TIGER configs (amazon/, amazon2023/)
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
