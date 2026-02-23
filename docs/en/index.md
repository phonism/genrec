# GenRec

A Model Zoo for Generative Recommendation.

## Overview

GenRec is a PyTorch-based framework that provides unified implementations of generative recommendation models. It features clean code architecture, Gin-config based experiment management, and support for multiple datasets and models.

## Key Features

- **7 Models**: SASRec, HSTU, RQVAE, TIGER, LCRec, COBRA, and NoteLLM
- **Multiple Datasets**: Amazon 2014 (Beauty, Sports, Toys, Clothing) and Amazon 2023 (32 categories)
- **Gin-Config**: Flexible experiment configuration and parameter override
- **W&B Integration**: Experiment tracking with Weights & Biases
- **Reproducible**: Consistent evaluation with Recall@K and NDCG@K metrics

## Supported Models

| Model | Type | Description |
|-------|------|-------------|
| **SASRec** | Baseline | Self-Attentive Sequential Recommendation |
| **HSTU** | Baseline | Hierarchical Sequential Transduction Unit |
| **RQVAE** | Generative | Residual Quantized VAE for semantic ID generation |
| **TIGER** | Generative | Generative Retrieval with trie-based constrained decoding |
| **LCRec** | Generative | LLM-based recommendation with collaborative semantics |
| **COBRA** | Generative | Cascaded sparse-dense representations |
| **NoteLLM** | Generative | Retrievable LLM for note recommendation (experimental) |

## Quick Start

```bash
# Install
git clone https://github.com/phonism/genrec.git
cd genrec
pip install -e .

# Train SASRec on Amazon Beauty
python genrec/trainers/sasrec_trainer.py config/sasrec/amazon.gin --split beauty

# Train TIGER (requires RQVAE checkpoint)
python genrec/trainers/rqvae_trainer.py config/tiger/amazon/rqvae.gin --split beauty
python genrec/trainers/tiger_trainer.py config/tiger/amazon/tiger.gin --split beauty
```

## Project Structure

```
genrec/
├── genrec/
│   ├── models/          # SASRec, HSTU, RQVAE, TIGER, LCRec, COBRA, NoteLLM
│   ├── trainers/        # Training scripts for each model
│   ├── modules/         # Transformer, embedding, metrics, loss, etc.
│   └── data/            # Amazon 2014 & 2023 dataset implementations
├── config/              # Gin configuration files
│   ├── sasrec/          # SASRec configs
│   ├── hstu/            # HSTU configs
│   ├── tiger/           # TIGER configs (amazon/, amazon2023/)
│   ├── lcrec/           # LCRec configs (amazon/, amazon2023/)
│   └── cobra/           # COBRA configs
├── scripts/             # Utility scripts
└── docs/                # Documentation (English & Chinese)
```

## Benchmark Results

See the [README](https://github.com/phonism/genrec) for full benchmark tables on Amazon 2014 and Amazon 2023 datasets.

## Contributing

We welcome Issues and Pull Requests! Please refer to our [Contributing Guide](contributing.md).

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

## Citation

```bibtex
@software{genrec2025,
  title = {GenRec: A Model Zoo for Generative Recommendation},
  author = {Qi Lu},
  year = {2025},
  url = {https://github.com/phonism/genrec}
}
```
