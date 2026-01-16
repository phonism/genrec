# genrec

A PyTorch-based generative recommender systems research framework.

## Overview

genrec is a modular framework for recommender systems research, implementing state-of-the-art generative recommendation algorithms. It provides clean code architecture, flexible configuration systems, and extensible data processing pipelines.

## Key Features

- ✨ **Modular Design**: Clean component separation for easy understanding and extension
- 🔧 **Configuration-Driven**: Flexible configuration system based on Gin-Config
- 📊 **Multiple Models**: Latest generative recommendation models like RQVAE and TIGER
- 🎯 **Dataset Support**: Popular recommendation datasets like P5 Amazon
- 🚀 **Distributed Training**: Multi-GPU training support with Accelerate
- 📈 **Experiment Tracking**: Weights & Biases integration for experiment management
- 🔍 **Cache Optimization**: Smart data preprocessing caching mechanisms

## Supported Models

### RQVAE (Residual Quantized Variational Autoencoder)
- Vector quantized variational autoencoder for recommendations
- Multiple quantization strategies: Gumbel-Softmax, STE, Rotation Trick, Sinkhorn
- Used for learning semantic item representations

### TIGER (Recommender Systems with Generative Retrieval)
- Transformer-based generative retrieval model
- Sequential modeling using semantic IDs
- Trie-constrained generation process

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Train RQVAE

```bash
python genrec/trainers/rqvae_trainer.py config/rqvae/p5_amazon.gin
```

### Train TIGER

```bash
python genrec/trainers/tiger_trainer.py config/tiger/p5_amazon.gin
```

## Project Structure

```
genrec/
├── genrec/          # Core code
│   ├── data/                        # Data processing modules
│   │   ├── configs.py               # Configuration classes
│   │   ├── base_dataset.py          # Abstract dataset classes
│   │   ├── p5_amazon.py             # P5 Amazon dataset
│   │   ├── processors/              # Data processors
│   │   └── dataset_factory.py       # Dataset factory
│   ├── models/                      # Model implementations
│   │   ├── rqvae.py                 # RQVAE model
│   │   └── tiger.py                 # TIGER model
│   ├── modules/                     # Base modules
│   │   ├── embedding.py             # Embedding layers
│   │   ├── encoder.py               # Encoders
│   │   ├── loss.py                  # Loss functions
│   │   └── metrics.py               # Evaluation metrics
│   └── trainers/                    # Training scripts
│       ├── rqvae_trainer.py         # RQVAE trainer
│       └── tiger_trainer.py         # TIGER trainer
├── config/                          # Configuration files
│   ├── rqvae/                       # RQVAE configs
│   └── tiger/                       # TIGER configs
└── docs/                           # Documentation
```

## Key Improvements

Compared to the original implementation, our refactored version provides:

1. **Cleaner Code Structure**: Modular design with clear responsibilities
2. **Configuration Management**: Support for flexible parameter configuration and experiment management
3. **Enhanced Generalizability**: Easy to extend to new datasets and models
4. **Performance Optimization**: Caching mechanisms and improved memory efficiency
5. **Better Documentation**: Complete API documentation and usage examples

## Benchmark Results

| Dataset | Model | Metric | Result |
|---------|-------|--------|--------|
| P5 Amazon-Beauty | TIGER | Recall@10 | 0.42 |

## Contributing

We welcome Issues and Pull Requests! Please refer to our [Contributing Guide](contributing.md).

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

## Citation

If you use this framework in your research, please cite the relevant papers:

```bibtex
@inproceedings{rqvae2023,
  title={RQ-VAE Recommender},
  author={Botta, Edoardo},
  year={2023}
}

@article{tiger2023,
  title={TIGER: Recommender Systems with Generative Retrieval},
  year={2023}
}
```