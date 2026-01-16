# SASRec

Self-Attentive Sequential Recommendation.

## Overview

SASRec (Self-Attentive Sequential Recommendation) is a baseline model that uses self-attention mechanisms to capture user behavior patterns from sequential interaction data.

## Architecture

```
User Sequence: [item_1, item_2, ..., item_n]
       ↓
Item Embeddings
       ↓
Positional Encoding
       ↓
Transformer Encoder (Self-Attention)
       ↓
Next Item Prediction
```

### Key Components

- **Item Embedding**: Learnable embeddings for all items
- **Positional Encoding**: Learnable position embeddings
- **Self-Attention Layers**: Multi-head attention with causal masking
- **Prediction Head**: Dot product with item embeddings

## Configuration

```gin
# config/sasrec/amazon.gin

train.epochs = 200
train.batch_size = 128
train.learning_rate = 1e-3
train.max_seq_len = 50

# Model architecture
train.hidden_dim = 64
train.num_heads = 2
train.num_layers = 2
train.dropout = 0.2
```

## Training

```bash
# Train on Amazon Beauty
python genrec/trainers/sasrec_trainer.py config/sasrec/amazon.gin

# Train on other datasets
python genrec/trainers/sasrec_trainer.py config/sasrec/amazon.gin --split sports
python genrec/trainers/sasrec_trainer.py config/sasrec/amazon.gin --split toys
```

## Evaluation Metrics

- **Recall@K**: Proportion of relevant items in top-K recommendations
- **NDCG@K**: Normalized Discounted Cumulative Gain at K

## Benchmark Results

### Amazon 2014 Beauty

| Model | R@5 | R@10 | N@5 | N@10 |
|-------|-----|------|-----|------|
| SASRec | 0.0469 | 0.0688 | 0.0305 | 0.0375 |

## Model API

```python
from genrec.models import SASRec

model = SASRec(
    num_items=10000,
    hidden_dim=64,
    num_heads=2,
    num_layers=2,
    max_seq_len=50,
    dropout=0.2,
)

# Forward pass
logits = model(item_ids, attention_mask)
```

## Reference

- [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) (ICDM 2018)
