# HSTU

Hierarchical Sequential Transduction Unit.

## Overview

HSTU (Hierarchical Sequential Transduction Unit) is an enhanced sequential recommendation model that improves upon SASRec with three key innovations:

1. **SiLU Attention**: Replaces softmax with SiLU activation to preserve preference intensity
2. **Update Gate**: Adds element-wise gating for hierarchical feature interaction
3. **Temporal Bias**: Optional log-bucket temporal position bias

## Architecture

```
User Sequence: [item_1, item_2, ..., item_n] + [timestamp_1, ..., timestamp_n]
       ↓
Item Embeddings + Positional Encoding
       ↓
HSTU Layers (SiLU Attention + Update Gate + Temporal RAB)
       ↓
RMS Normalization
       ↓
Next Item Prediction
```

### Key Differences from SASRec

| Component | SASRec | HSTU |
|-----------|--------|------|
| Attention | softmax(QK^T/√d) | SiLU(QK^T + RAB) |
| Output | Attention @ V | Norm(Attention @ V) ⊙ U |
| Temporal | No | Log-bucket time bias |
| Matrices | Q, K, V (3) | Q, K, V, U (4) |

### SiLU vs Softmax

- **Softmax**: Normalizes attention weights, loses absolute preference intensity
- **SiLU**: Non-normalized activation, preserves preference magnitude (44.7% gap in synthetic tests)

## Configuration

```gin
# config/hstu/amazon.gin

train.epochs = 200
train.batch_size = 128
train.learning_rate = 1e-3
train.max_seq_len = 50

# Model architecture
train.hidden_dim = 64
train.num_heads = 2
train.num_layers = 2
train.dropout = 0.2
train.use_temporal_bias = True  # Enable temporal RAB
```

## Training

```bash
# Train with temporal bias (default)
python genrec/trainers/hstu_trainer.py config/hstu/amazon.gin

# Train without temporal bias (similar to SASRec)
python genrec/trainers/hstu_trainer.py config/hstu/amazon.gin \
    --gin "train.use_temporal_bias=False"

# Train on other datasets
python genrec/trainers/hstu_trainer.py config/hstu/amazon.gin --split sports
```

## Benchmark Results

### Amazon 2014 Beauty

| Model | R@5 | R@10 | N@5 | N@10 |
|-------|-----|------|-----|------|
| SASRec | 0.0469 | 0.0688 | 0.0305 | 0.0375 |
| HSTU | 0.0486 | 0.0708 | 0.0340 | 0.0412 |

## Model API

```python
from genrec.models import HSTU

model = HSTU(
    num_items=10000,
    hidden_dim=64,
    num_heads=2,
    num_layers=2,
    max_seq_len=50,
    dropout=0.2,
    use_temporal_bias=True,
)

# Forward pass with timestamps
logits = model(item_ids, timestamps, attention_mask)
```

## Reference

- [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152) (ICML 2024)
