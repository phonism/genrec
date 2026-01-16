# COBRA

Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations.

## Overview

COBRA is a hybrid generative recommendation model that combines sparse semantic IDs with dense vector representations. It uses an interleaved sparse-dense architecture to capture both discrete and continuous item features.

## Architecture

```
Item Representation = [sparse_id_0, sparse_id_1, ..., sparse_id_C, dense_vector]

Input Sequence:
[s0_1, s1_1, s2_1, d_1, s0_2, s1_2, s2_2, d_2, ..., s0_n, s1_n, s2_n, d_n]
     └── item 1 ──┘        └── item 2 ──┘              └── item n ──┘

       ↓
Transformer Decoder
       ↓
Sparse Head (codebook prediction) + Dense Head (vector reconstruction)
```

### Key Components

- **Sparse Representation**: C codebook tokens from RQ-VAE
- **Dense Representation**: Continuous embedding vector
- **Interleaved Input**: Alternating sparse and dense tokens
- **Dual Prediction Heads**: Separate heads for sparse and dense outputs

## Training Pipeline

### Step 1: Train RQ-VAE

```bash
python genrec/trainers/rqvae_trainer.py config/cobra/amazon/rqvae.gin
```

### Step 2: Train COBRA

```bash
python genrec/trainers/cobra_trainer.py config/cobra/amazon/cobra.gin
```

## Configuration

```gin
# config/cobra/amazon/cobra.gin

# Training
train.epochs = 100
train.batch_size = 32
train.learning_rate = 1e-4

# Model architecture
train.n_codebooks = 3
train.id_vocab_size = 256
train.d_model = 768
train.decoder_n_layers = 2
train.decoder_num_heads = 6

# Loss weights
train.sparse_loss_weight = 1.0
train.dense_loss_weight = 1.0

# Encoder
train.encoder_type = "pretrained"
train.encoder_model_name = %MODEL_HUB_SENTENCE_T5_BASE
```

## Loss Function

COBRA uses a combined loss:

```
L = λ_sparse * L_sparse + λ_dense * L_dense
```

- **L_sparse**: Cross-entropy loss for codebook prediction
- **L_dense**: Cosine similarity loss for vector reconstruction

## Model API

```python
from genrec.models import Cobra

model = Cobra(
    n_codebooks=3,
    id_vocab_size=256,
    d_model=768,
    decoder_n_layers=2,
    decoder_num_heads=6,
    encoder_type="pretrained",
)

# Training forward
output = model(input_ids, encoder_input_ids)
loss = output.loss_sparse + output.loss_dense

# Generation
generated = model.generate(
    input_ids=history_ids,
    encoder_input_ids=history_texts,
    n_candidates=10,
)
```

## Comparison with TIGER

| Aspect | TIGER | COBRA |
|--------|-------|-------|
| Representation | Sparse only | Sparse + Dense |
| Item encoding | Semantic IDs | Semantic IDs + Embeddings |
| Input format | Flat sequence | Interleaved sequence |
| Prediction | Codebook tokens | Codebook tokens + Vector |

## Reference

- COBRA: Sparse Meets Dense - Unified Generative Recommendations with Cascaded Sparse-Dense Representations
