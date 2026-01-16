# LCRec

Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation.

## Overview

LCRec adapts large language models (LLMs) for recommendation by integrating collaborative semantics through codebook tokens. It uses semantic IDs from RQ-VAE to represent items and fine-tunes an LLM to generate item recommendations.

## Architecture

```
Item Text → RQ-VAE → Semantic IDs [<C0_x>, <C1_y>, ..., <C4_z>]
                          ↓
User History + Prompt → LLM (Qwen) → Generated Semantic IDs
                          ↓
                    Constrained Decoding
                          ↓
                    Recommended Items
```

### Key Components

- **Semantic ID Generation**: RQ-VAE with 5 codebooks (256 codes each)
- **Codebook Tokens**: Special tokens `<Ci_j>` added to LLM vocabulary
- **Constrained Decoding**: Beam search with prefix constraints
- **Multi-task Training**: seqrec, item2index, index2item tasks

## Training Pipeline

### Step 1: Train RQ-VAE for Semantic IDs

```bash
python genrec/trainers/rqvae_trainer.py config/lcrec/amazon/rqvae.gin --split beauty
```

### Step 2: Fine-tune LLM

```bash
python genrec/trainers/lcrec_trainer.py config/lcrec/amazon/lcrec.gin --split beauty
```

### Debug Mode (Quick Testing)

```bash
python genrec/trainers/lcrec_trainer.py config/lcrec/amazon/lcrec_debug.gin
```

## Configuration

```gin
# config/lcrec/amazon/lcrec.gin

# Training
train.epochs = 4
train.batch_size = 32
train.learning_rate = 2e-5
train.max_length = 512

# Model
train.pretrained_path = %MODEL_HUB_QWEN3_1_7B
train.use_lora = False

# Codebook
train.num_codebooks = 5
train.codebook_size = 256

# Evaluation
train.eval_beam_width = 10
```

## Tasks

LCRec supports three training tasks:

| Task | Input | Output |
|------|-------|--------|
| **seqrec** | User history | Next item semantic ID |
| **item2index** | Item text | Item semantic ID |
| **index2item** | Semantic ID | Item text |

## Model API

```python
from genrec.models import LCRec

model = LCRec(pretrained_path="Qwen/Qwen2.5-1.5B")
model.add_codebook_tokens(num_codebooks=5, codebook_size=256)

# Generate recommendations
outputs = model.generate(
    input_ids=prompt_ids,
    max_new_tokens=6,
    num_beams=10,
)
```

## Reference

- [LC-Rec: Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation](https://arxiv.org/abs/2311.09049)
