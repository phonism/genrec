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

## Experiment Results

### Setup

- **LLM**: Qwen2.5-1.5B (full fine-tuning, no LoRA)
- **Tokenizer**: RQ-VAE (5 codebooks × 256 codes, sentence-t5-xl encoder)
- **Training**: batch_size=16, lr=2e-5, max_seq_len=50, 10 epochs, cosine schedule
- **Evaluation**: Constrained beam search (beam_width=10), seqrec task
- **Dataset**: Amazon Reviews (5-core)
- **Hardware**: Single A800-80GB GPU

### Amazon Beauty

| Epoch | Valid R@5 | Valid R@10 | Valid N@10 | Test R@5 | Test R@10 | Test N@10 |
|-------|----------|-----------|-----------|---------|----------|----------|
| 0 | 0.0231 | 0.0313 | 0.0172 | 0.0165 | 0.0224 | 0.0123 |
| 1 | 0.0327 | 0.0522 | 0.0268 | 0.0250 | 0.0403 | 0.0204 |
| 2 | 0.0428 | 0.0636 | 0.0339 | 0.0305 | 0.0462 | 0.0247 |
| 3 | 0.0512 | 0.0774 | 0.0422 | 0.0391 | 0.0586 | 0.0323 |
| 4 | 0.0584 | 0.0833 | 0.0479 | 0.0474 | 0.0683 | 0.0390 |
| **5** | **0.0619** | **0.0878** | **0.0510** | **0.0481** | **0.0704** | **0.0403** |
| 6 | 0.0608 | 0.0829 | 0.0490 | 0.0437 | 0.0636 | 0.0364 |
| 7 | 0.0536 | 0.0766 | 0.0453 | 0.0391 | 0.0577 | 0.0330 |
| 8 | 0.0510 | 0.0727 | 0.0426 | 0.0364 | 0.0521 | 0.0301 |
| 9 | 0.0494 | 0.0704 | 0.0415 | 0.0351 | 0.0513 | 0.0292 |

**Best**: Epoch 5 — Valid R@10=0.0878, Test R@10=0.0704

### Amazon Toys

| Epoch | Valid R@5 | Valid R@10 | Valid N@10 | Test R@5 | Test R@10 | Test N@10 |
|-------|----------|-----------|-----------|---------|----------|----------|
| 0 | 0.0081 | 0.0092 | 0.0064 | 0.0073 | 0.0090 | 0.0060 |
| 1 | 0.0267 | 0.0423 | 0.0217 | 0.0243 | 0.0370 | 0.0189 |
| 2 | 0.0387 | 0.0557 | 0.0305 | 0.0286 | 0.0443 | 0.0229 |
| 3 | 0.0462 | 0.0668 | 0.0365 | 0.0379 | 0.0573 | 0.0307 |
| 4 | 0.0518 | 0.0764 | 0.0433 | 0.0410 | 0.0631 | 0.0351 |
| 5 | 0.0577 | 0.0814 | 0.0476 | 0.0448 | **0.0670** | **0.0383** |
| **6** | **0.0566** | **0.0821** | **0.0481** | 0.0433 | 0.0614 | 0.0368 |
| 7 | 0.0528 | 0.0739 | 0.0449 | 0.0392 | 0.0553 | 0.0330 |
| 8 | 0.0498 | 0.0677 | 0.0420 | 0.0367 | 0.0509 | 0.0301 |
| 9 | 0.0478 | 0.0664 | 0.0408 | 0.0363 | 0.0506 | 0.0298 |

**Best**: Epoch 6 — Valid R@10=0.0821, Test R@10=0.0670 (E5)

### Amazon Sports

| Epoch | Valid R@5 | Valid R@10 | Valid N@10 | Test R@5 | Test R@10 | Test N@10 |
|-------|----------|-----------|-----------|---------|----------|----------|
| 0 | 0.0160 | 0.0227 | 0.0130 | 0.0119 | 0.0160 | 0.0094 |
| 1 | 0.0199 | 0.0315 | 0.0167 | 0.0143 | 0.0225 | 0.0120 |
| 2 | 0.0229 | 0.0355 | 0.0189 | 0.0174 | 0.0275 | 0.0145 |
| 3 | 0.0301 | 0.0430 | 0.0242 | 0.0219 | 0.0317 | 0.0177 |
| 4 | 0.0326 | 0.0470 | 0.0259 | 0.0226 | 0.0344 | 0.0187 |
| **5** | **0.0331** | **0.0472** | **0.0265** | **0.0238** | **0.0360** | **0.0198** |
| 6 | 0.0310 | 0.0455 | 0.0254 | 0.0221 | 0.0319 | 0.0177 |
| 7 | 0.0261 | 0.0381 | 0.0214 | 0.0174 | 0.0260 | 0.0144 |
| 8 | 0.0226 | 0.0344 | 0.0191 | 0.0148 | 0.0229 | 0.0125 |
| 9 | 0.0222 | 0.0329 | 0.0184 | 0.0143 | 0.0217 | 0.0118 |

**Best**: Epoch 5 — Valid R@10=0.0472, Test R@10=0.0360

### Amazon Home

(Still training, Epoch 5 eval in progress)

| Epoch | Valid R@5 | Valid R@10 | Valid N@10 | Test R@5 | Test R@10 | Test N@10 |
|-------|----------|-----------|-----------|---------|----------|----------|
| 0 | 0.0062 | 0.0105 | 0.0053 | 0.0047 | 0.0084 | 0.0041 |
| 1 | 0.0094 | 0.0151 | 0.0079 | 0.0082 | 0.0128 | 0.0065 |
| 2 | 0.0137 | 0.0209 | 0.0112 | 0.0102 | 0.0159 | 0.0086 |
| 3 | 0.0168 | 0.0245 | 0.0135 | 0.0137 | 0.0204 | 0.0112 |
| **4** | **0.0194** | **0.0278** | **0.0160** | **0.0163** | **0.0234** | **0.0133** |

**Best so far**: Epoch 4 — Valid R@10=0.0278, Test R@10=0.0234

### Summary (Best Valid Recall@10)

| Dataset | R@5 | R@10 | N@10 | Test R@10 | Test N@10 | Best Epoch |
|---------|-----|------|------|-----------|-----------|------------|
| **Beauty** | 0.0619 | 0.0878 | 0.0510 | 0.0704 | 0.0403 | 5 |
| **Toys** | 0.0566 | 0.0821 | 0.0481 | 0.0670 | 0.0383 | 6 (test best E5) |
| **Sports** | 0.0331 | 0.0472 | 0.0265 | 0.0360 | 0.0198 | 5 |
| **Home** | 0.0194 | 0.0278 | 0.0160 | 0.0234 | 0.0133 | 4 (in progress) |

## Reference

- [LC-Rec: Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation](https://arxiv.org/abs/2311.09049)
