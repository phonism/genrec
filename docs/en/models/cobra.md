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
train.epochs = 20
train.batch_size = 32
train.learning_rate = 1e-4

# Model architecture
train.n_codebooks = 3
train.id_vocab_size = 32      # 32x32x32 codebook, per the paper
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

## Reproduction Status

**Implemented but not yet reproduced.** On Amazon 2014 Beauty (leave-one-out, full eval set, `checkpoint_final.pt` = epoch 19, n_beam=32):

| Metric | R@5 | R@10 | N@5 | N@10 |
|--------|-----|------|-----|------|
| Honest item-level (test) | 0.0054 | **0.0099** | 0.0033 | 0.0048 |
| Honest item-level (valid) | 0.0060 | 0.0101 | 0.0036 | 0.0049 |
| Sem-ID level (test, inflated) | 0.0071 | 0.0127 | 0.0045 | 0.0063 |
| **Paper** | 0.0537 | **0.0725** | 0.0395 | 0.0456 |

About 7x below the paper. Three bugs/issues were found and fixed, then the *remaining* gap was diagnosed:

1. **Eval was collision-inflated (fixed).** `cobra_trainer.py` compared the target's *semantic-ID tuple* against retrieved items' sem-ids. With a 32×32×32 codebook and no uniquifying code, collision rate is 0.43 → any item in the target's bucket counted as a hit (~1.3x inflation; also hid BeamFusion's dense disambiguation). Eval now compares **item ids**; sem-id is kept as a logged reference. `beam_fusion` was vectorized (the per-(batch,beam) `.cpu()` sync loop dominated eval time) and `generate()`'s encoder call was chunked (OOM fix).

2. **Encoder/RQ-VAE mismatch (fixed — this one silently breaks any fresh run).** The trainer passed the *model's* encoder name (`sentence-t5-base`) to the dataset, overriding the gin binding that says the dataset should embed items with `sentence-t5-xl` — the encoder the RQ-VAE was trained on. Result: items embedded in the wrong space → semantic IDs collapse from **6901 → 529** unique. Fix: don't override the dataset encoder (base/xl share the T5 tokenizer, so the model still sees the same text).

3. **Decode beam too narrow (fixed).** `n_beam` 20 → 32 (the codebook size): +13% item R@10.

**Remaining gap = model ceiling, not the quantizer or decoding.** The RQ-VAE codebooks are healthy (perplexity ≈28-32/32) and the sparse branch *does* learn under teacher forcing (per-codebook 0.26/0.21/0.43; the eval per-codebook numbers measure *retrieved-item* codes, not the model's). The bottleneck is that the model's intrinsic ability on the held-out item is low and a deeper/longer-trained decoder is needed to lift it — a pilot with a 4-layer decoder raised the teacher-forced ceiling steadily (epoch-9 train recall already > the 2-layer baseline's epoch-19) but item-level recall only moves once that ceiling is high enough. Closing to the paper is open-ended training work (the paper does not publish its hyperparameters). `generate()` also still lacks a KV-cache, so eval is slow.

Tools: `scripts/eval_cobra_itemlevel.py` (item-level + sem-id), `scripts/diag_cobra_{loss,gen,final}.py` (teacher-forced vs generation diagnostics).

## Reference

- COBRA: Sparse Meets Dense - Unified Generative Recommendations with Cascaded Sparse-Dense Representations
