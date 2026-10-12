# Training RQ-VAE

RQ-VAE is the learned item tokenizer used before TIGER, LCRec, and COBRA. A
useful run must match the downstream encoder and geometry and must be evaluated
for codebook health, not only training loss.

## 1. Choose the downstream contract

| Downstream model | Config | Key geometry |
|---|---|---|
| TIGER | `config/tiger/amazon/rqvae.gin` | 3 × 256 codes, latent 32 |
| LCRec | `config/lcrec/amazon/rqvae.gin` | 5 × 256 codes, latent 64 |
| COBRA | `config/cobra/amazon/rqvae.gin` | 3 × 32 codes, latent 32 |

There is no universal checkpoint across these models. Hidden dimensions,
codebook size and depth, latent width, and sentence encoder are part of the
checkpoint contract.

## 2. Prepare data and the encoder

```bash
python scripts/prepare_amazon2014.py \
  --splits beauty --kinds reviews meta

ls /data/models/sentence-t5-xl
```

The first run creates a cache such as:

```text
dataset/amazon/processed/beauty/item_emb_sentence-t5-xl.parquet
```

Override the model-hub Gin macro when `config/base.gin` does not match the local
machine.

## 3. Train

```bash
python genrec/trainers/rqvae_trainer.py \
  config/tiger/amazon/rqvae.gin --split beauty \
  --gin 'MODEL_HUB_SENTENCE_T5_XL="/data/models/sentence-t5-xl"' \
  --gin 'train.wandb_logging=False'
```

The formal configs use epoch mode. The trainer also accepts iteration mode, but
exactly one of `epochs` and `iterations` must be set. The TIGER config currently
uses 5000 epochs, batch size 1024, learning rate `1e-3`, 50 warmup epochs, and
evaluation/checkpointing every 50 epochs.

## 4. First-step K-Means bootstrap

On the first training step, the trainer collects up to roughly 20,000 item
embeddings and initializes each codebook. It should happen once:

- fresh training: `use_kmeans_init=True`;
- resume/inference: `load_pretrained` and `from_pretrained` disable another
  initialization and mark K-Means initialized.

If semantic IDs change on the first forward pass after resume, check that the
model was restored through these loading methods. Coverage lives in
`tests/test_rqvae_checkpoint.py`.

## 5. What to monitor

| Metric | Meaning | Caveat |
|---|---|---|
| `total_loss` | Reconstruction plus quantization loss | Not downstream recommendation quality |
| `reconstruction_loss` | Item-embedding reconstruction | Depends on embedding normalization |
| `rqvae_loss` | Codebook and commitment losses | Interpret with usage statistics |
| `p_unique_ids` | Batch-local unique-SID diagnostic | Not full-catalog collision rate |
| `collision_rate` | Full training-item SID collision during evaluation | Semantic IDs are gathered across ranks before aggregation |
| `emb_avg_norm_i` | Quantized-vector norm at layer `i` | Detects collapsed layers or scale drift |

Training temperature is currently fixed at `0.2`; it is not annealed by epoch.
SID extraction commonly uses `0.001`, while evaluation mode still performs hard
nearest-neighbor assignment.

`checkpoint_best_health.pt` is written only after every layer reaches the
configured minimum usage. Before that, the strongest diagnostic snapshot is
stored as `checkpoint_best_candidate.pt` and must not be consumed downstream.

## 6. Checkpoints and resume

Epoch-mode checkpoints are written as:

```text
out/<downstream>/amazon/<split>/rqvae/checkpoint_epoch_<zero-based-epoch>.pt
```

They include model state, `model_config`, optimizer, scheduler, and epoch or
iteration state. TIGER and COBRA configs currently reference epoch 4999. LCRec's
downstream config references epoch 1999 even though its tokenizer config still
declares 5000 epochs; always follow the explicit downstream path.

Resume optimization state with:

```bash
python genrec/trainers/rqvae_trainer.py \
  config/tiger/amazon/rqvae.gin --split beauty \
  --gin 'train.pretrained_rqvae_path="/abs/path/checkpoint_epoch_2499.pt"'
```

For inference only:

```python
from genrec.models.rqvae import RqVae

model = RqVae.from_pretrained(path, map_location="cpu").eval()
```

## 7. Post-training health checks

```bash
python scripts/verify_codebook_health.py \
  --kind rqvae \
  --checkpoint out/tiger/amazon/beauty/rqvae/checkpoint_epoch_4999.pt \
  --dataset-folder dataset/amazon \
  --split beauty \
  --encoder /data/models/sentence-t5-xl

python scripts/dump_sem_ids.py --help
python scripts/report_sid_collisions.py --help
```

Record per-layer usage, dead codes, maximum frequency, full-SID collision rate,
and collision-group sizes. High per-layer usage does not imply unique complete
SIDs.

Before downstream training, confirm that:

- parquet width equals RQ-VAE `input_dim`;
- hidden dimensions, latent width, codebook size, and layer count match;
- loading does not reinitialize codebooks;
- SID rows align with the downstream item mapping;
- headline evaluation resolves SIDs to items rather than scoring tuple hits.

## Common failures

- **OOM during initialization:** the first step aggregates a large bootstrap
  sample; reduce the bootstrap sample or batch size.
- **Wrong parquet width:** rebuild the cache with the intended encoder; do not
  silently truncate vectors.
- **One code dominates a late layer:** inspect catalog size, K, and residual
  scale. Do not use an 8192-entry codebook for a small catalog.
- **Loss falls while collisions stay high:** reconstruction does not enforce SID
  uniqueness; use explicit health metrics.
- **Trainer and offline collision rates disagree on multiple GPUs:** prefer the
  full-catalog offline report because the trainer does not merge sets across
  ranks.

See [the RQ-VAE model page](../models/rqvae.md) for architecture and equations.
