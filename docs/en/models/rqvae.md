# RQ-VAE

RQ-VAE is an **item tokenizer** in GenRec. It compresses continuous metadata embeddings into multi-level semantic IDs (SIDs) for TIGER, LCRec, and COBRA. It does not consume user histories and must not be reported as a recommender with Recall@K.

| Field | Current repository |
|---|---|
| Input | item semantic embedding `[B,input_dim]` |
| Output | reconstruction, per-layer code/vector/residual, quantization loss |
| Downstream | TIGER, LCRec, COBRA with incompatible geometries |
| Diagnostics | reconstruction, quantization, usage/perplexity, full-SID collision |
| Status | training and checkpoint recovery implemented |

## Residual quantization

An MLP first maps an item vector to a latent `z`. Each codebook approximates the current residual:

\[
z=E(x),\quad r_1=z,
\]

\[
k_\ell=\arg\min_k d(r_\ell,e_{\ell,k}),\quad
q_\ell=e_{\ell,k_\ell},\quad
r_{\ell+1}=r_\ell-q_\ell.
\]

The SID is `(k_1,...,k_C)`, and reconstruction is

\[
\hat x=D\!\left(\sum_{\ell=1}^{C}q_\ell\right).
\]

Codebook position is part of the identity: code 7 in layer 0 is not the same token as code 7 in layer 1.

The loss combines reconstruction and each layer's codebook/commitment loss:

\[
\mathcal L_q^{(\ell)}=
\lVert\operatorname{sg}(r_\ell)-q_\ell\rVert_2^2+
\beta\lVert r_\ell-\operatorname{sg}(q_\ell)\rVert_2^2.
\]

Current embedding pipelines set `n_cat_features=0`; a positive value treats trailing decoder dimensions as BCE-with-logits targets.

## Quantization modes and outputs

| Mode | Training behavior |
|---|---|
| `GUMBEL_SOFTMAX` | soft mixture from temperature-scaled noisy negative distances |
| `STE` | hard nearest code with straight-through gradient |
| `ROTATION_TRICK` | hard nearest code with rotation-trick encoder gradient |
| `SINKHORN` | balanced batch assignment followed by hard STE |

Distance is L2 in current RQ-VAE construction. `codebook_normalize` affects the encoder and first codebook path only; `sim_vq` adds a shared projection after codebook lookup.

```python
out = rqvae.get_semantic_ids(x)
# out.embeddings:   [B,D,C]
# out.residuals:    [B,D,C]
# out.sem_ids:      [B,C]
# out.quantize_loss:[B]
```

`forward(batch, gumbel_t)` returns scalar total/reconstruction/RQ losses plus diagnostic embedding norms and the batch tuple-uniqueness ratio. Batch uniqueness is not a substitute for full-catalog collision analysis.

## Downstream geometries

These checkpoints are not interchangeable:

| Downstream | input | latent | hidden dimensions | K | layers |
|---|---:|---:|---|---:|---:|
| TIGER | 768 | 32 | 512,256,128,64 | 256 | 3 |
| LCRec | 768 | 64 | 512,256,128 | 256 | 5 |
| COBRA | 768 | 32 | 512,256,128,64 | 32 | 3 |

All current variants use Sentence-T5-XL item embeddings, STE in earlier layers, and Sinkhorn in the last layer. The downstream dataset's `rqvae_*` parameters must match checkpoint geometry and the same item order/encoder provenance.

## Data, training, and loading

`AmazonItemDataset` builds an item mapping from review order, encodes metadata, and caches a split/encoder-specific parquet file. RQ-VAE randomly reserves 5% of item embeddings for reconstruction monitoring; that split is unrelated to user-sequence validation/test.

```bash
python genrec/trainers/rqvae_trainer.py config/tiger/amazon/rqvae.gin --split beauty
python genrec/trainers/rqvae_trainer.py config/lcrec/amazon/rqvae.gin --split beauty
python genrec/trainers/rqvae_trainer.py config/cobra/amazon/rqvae.gin --split beauty
```

Current downstream configs train for 5,000 epochs with 50 warmup epochs and batch 1,024. Initial K-means bootstrap accumulates up to roughly 20,000 examples. A recovered checkpoint must not run bootstrap again.

```python
from genrec.models.rqvae import RqVae

rqvae = RqVae.from_pretrained(checkpoint, map_location="cpu")
rqvae.eval()
sem_ids = rqvae.get_semantic_ids(item_embeddings).sem_ids
```

The checkpoint stores epoch/iteration, model, `model_config`, optimizer, and scheduler. `from_pretrained` reconstructs geometry and disables reinitialization; `load_pretrained` loads into an explicitly constructed model and does the same. There is no `load_from_checkpoint` or `generate_semantic_ids` method.

## Collisions and health

Track layer usage, dead codes, complete-tuple collision rate, and maximum collision group separately. Perfect usage at each layer can coexist with many duplicate full SIDs. Downstream headline metrics must resolve predictions to item IDs and deduplicate them.

`get_semantic_ids_dedup` freezes the first `C-1` codes and Sinkhorn-reassigns colliding last-layer groups. It can still leave or introduce collisions and only warns; it is not a uniqueness guarantee.

```bash
python scripts/verify_codebook_health.py \
  --kind rqvae \
  --checkpoint out/tiger/amazon/beauty/rqvae/checkpoint_epoch_4999.pt \
  --dataset-folder dataset/amazon --split beauty \
  --encoder ./models_hub/sentence-t5-xl
```

Useful source maps: [`rqvae.py`](https://github.com/phonism/genrec/blob/main/genrec/models/rqvae.py), [`loss.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/loss.py), [`amazon.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon.py), [`rqvae_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/rqvae_trainer.py), and [`verify_codebook_health.py`](https://github.com/phonism/genrec/blob/main/scripts/verify_codebook_health.py).

Known limits include staged rather than end-to-end objectives, batch-level rather than catalog-global Sinkhorn balance, `torch.compile` portability, and severe high-K collapse risk on small catalogs. Lower collision does not guarantee higher recommendation quality; report both tokenizer health and item-level metrics.

See the exact [RQ-VAE API](../api/rqvae.md) and the [generative-recommendation taxonomy](../landscape/taxonomy.md).
