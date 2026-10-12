# RQ-VAE API

This page documents [`genrec/models/rqvae.py`](https://github.com/phonism/genrec/blob/main/genrec/models/rqvae.py). RQ-VAE maps item embeddings to semantic IDs; it does not produce recommendation lists by itself. See [the model page](../models/rqvae.md) for the algorithm and paper mapping.

## Imports and enums

```python
from genrec.models.rqvae import (
    RqVae, RqVaeOutput, RqVaeComputedLosses,
    Quantize, QuantizeOutput,
    QuantizeForwardMode, QuantizeDistance,
)
```

`QuantizeForwardMode` has `GUMBEL_SOFTMAX`, `STE`, `ROTATION_TRICK`, and `SINKHORN`; it is registered as Gin constants. `QuantizeDistance` has `L2` and `COSINE`, but is not registered as Gin constants. `RqVae` currently builds all layers with L2 distance.

## One quantization layer

```python
Quantize(
    embed_dim: int,
    n_embed: int,
    do_kmeans_init: bool = True,
    codebook_normalize: bool = False,
    sim_vq: bool = False,
    commitment_weight: float = 0.25,
    forward_mode=QuantizeForwardMode.GUMBEL_SOFTMAX,
    distance_mode=QuantizeDistance.L2,
)

quantizer.distances(x: Tensor[B,E]) -> Tensor[B,K]
quantizer.get_item_embeddings(item_ids) -> Tensor[...,E]
quantizer(x, temperature) -> QuantizeOutput

class QuantizeOutput(NamedTuple):
    embeddings: Tensor  # [B,E]
    ids: Tensor         # [B]
    loss: Tensor        # [B]
```

`x.shape[-1]` must equal `embed_dim`. With K-means initialization enabled, the first forward pass initializes from that batch; fewer than `n_embed` examples fail during no-replacement sampling.

In training, Gumbel produces a soft codebook mixture, STE/rotation use a nearest code with their respective gradient paths, and Sinkhorn performs balanced assignment then STE. Evaluation always uses nearest-neighbor lookup. Cosine distance has no explicit zero-norm epsilon and can produce NaNs.

## `RqVae`

```python
RqVae(
    input_dim: int,
    embed_dim: int,
    hidden_dims: list[int],
    codebook_size: int,
    codebook_kmeans_init: bool = True,
    codebook_normalize: bool = False,
    codebook_sim_vq: bool = False,
    codebook_mode=QuantizeForwardMode.GUMBEL_SOFTMAX,
    codebook_last_layer_mode=QuantizeForwardMode.GUMBEL_SOFTMAX,
    n_layers: int = 3,
    commitment_weight: float = 0.25,
    n_cat_features: int = 18,
)
```

The encoder maps `input_dim -> hidden_dims -> embed_dim`; the decoder reverses that path. Residual quantization uses `n_layers`, with a separate mode for the last layer.

```python
model.encode(x: Tensor[B,D]) -> Tensor[B,E]
model.decode(z: Tensor[B,E]) -> Tensor[B,D]
model.get_semantic_ids(x: Tensor[B,D], gumbel_t=0.001) -> RqVaeOutput

class RqVaeOutput(NamedTuple):
    embeddings: Tensor    # [B,E,C]
    residuals: Tensor     # [B,E,C]
    sem_ids: Tensor       # [B,C]
    quantize_loss: Tensor # [B]
```

The reconstructed latent is `output.embeddings.sum(dim=-1)`.

```python
model.get_semantic_ids_dedup(
    x: Tensor[N,D], max_iter=20, sk_epsilon=0.003, sk_iters=100,
) -> LongTensor[N,C]
```

Deduplication freezes the first `C-1` nearest codes and reassigns only colliding last-layer groups with Sinkhorn. It calls `model.eval()` without restoring the previous mode. Residual collisions only emit a warning, so callers must measure uniqueness. `n_layers=1` works; zero layers does not.

## Training forward result

```python
model(batch: Tensor[B,D], gumbel_t: float) -> RqVaeComputedLosses

class RqVaeComputedLosses(NamedTuple):
    loss: Tensor                 # scalar
    reconstruction_loss: Tensor # scalar
    rqvae_loss: Tensor           # scalar
    embs_norm: Tensor            # [B,C]
    p_unique_ids: Tensor         # scalar batch tuple uniqueness
```

`forward` is decorated with `torch.compile(mode="reduce-overhead")`. With `n_cat_features > 0`, the trailing dimensions are BCE-with-logits targets and should contain 0/1 values; preceding continuous output is normalized. Keep `n_cat_features <= input_dim`.

## Loading

```python
model.device -> torch.device
model.config -> dict
model.load_pretrained(path: str) -> None
RqVae.from_pretrained(path: str, map_location="cpu") -> RqVae
```

Both loaders require a trusted checkpoint containing `"model"`. `load_pretrained` uses the existing geometry; `from_pretrained` reconstructs from `"model_config"`. Loading disables future K-means initialization. Older checkpoints without complete geometry must be loaded into an explicitly constructed model. The captured `config` is checkpoint metadata, not a stable JSON schema.

## Minimal example

```python
import torch
from genrec.models.rqvae import RqVae, QuantizeForwardMode

model = RqVae(
    input_dim=768, embed_dim=32,
    hidden_dims=[512, 256, 128, 64],
    codebook_size=256, n_layers=3, n_cat_features=0,
    codebook_mode=QuantizeForwardMode.STE,
    codebook_last_layer_mode=QuantizeForwardMode.SINKHORN,
)
x = torch.randn(1024, 768)  # first K-means batch must contain at least K rows
model(x, gumbel_t=0.2).loss.backward()
model.eval()
sids = model.get_semantic_ids(x).sem_ids  # [1024,3]
```

## Trainer

[`rqvae_trainer.train`](https://github.com/phonism/genrec/blob/main/genrec/trainers/rqvae_trainer.py) is Gin-configurable. Its principal arguments are `epochs=None`, `iterations=None`, warmup/batch/optimizer settings, `dataset`, item encoder and `vae_*` geometry, checkpoint path, save/eval intervals, Accelerate precision, W&B, and seed.

Exactly one of `epochs` and `iterations` must be set. The injected dataset is constructed for `train` and `eval` with `root`, `train_test_split`, and `encoder_model_name`; each row must become a `vae_input_dim` float tensor. First-time K-means initialization accumulates about 20,000 examples. The DataLoader currently hard-codes 16 persistent workers and a high prefetch count.

```bash
python genrec/trainers/rqvae_trainer.py config/tiger/amazon/rqvae.gin --split beauty
python genrec/trainers/rqvae_trainer.py config/lcrec/amazon/rqvae.gin --split beauty
python genrec/trainers/rqvae_trainer.py config/cobra/amazon/rqvae.gin --split beauty
```

Use the downstream model's matching config. Checkpoints are `checkpoint_epoch_<epoch>.pt` or `checkpoint_<iter>.pt` and contain epoch/iteration, model, model config, optimizer, and scheduler. Evaluation also keeps `checkpoint_best_candidate.pt`; only a snapshot satisfying the configured per-layer usage gate is promoted to `checkpoint_best_health.pt`. `results.json` is also written; metrics may be empty with `do_eval=False`.

The public method is `get_semantic_ids()`, not `generate_semantic_ids()`. `forward()` returns the named tuple above, and SID tuples are not guaranteed to identify unique catalog items.
