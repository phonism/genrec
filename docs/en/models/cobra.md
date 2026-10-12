# COBRA: cascaded sparse and dense generation

| Field | Current repository |
|---|---|
| Paper | [Sparse Meets Dense: Unified Generative Recommendations](https://arxiv.org/abs/2503.02453), NeurIPS 2025 |
| Task | coarse-to-fine next-item generative retrieval |
| Representation | three RQ-VAE codes plus one text-encoder dense vector |
| Backbone | causal `nn.TransformerDecoder` with empty cross-attention memory |
| Training | per-codebook CE plus in-batch dense InfoNCE |
| Decoding | sparse beam, exact SID bucket, dense cosine, BeamFusion |
| Headline | item Recall/NDCG on `BeamFusionOutput.item_ids` |
| Status | complete code path; no verified publishable benchmark row yet |

GenRec implements the paper's central sparse-then-dense cascade, but vendors no official COBRA code and has no layer-by-layer parity suite. Treat it as a research reproduction, not an official mirror.

## Two encoder spaces

Do not conflate:

1. the Sentence-T5-XL catalog embeddings consumed by the dedicated 3×32 RQ-VAE;
2. the trainable COBRA dense encoder, Sentence-T5-base in the baseline or random `LightT5Encoder` in v2.

The trainer intentionally does not pass the model's base encoder name to the dataset. Changing the dataset from XL to base quantizes in the wrong RQ-VAE space.

## Interleaved representation and loss

Each item becomes

\[
z_t=[e(s_{t,0}),e(s_{t,1}),e(s_{t,2}),v_t].
\]

Default dimensions are `C=3`, `K=32`, and `d=768`. Sparse IDs use a `C*K+1` table with `C*K` reserved for padding; absolute position and sparse/dense type embeddings are added. Dense vectors enter directly with no extra input projection.

`CobraDecoder` receives empty `[B,0,d]` memory, so its useful behavior is decoder-only causal self-attention even though cross-attention sublayers still exist.

Independent heads predict codes in a cascade. The final sparse hidden state, normalized directly, predicts the dense vector. The trainer minimizes

\[
\mathcal L=\lambda_s\mathcal L_{sparse}+\lambda_d\mathcal L_{dense},
\]

where sparse loss averages codebook CE and dense loss is in-batch InfoNCE over valid transitions. The legacy scope masks same-sequence transitions; the paper-aligned scope uses all batch targets while masking repeated copies of the exact catalog item. Target vectors are detached. The memory queue is an optional research extension and is disabled for the paper-aligned control.

With `use_dense=False`, dense history positions are zeroed and InfoNCE is zero. The encoder remains untrained but current evaluation still invokes it for BeamFusion, so sparse-only `results.json.metrics` is not a clean sparse-only headline.

## Data and generation

`AmazonCobraDataset` caches each item's raw SID, configured text tokens, and XL embedding. The paper-aligned path uses title, price, category, and description for both its dedicated RQ-VAE and dense encoder. Under leave-one-out, each user contributes one training row; the collator appends its target and model forward supervises all adjacent transitions in the resulting sequence.

```text
CobraGenerationOutput
  sem_ids    [B,M,C]
  dense_vecs [B,M,d]
  scores     [B,M]
```

`Cobra.generate()` produces each code by rerunning the full decoder; there is no KV cache or catalog trie. BeamFusion finds items with an exactly matching full SID, then combines normalized beam probability and within-bucket dense cosine:

\[
w(m,a)=\operatorname{softmax}_m(\tau\phi_m)
\operatorname{softmax}_{a\in\mathcal C(s_m)}
(\psi\cos(\hat v_m,v_a)).
\]

Beam width, `tau`, `psi`, and catalog-prefix constraints are Gin-configurable;
top 10 is the benchmark output width.

Evaluation recalculates the catalog's dense vectors and records BeamFusion item
metrics as headline plus two SID-level diagnostic streams. Validation alone
selects `best_model.pt`; every rank reloads that snapshot and test runs once.

Missing candidates use item ID/SID `-1`, so valid catalog item 0 cannot receive a
padding-induced hit. Dense InfoNCE also masks off-diagonal copies of the same
catalog item instead of treating them as contradictory negatives.

## Code and run

| Layer | Source | Role |
|---|---|---|
| model/outputs | [`cobra.py`](https://github.com/phonism/genrec/blob/main/genrec/models/cobra.py) | encoder, interleave, decoder, losses, generation, fusion |
| dense encoders | [`encoder.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/encoder.py) | pretrained Sentence-T5 or random LightT5 |
| data | [`amazon_cobra.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_cobra.py) | RQ-VAE catalog and text tokens |
| collator/trainer | [`cobra_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/cobra_trainer.py) | Accelerate, cosine schedule, evaluation, resume |
| configs | [`config/cobra/amazon/`](https://github.com/phonism/genrec/tree/main/config/cobra/amazon) | baseline, v2, and sparse-only |

```bash
python -m genrec.trainers.rqvae_trainer \
  config/cobra/amazon/rqvae.gin --split beauty
python -m genrec.trainers.cobra_trainer \
  config/cobra/amazon/cobra.gin --split beauty

python -m genrec.trainers.rqvae_trainer \
  config/cobra/amazon/rqvae_paper_attributes.gin --split beauty
python -m genrec.trainers.cobra_trainer \
  config/cobra/amazon/cobra_paper_aligned.gin --split beauty
```

Stage 2 reads `out/cobra/amazon/<split>/rqvae/checkpoint_epoch_4999.pt`. TIGER/LCRec RQ-VAE checkpoints are geometrically incompatible. Optional variants are `cobra_v2.gin` and `cobra_sparse_only.gin`; no Amazon 2023 COBRA config currently exists.

The baseline uses 20 epochs, batch 32 with accumulation 4, `1e-4`, 500 warmup steps, 3×32 SID, 768 width, two decoder layers, Sentence-T5-base dense encoder, 100-item histories, and fp32 (`amp=False`). Checkpoints are dicts containing epoch, model, optimizer, and scheduler, but not constructor config, catalog fingerprint, or RNG/DataLoader state. There is no `eval_only` route.

## Limits before formal reproduction

The implementation is only safe for `C=3` in several hard-coded paths. Catalog-prefix constrained beam is optional, while KV cache, clean sparse-only item resolution, and full C-generic/distributed/resume coverage remain missing. Validation now selects `best_model.pt`, all ranks reload it, and test runs exactly once. Several scripts under `scripts/` are hard-coded legacy probes rather than portable evaluators. Publish only item-level artifacts that pass matrix validation; keep SID values labeled pre-resolution diagnostics.

See [RQ-VAE](rqvae.md), [trainer API](../api/trainers.md#semantic-id-recommenders), and [model status](../reference/model-status.md).
