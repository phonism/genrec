# Shared modules API

This page indexes reusable PyTorch components in [`genrec/modules/`](https://github.com/phonism/genrec/tree/main/genrec/modules). They are research utilities, not a versioned SDK.

`genrec.modules` re-exports the three losses, `TopKAccumulator`, `L2Norm`, `RMSNorm`, `SemIdEmbedding`, `UserIdEmbedding`, `parse_config`, and `setup_logger`. Import encoders, transformers, T5, K-means, schedulers, and `SemanticIdItemMapper` from their concrete modules.

## Embeddings

Source: [`embedding.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/embedding.py)

```python
SemIdEmbedding(num_embeddings, sem_ids_dim, embeddings_dim)
SemIdEmbedding.forward(input_ids: Tensor[B,T],
                       token_type_ids: Tensor[B,T]) -> Tensor[B,T,D]
```

Code `i` at codebook position `c` maps to `c * num_embeddings + i`. The table has `sem_ids_dim * num_embeddings + 1` rows; the last is padding. Inputs are not range-checked.

```python
UserIdEmbedding(num_embeddings, embeddings_dim)
UserIdEmbedding.forward(input_ids) -> Tensor
```

User IDs are reduced modulo `num_embeddings`, so collisions are intentional.

## Encoders and MLP

Source: [`encoder.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/encoder.py)

| Class | Contract |
|---|---|
| `LightT5Encoder(...)` | Randomly initialized `nn.TransformerEncoder`; token 0 is padding; `[B,L]` or `[B,T,L]` to normalized embeddings. It is not a pretrained T5. |
| `SentenceT5Encoder(model_name, output_dim=768)` | Extracts encoder/pooling/dense modules from a SentenceTransformers model and returns normalized vectors. |
| `ErnieEncoder(model_name, out_dim=768)` | Hugging Face tokenizer/model CLS vectors; exposes `tokenize`, `encode`, and `forward`. |
| `BgeEncoder(model_name)` | Normalized CLS from token tensors; its `encode(..., device="cuda")` argument is currently unused. |
| `MLP(input_dim, hidden_dims, out_dim, dropout=0, normalize=False)` | SiLU hidden layers; asserts the final input dimension and optionally normalizes output. |

All four encoders flatten a 3D item axis and restore it only when `T > 1`. An input shaped `[B,1,L]` therefore returns `[B,D]`, not `[B,1,D]`; add the item axis yourself when required. `LightT5Encoder` raises `ValueError` outside 2D/3D and can index out of range above `max_seq_len`.

## Transformer building blocks

Source: [`transformer.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/transformer.py)

```python
T5Attention(d_model, n_heads, dropout=0.0,
            is_cross_attention=False, has_relative_bias=True,
            num_relative_buckets=32, max_distance=128,
            d_kv=0, scale_attn=True, bidirectional=True)

T5Attention.forward(query, key=None, value=None, attn_mask=None,
                    key_padding_mask=None, position_bias=None,
                    need_weights=False) -> tuple[Tensor, Tensor | None]
```

Inputs are batch-first `[B,L,D]`; the optional shared bias is `[1,H,Q,K]`. A boolean padding mask uses `True` for ignored positions; an attention mask is additive, not a Hugging Face 0/1 mask.

The same module provides `FeedForward`, `MultiHeadAttention`, `TransformerBlock`, `TransformerEncoder`, `TransformerDecoder`, and `TransformerEncoderDecoder`. Containers take `[B,L,D]`; the encoder-decoder creates a causal target mask when none is supplied.

There are two unrelated `T5Attention` classes:

```python
from genrec.modules.transformer import T5Attention  # generic blocks
from genrec.modules.t5 import T5Attention           # local full T5 internals
```

## Local T5

Source: [`t5.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/t5.py)

```python
T5Config(...)
T5ForConditionalGeneration(config)

model.forward(input_ids=None, attention_mask=None,
              decoder_input_ids=None, decoder_attention_mask=None,
              labels=None, encoder_outputs=None,
              use_cache=False, past_key_values=None)

model.generate(input_ids, attention_mask=None, max_length=20,
               num_beams=1, num_return_sequences=1) -> LongTensor
```

This is TIGER's pure-PyTorch implementation; it does not import Hugging Face Transformers at runtime. Forward output exposes `.loss`, `.logits`, and `.past_key_values`. Label loss ignores `-100`. Generation implements greedy and beam search only—no sampling, logits processors, `GenerationConfig`, or prefix-trie callback. Generated tensors include the decoder-start token.

## Losses

Source: [`loss.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/loss.py)

- `ReconstructionLoss()(x_hat, x)` returns per-example squared error summed over the final dimension, without a batch mean.
- `CategoricalReconstructionLoss(n_cat_feats)` adds elementwise BCE-with-logits for trailing categorical features. Use `n_cat_feats > 0`; zero triggers an empty `:-0` continuous slice.
- `QuantizeLoss(commitment_weight=1.0)(query, value)` returns codebook plus weighted commitment loss, summed only over the last dimension.

## Metrics and SID resolution

Source: [`metrics.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/metrics.py)

```python
SemanticIdItemMapper(semantic_ids, item_popularity=None,
                     padding_item_id=-1)
SemanticIdItemMapper.from_sequences(semantic_ids, sequences,
                                    train_holdout=2,
                                    padding_item_id=-1)
mapper.map_predictions(predicted_semantic_ids: Tensor[B,K,C],
                       topk=None, deduplicate=True) -> Tensor[B,topk]
```

SID collisions resolve to the most popular **training-prefix** item, with smaller item ID as tie-breaker. Unknown SIDs are skipped, duplicate items can be removed, and short rows are padded. Diagnostics include collision rate, unique SID count, collision-item count, and maximum group size.

```python
TopKAccumulator(ks=[1,5,10])
acc.accumulate(actual: Tensor[B,D], top_k: Tensor[B,K,D])
acc.sync_across_processes(accelerator)
acc.reduce() -> dict[str, float]
```

Matching requires equality over all `D`; item IDs can use `D=1`. Every distributed rank must enter synchronization.

## K-means, normalization, and scheduling

```python
kmeans_init_(tensor: Tensor, x: Tensor) -> None
Kmeans(k, max_iters=None, stop_threshold=1e-10).run(x) -> KmeansOutput
sample_gumbel(shape, device, eps=1e-20) -> Tensor
gumbel_softmax_sample(logits, temperature, device) -> Tensor
```

K-means inputs are 2D; `k > batch_size` fails during no-replacement initialization. Empty clusters are reset to samples. `max_iters=None` has no hard cap. Gumbel softmax returns probabilities, not a hard straight-through one-hot.

Normalization utilities are `l2norm`, `L2Norm`, `RMSNorm`, `SwishLayerNorm`, and `RootMeanSquareLayerNorm`. `InverseSquareRootScheduler` linearly warms up then decays as an inverse square root; avoid `warmup_steps=0`, which yields zero scale in the decay branch.

## Utilities

[`utils.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/utils.py) provides `setup_logger`, `parse_config`, `get_run_split`, cache-reset/eval decorators, `select_columns_per_row`, `maybe_repeat_interleave`, and `compute_debug_metrics`. The decorators do not use `finally`; if the wrapped function raises, cache cleanup or training-mode restoration may not run.

Dependencies are feature-specific: core components need PyTorch; K-means also uses NumPy/einops; pretrained encoders need SentenceTransformers or Transformers. The local T5 itself has no Transformers dependency.
