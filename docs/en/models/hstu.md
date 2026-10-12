# HSTU

HSTU uses non-normalized SiLU attention, an update gate, and relative attention bias for sequential recommendation. In GenRec it is a strong discriminative counterpart to SASRec—not a semantic-ID generator.

| Field | Current repository |
|---|---|
| Paper | [Actions Speak Louder than Words](https://arxiv.org/abs/2402.17152), ICML 2024 |
| Official code | [facebookresearch/generative-recommenders](https://github.com/facebookresearch/generative-recommenders) |
| Representation/output | unique item ID / full-catalog logits |
| Data | Amazon and MovieLens-1M adapters |
| Status | small-scale dense-tensor HSTU-inspired baseline, not Meta's production stack |

## How it differs from SASRec

| Component | SASRec | HSTU here |
|---|---|---|
| attention activation | normalized softmax | non-normalized SiLU |
| projections | Q, K, V | U, V, Q, K |
| aggregation | `softmax(QK) @ V` | `SiLU(QK + RAB) / max_seq_len @ V` |
| gate | none | `Norm(A) * U`, optional concatenated gate |
| temporal signal | position embedding | relative position and optional time-difference bias |

For input `X`, one layer computes

\[
[U,V,Q,K]=W\operatorname{LN}(X),
\]

\[
A=\frac{\operatorname{SiLU}(QK^\top+B_{pos}+B_{time}+M)}{L_{max}}V,
\qquad
Y=X+W_o\left(U\odot\operatorname{LN}(A)\right).
\]

The repository adds a SiLU FFN residual after this transformation. With `concat_ua=True`, the projection consumes `[U,A,U*A]`. Scaling uses the configured maximum sequence length, not the longest sequence in the current batch.

`RelativePositionBias` uses exact small-distance and logarithmic large-distance buckets. `TemporalBias` buckets absolute Unix-time differences logarithmically. Timestamp padding is 0 but is also key-masked.

## Objectives and tensor contract

```python
logits, loss = model(
    input_ids,                # [B,L]
    timestamps=None,          # [B,L]
    targets=None,             # [B,L]
    sampled_softmax=False,
    ...,
)
```

Full CE returns `[B,L,V+1]` logits. During sampled-softmax training logits are `None`. The model's `score_l2_norm` and `score_temperature` keep training and evaluation scoring consistent.

The sampled-softmax config creates a positive-plus-negatives classification problem. Meta's `sampling_strategy="local"` means the item embedding table is local rather than sharded: every positive independently draws 128 uniform catalog IDs with replacement. It is not in-batch sampling. Accidental positives are masked, query/items are L2-normalized, and temperature is 0.05. `in_batch` is a separate supported sampler; `global` preserves GenRec's older shared, deduplicated negative set and must not be mixed with the corrected local results.

## Data and evaluation

The Amazon adapter follows the same leave-one-out structure as SASRec and retains timestamps. `dataset_type="ml1m"` selects `MovieLens1MHSTUDataset`. Evaluation scores the full catalog at the last valid position, excludes padding ID 0, and does not mask history items.

| Layer | Source | Symbols |
|---|---|---|
| model | [`hstu.py`](https://github.com/phonism/genrec/blob/main/genrec/models/hstu.py) | `HSTU`, `HSTULayer`, position/time bias |
| Amazon data | [`amazon_hstu.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_hstu.py) | dataset and collators |
| MovieLens | [`ml1m_hstu.py`](https://github.com/phonism/genrec/blob/main/genrec/data/ml1m_hstu.py) | `MovieLens1MHSTUDataset` |
| training | [`hstu_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/hstu_trainer.py) | `train`, `evaluate` |
| configs | [`config/hstu/`](https://github.com/phonism/genrec/tree/main/config/hstu) | CE and sampled-softmax variants |

## Run

```bash
# full-vocabulary CE with temporal bias
python genrec/trainers/hstu_trainer.py config/hstu/amazon.gin --split beauty

# Meta local-table sampled softmax
python genrec/trainers/hstu_trainer.py config/hstu/amazon_ss.gin --split beauty

# temporal-bias ablation
python genrec/trainers/hstu_trainer.py config/hstu/amazon.gin \
  --split beauty --gin 'train.use_temporal_bias=False'

# separate global-temporal experiment
python genrec/trainers/hstu_trainer.py config/hstu/amazon.gin \
  --split beauty \
  --gin 'AmazonHSTUDataset.evaluation_protocol="global_temporal"' \
  --gin 'train.save_dir_root="out/hstu/amazon/beauty/global_temporal/s42"'
```

Disabling temporal bias does not disable relative position bias; bind `train.use_position_bias=False` as well when that is the intended ablation.

## Reproduction boundary

Meta's work targets long jagged sequences, specialized kernels, many action/features, and industrial scale. This implementation uses dense `[B,H,L,L]` attention, item ID plus timestamp, small academic configurations, and an extra FFN. It is suitable for testing the SiLU-attention/U-gate/temporal-RAB ideas under a common benchmark, not for validating production throughput or scaling claims.

See [trainer parameters](../api/trainers.md#baselines), [dataset schemas](../api/datasets.md#discriminative-and-retrieval-adapters), and [implementation status](../reference/model-status.md).
