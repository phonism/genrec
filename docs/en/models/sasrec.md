# SASRec

SASRec is GenRec's primary discriminative sequential-recommendation baseline. It encodes an item-ID history with causal self-attention and scores the entire catalog directly. Generative models should be compared against a converged SASRec under the same split, sequence length, and item-level protocol.

| Field | Current repository |
|---|---|
| Paper | [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781), ICDM 2018 |
| Task | next-item prediction |
| Representation | unique item ID; 0 is padding |
| Output | logits over `num_items + 1` at each position |
| Evaluation | full-catalog item Recall/NDCG |
| Status | end-to-end implementation; baseline values belong in `BENCHMARKS.md` |

SASRec needs no semantic tokenizer, collision resolver, or beam search. It is therefore the clean anchor for judging whether a generative route improves actual catalog retrieval rather than only SID-tuple accuracy.

## Model

For an item sequence, the input is

\[
X_t = E_{i_t} + P_t.
\]

Each block applies causal multi-head attention and a point-wise feed-forward network:

\[
A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M_{causal}+M_{pad}\right)V,
\qquad s_{t,j}=h_t^\top E_j.
\]

The current implementation retains details from the official TensorFlow code: Q comes from normalized input while K/V use the block input; key masking precedes softmax and query masking follows it; residuals live inside attention/FFN modules.

```python
logits, loss = model(
    input_ids,            # [B,L], left-padded
    targets=None,         # [B,L], 0 ignored
    negatives=None,       # [B,L], BCE only
)
```

CE or inference returns logits `[B,L,V+1]`. BCE training returns `(None, loss)` to avoid the full-catalog matrix multiplication. `predict(input_ids, top_k)` uses the final position and excludes padding ID 0.

## Training objectives

The current Amazon 2014 headline config uses full-vocabulary cross entropy:

\[
\mathcal L_{CE}=-\log\frac{\exp(h_t^\top E_{y_t})}
{\sum_{j=0}^{V}\exp(h_t^\top E_j)}.
\]

`loss_type="bce"` samples one non-positive item per valid position:

\[
\mathcal L_{BCE}=-\log\sigma(h_t^\top E_{y_t})
-\log(1-\sigma(h_t^\top E_{n_t})).
\]

The negative may still be another item in the user's history. CE and BCE are different experiments and must be labeled separately.

## Data and evaluation

`AmazonSASRecDataset` sorts interactions by time. Under leave-one-out, training removes the final two interactions and the collator builds multi-position supervision; validation predicts the penultimate item and test predicts the last. Long histories keep the most recent items and batches are left-padded.

`global_temporal` is also supported. Keep its outputs separate from leave-one-out. Full-catalog evaluation excludes padding but currently does not mask previously seen items.

| Layer | Source | Symbols |
|---|---|---|
| model | [`sasrec.py`](https://github.com/phonism/genrec/blob/main/genrec/models/sasrec.py) | `SASRec`, `SASRecBlock` |
| data/collation | [`amazon_sasrec.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_sasrec.py) | `AmazonSASRecDataset`, train/eval collators |
| training | [`sasrec_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/sasrec_trainer.py) | `train`, `evaluate` |
| config | [`config/sasrec/`](https://github.com/phonism/genrec/tree/main/config/sasrec) | Amazon 2014 and converted Amazon 2023 |

## Run

```bash
python genrec/trainers/sasrec_trainer.py config/sasrec/amazon.gin --split beauty

python genrec/trainers/sasrec_trainer.py config/sasrec/amazon.gin \
  --split beauty --gin 'train.loss_type="bce"'

python genrec/trainers/sasrec_trainer.py config/sasrec/amazon.gin \
  --split beauty \
  --gin 'AmazonSASRecDataset.evaluation_protocol="global_temporal"' \
  --gin 'train.save_dir_root="out/sasrec/amazon/beauty/global_temporal/s42"'
```

The Amazon 2014 Gin config currently overrides Python defaults to 300 epochs, batch 128, embedding 64, one head, two blocks, FFN 256, dropout 0.5, and CE. Validation Recall@10 selects `best_model.pt`; the best model is tested and `results.json` is written.

Amazon 2023 requires prior conversion with `python -m genrec.data.amazon2023`; changing configs alone does not download/filter it.

## Reproduction boundary

The original work commonly uses point-wise BCE and sampled-negative evaluation. This repository's headline uses CE and full-catalog evaluation on its own Amazon protocol, without a seen-item mask. Treat repository results as internal fair-comparison numbers, not unconditional numerical reproduction of the paper table.

See the [API signature](../api/trainers.md#baselines), [dataset contract](../api/datasets.md#discriminative-and-retrieval-adapters), and [implementation status](../reference/model-status.md).
