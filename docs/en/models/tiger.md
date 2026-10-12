# TIGER

TIGER reformulates next-item recommendation as short-sequence generation. An RQ-VAE assigns each item a semantic ID, then a T5 encoder-decoder generates the target item's SID from history SIDs.

| Field | Current repository |
|---|---|
| Paper | [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065), NeurIPS 2023 |
| Reference code | [Google Research TIGER](https://github.com/google-research/google-research/tree/master/tiger) |
| Representation | default 3 × 256 RQ-VAE codes |
| Backbone | repository-local pure-PyTorch T5 |
| Decoding | ordinary beam search; **no catalog trie** |
| Headline | item Recall/NDCG after training-only SID resolution and item deduplication |
| Status | end-to-end path; historical SID-tuple checkpoints require item-level reevaluation |

## Two-stage pipeline

Stage 1 encodes item metadata with Sentence-T5-XL and trains the matching RQ-VAE. Stage 2 replaces every history item with its SID and trains T5. Both stages must preserve the same catalog order, sentence encoder, and RQ-VAE geometry.

Raw code `c_j` at layer `j` becomes a position-specific token:

\[
\tilde c_j=c_j+jK+1.
\]

Token 0 is padding/decoder start/EOS and vocabulary size is `C*K+1`. A history of `max_seq_len` items contains at most `max_seq_len*C` tokens.

Given history `x` and target codes, teacher forcing minimizes

\[
\mathcal L=-\sum_{j=1}^{C}\log p_\theta(y_j\mid y_{<j},x).
\]

The current model is [`genrec/modules/t5.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/t5.py), not a runtime Hugging Face T5. Input/output embeddings and LM head are tied.

## Current experiment shape

The Amazon Gin uses 4 encoder and 4 decoder layers, `d_model=128`, `d_ff=1024`, 6 heads with `d_kv=64`, dropout 0.1, 3×256 SIDs, 50 history items (150 tokens), beam 30, and Adam at `1e-4` without a scheduler.

`AmazonSeqDataset` produces sliding-window training prefixes under leave-one-out; validation targets the penultimate and test the last interaction. `SeqData` contains flattened raw history codes, raw target SID, and the true catalog item ID.

Collision options are mutually exclusive in intent:

- `add_disambiguation=True` appends a group index, increasing SID width to `C+1`;
- `dedup_sinkhorn=True` preserves width and reassigns the last RQ-VAE code.

The standard config disables both and resolves collisions at evaluation. If a suffix is enabled, `sem_id_dim`, vocabulary, and suffix range must all be updated.

## Beam search is unconstrained

`Tiger.generate()` sets `max_length=sem_id_dim+1`, uses `num_beams` return sequences, and removes the initial decoder-start token before decoding codes. There is no prefix callback, catalog trie, or per-position logits mask. A beam may emit a token from the wrong codebook interval or an SID absent from the catalog; the resolver drops it.

This is an implementation fact, not merely a documentation caveat. There is no `generate_with_trie()` method. OneRec has a `PrefixTrie`; TIGER does not.

## Item-level evaluation

SID tuples can collide. `SemanticIdItemMapper.from_sequences` estimates popularity only from each user's training prefix and chooses the most popular item in a collision group, breaking ties by smaller item ID:

\[
g(s)=\arg\max_{i:\operatorname{SID}(i)=s}\operatorname{pop}_{train}(i).
\]

Evaluation decodes offsets, maps catalog SIDs to items, removes duplicate items while retaining first rank, fills from deeper beams, and computes item Recall/NDCG against `target_item`. SID-tuple metrics remain diagnostic only. Item-level validation Recall@10 controls early stopping and checkpoint selection.

| Layer | Source | Role |
|---|---|---|
| wrapper | [`tiger.py`](https://github.com/phonism/genrec/blob/main/genrec/models/tiger.py) | T5 construction, forward, beam generation |
| local T5 | [`t5.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/t5.py) | encoder-decoder and generation |
| tokenizer | [`rqvae.py`](https://github.com/phonism/genrec/blob/main/genrec/models/rqvae.py) | item embedding to SID |
| data | [`amazon.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon.py) | SID catalog and sequence samples |
| collation/training | [`tiger_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/tiger_trainer.py) | offsets, single-device loop, dual metrics |
| resolver | [`metrics.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/metrics.py) | training-only SID-to-item mapping |

## Run and artifacts

```bash
python genrec/trainers/rqvae_trainer.py config/tiger/amazon/rqvae.gin --split beauty
python genrec/trainers/tiger_trainer.py config/tiger/amazon/tiger.gin --split beauty
```

The default stage-2 dependency is `out/tiger/amazon/<split>/rqvae/checkpoint_epoch_4999.pt`. When overriding it, also verify item encoder and all RQ-VAE dimensions. The TIGER trainer is single-process and must not be assumed to support an Accelerate multi-GPU launch.

`best_model.pt` contains only the TIGER state dict. `results.json` contains item-level main metrics, seed/config/best epoch, resolver provenance, and separate SID diagnostic metrics.

## Reproduction boundary and pitfalls

Compared with the TIGER work/reference pipeline, GenRec uses a local T5, unconstrained beam followed by invalid-candidate filtering, explicit training-only collision resolution, item-level model selection, and a single-device trainer. Therefore historical SID-level values cannot be reused as current headlines.

Common failures are wrong RQ-VAE geometry, codebook bootstrap rerunning after load, high SID but low item recall due to collisions, illegal beam tokens due to absent constraints, or a fourth disambiguation code with a three-code model.

See the exact [TIGER API](../api/tiger.md), [RQ-VAE](rqvae.md), and [evaluation terminology](../reference/terminology.md).
