# Training TIGER

TIGER has two stages: an RQ-VAE item tokenizer and a T5 recommender. The current
trainer is a single-process device path using ordinary beam search. It does not
provide the old documented Accelerate/DDP or Trie-constrained workflow.

## Prerequisites

```bash
test -f dataset/amazon/raw/beauty/reviews_Beauty_5.json.gz
test -f dataset/amazon/processed/beauty/item_emb_sentence-t5-xl.parquet
test -f out/tiger/amazon/beauty/rqvae/checkpoint_epoch_4999.pt
```

The tokenizer must use TIGER's 3 × 256 codebooks, latent width 32, hidden widths
`[512, 256, 128, 64]`, and the same Sentence-T5 cache consumed by the dataset.

## Stage 1: tokenizer

```bash
python genrec/trainers/rqvae_trainer.py \
  config/tiger/amazon/rqvae.gin --split beauty
```

Run collision and codebook-health checks before starting the recommender. See
[training RQ-VAE](rqvae.md).

## Stage 2: recommender

```bash
python genrec/trainers/tiger_trainer.py \
  config/tiger/amazon/tiger.gin --split beauty \
  --gin 'train.wandb_logging=False'
```

Current defaults are 200 epochs, batch size 256, Adam with learning rate
`1e-4`, 50 history items, 30 beams, and early stopping after 10 validation
Recall@10 checks without improvement.

Override the tokenizer path when necessary:

```bash
python genrec/trainers/tiger_trainer.py \
  config/tiger/amazon/tiger.gin --split beauty \
  --gin 'train.pretrained_rqvae_path="/abs/path/checkpoint.pt"'
```

## Samples and token counts

For each training user, the adapter first removes the validation and test items,
then creates every non-empty prefix-to-next-item sample. With three SID codes:

```text
50 history items × 3 codes = 150 encoder tokens
target = 3 decoder labels
vocabulary = 3 × 256 + padding = 769
```

`t5_collate` left-pads encoder tokens and offsets every code by codebook
position. `max_seq_len` counts history items; do not pass it as an already
flattened token length.

## Item-level evaluation

Each validation pass:

1. Generates `beam_size` sequences of length C.
2. Removes the decoder-start token and reverses positional offsets.
3. Records exact SID-tuple match as a diagnostic.
4. Resolves every SID to one item using popularity from training prefixes only.
5. Removes invalid SIDs and duplicate items.
6. Computes item-level Recall@5/10 and NDCG@5/10.
7. Selects `best_model.pt` by item-level validation Recall@10.

The test set is evaluated whenever validation improves and may also be run at
the `eval_test_every_epoch` diagnostic interval. Headline values in
`results.json` are the item-level test metrics associated with the best
validation checkpoint.

## Decoding is not constrained

`Tiger.generate` has no Trie or prefix callback. To diagnose beam quality,
measure:

- whether each generated token falls in the offset range for its codebook;
- whether the decoded SID exists in the catalog;
- how many candidates remain after item resolution and deduplication.

A larger beam gives post-filtering more candidates, but it does not prevent
invalid generation and increases decoder work.

## Outputs

```text
out/tiger/amazon/beauty/
├── best_model.pt
├── tiger_<timestamp>.log
└── results.json
```

`best_model.pt` is a model `state_dict`, not a Hugging Face
`save_pretrained` directory. Reconstruct `Tiger(model_config)` with matching Gin
values before loading it.

## Fair-comparison rules

- `AmazonSeqDataset.add_disambiguation=False` is the current default; changing
  it is a semantic-ID design ablation.
- Do not use `dedup_sinkhorn` together with a disambiguation suffix.
- Changing SID width, codebook size, and RQ-VAE depth together prevents clean
  attribution.
- Legacy SID-tuple results are diagnostics, not headline metrics.
- Comparisons with SASRec/HSTU must share split, seed, catalog, item-level metric,
  and `max_seq_len=50`.

## Troubleshooting

- **RQ-VAE state shape mismatch:** check all tokenizer geometry parameters;
  never hide it with `strict=False`.
- **Unexpected beam shape:** generation returns `[B * beam, C + 1]`; the first
  token is decoder start.
- **Item metrics are zero but SID match is nonzero:** verify `target_item`, row
  alignment, and the training-only SID resolver.
- **Fewer than 10 valid candidates:** report invalid-SID and deduplication rates;
  the mapper pads missing positions with `-1`.
- **No multi-GPU speedup:** the current trainer has no Accelerate/DDP wrapper.
- **Old code calls `build_trie`:** that function is not present in
  `genrec.models.tiger`.

Useful checks:

```bash
pytest -q tests/test_honest_metrics.py tests/test_rqvae_checkpoint.py
python scripts/verify_t5_equivalence.py
python scripts/verify_pure_t5.py
```

The two parity scripts execute immediately rather than exposing `--help`; they
require `transformers` and `torch`. See [the TIGER model page](../models/tiger.md)
for the architecture and token-offset equations.
