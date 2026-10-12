# TIGER API

This page covers [`genrec/models/tiger.py`](https://github.com/phonism/genrec/blob/main/genrec/models/tiger.py) and [`tiger_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/tiger_trainer.py). See [the model page](../models/tiger.md) for architecture and paper differences.

## Implementation and token contract

`Tiger` wraps the repository's pure-PyTorch [`T5ForConditionalGeneration`](https://github.com/phonism/genrec/blob/main/genrec/modules/t5.py), not a Hugging Face model.

For codebook position `c`, code `q_c` in `[0,K-1]` becomes:

```text
token = q_c + c * K + 1
```

The vocabulary is `C*K+1`; token 0 is padding, decoder start, and EOS. History item codes are flattened, while a target contains exactly `C` offset tokens and no appended EOS.

Offset ranges identify codebook positions but are **not** enforced during generation. Current TIGER has no logits mask, prefix trie, or constrained-decoding callback; invalid raw SIDs are skipped by the item mapper.

## Model

```python
from genrec.models.tiger import Tiger
model = Tiger(config: dict[str, Any])
```

Recognized keys are `num_layers=4`, `num_decoder_layers=4`, `d_model=128`, `d_ff=1024`, `num_heads=6`, `d_kv=64`, `dropout_rate=0.1`, `vocab_size=769`, `pad_token_id=0`, `eos_token_id=0`, `feed_forward_proj="relu"`, and `sem_id_dim=3`. Unknown keys are ignored. The constructor cannot verify `vocab_size == sem_id_dim * codebook_size + 1` because codebook size is held by the trainer.

```python
model.num_parameters -> tuple[int, int]  # all trainable, input-embedding trainable

loss, logits = model(
    input_ids: LongTensor[B,L],
    attention_mask: Tensor[B,L] | None,
    labels: LongTensor[B,C] | None,
)
```

With labels, loss is scalar cross entropy ignoring `-100` and logits are `[B,C,V]`. `Tiger.forward` does not expose decoder inputs, so a label-free call normally fails in the underlying decoder.

```python
predictions = model.generate(
    input_ids: LongTensor[B,L],
    attention_mask: Tensor[B,L] | None = None,
    num_beams: int = 20,
)
```

Generation sets `max_length=C+1` and `num_return_sequences=num_beams`. Beam output is `[B*num_beams,C+1]`; column 0 is decoder-start token 0 and must be removed. Greedy generation may stop early and return fewer than `C+1` columns. There is no sampling, length penalty, repetition penalty, EOS filtering, or prefix constraint.

## Collation

```python
from genrec.trainers.tiger_trainer import t5_collate

t5_collate(batch: list[SeqData], codebook_size=256,
           sem_id_dim=3, max_history_tokens=60) -> dict[str, Tensor]
```

The result contains:

| Key | Shape | Meaning |
|---|---|---|
| `input_ids` | `[B,max_history_tokens]` | offset history tokens, tail-truncated and left-padded |
| `attention_mask` | `[B,max_history_tokens]` | 1 for history, 0 for padding |
| `labels` | `[B,C]` | offset target tokens |
| `raw_targets` | `[B,C]` | target SID before offsets |
| `target_items` | `[B]` | catalog item ID |

All targets must have equal length. Keep `max_history_tokens` divisible by `C` or truncation can begin in the middle of an item's SID. The collator does not validate raw code ranges.

Legacy diagnostic helpers `calculate_pos_index(preds, labels, maxk=20)`, `recall_at_k`, and `ndcg_at_k` compare exact SID tuples. `maxk` is unused, and the current primary evaluator uses `TopKAccumulator` instead.

## Trainer

```python
@gin.configurable
def train(
    epochs=200, batch_size=256, infer_batch_size=96,
    learning_rate=1e-4, weight_decay=0.0,
    dataset_folder="dataset/amazon", save_dir_root="out/tiger/",
    dataset=None, wandb_logging=True, wandb_project="tiger_training",
    wandb_log_interval=10,
    num_layers=4, num_decoder_layers=4,
    d_model=128, d_ff=1024, num_heads=6, d_kv=64,
    dropout_rate=0.1, feed_forward_proj="relu",
    codebook_size=256, sem_id_dim=3, max_seq_len=20,
    pretrained_rqvae_path="./out/tiger/amazon/{split}/rqvae/checkpoint_epoch_4999.pt",
    lr_schedule="none", num_warmup_steps=0, beam_size=30,
    early_stop_patience=10, eval_test_every_epoch=2,
    eval_valid_every_n_epochs=1, seed=42,
) -> None
```

There is no `TIGERTrainer` class. This function runs single-process PyTorch, chooses CUDA/CPU itself, and does not use Accelerate.

The injected class is constructed for train/valid/test with `root`, `train_test_split`, `max_seq_len`, and `pretrained_rqvae_path`. It must return `SeqData` and expose `sem_ids_list`, `sequences`, and `evaluation_protocol`. The standard `AmazonSeqDataset` RQ-VAE geometry is bound separately in Gin and must match its checkpoint.

## Evaluation semantics

Evaluation records two streams:

1. exact SID tuple metrics before collision resolution, as diagnostics;
2. item-level Recall/NDCG after `SemanticIdItemMapper` resolves collisions with training-prefix popularity and deduplicates items.

Early stopping and the main result use item-level `Recall@10`. A negative `target_item` raises `ValueError`. Only K=5 and 10 are calculated; a beam smaller than 10 yields fewer available candidates.

`lr_schedule="cosine"` enables step-wise cosine decay and optional linear warmup; other values use no scheduler. Validation runs at epoch 0 and then by its interval. A model is saved only on a strict Recall@10 improvement over the initial zero, so an all-zero run may never create `best_model.pt`.

## Run and artifacts

```bash
python genrec/trainers/rqvae_trainer.py config/tiger/amazon/rqvae.gin --split beauty
python genrec/trainers/tiger_trainer.py config/tiger/amazon/tiger.gin --split beauty
```

`best_model.pt` is only the raw model `state_dict`; reconstruct the identical `Tiger` config before loading. The directory also contains timestamped logs and `results.json`, whose main metrics are item-level and whose extra fields retain SID diagnostics and collision-resolution metadata.

There is no `generate_with_trie()`, `TIGERTrainer`, or generic `Trainer.fit()`. Also keep `AmazonSeqDataset.add_disambiguation=False` with the standard three-code TIGER config; a fourth suffix code would misalign `sem_id_dim=3`.
