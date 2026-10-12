# Trainer API

There is no `Trainer`, `BaseTrainer`, `RQVAETrainer`, or shared callback lifecycle. Each trainer module exposes an independent top-level `@gin.configurable` function and starts it after `parse_config()`. Training entry points return `None`; metrics are delivered through logs, W&B, and `results.json`.

Import repeated names explicitly:

```python
from genrec.trainers.sasrec_trainer import train as train_sasrec
from genrec.trainers.tiger_trainer import train as train_tiger
from genrec.trainers.rqvae_trainer import train as train_rqvae
from genrec.trainers.onerec_sft_trainer import train_sft
```

`genrec.trainers.__init__` imports shared utilities only. Do not rely on its `__all__` to obtain trainer modules.

## Entry-point index

| Pipeline | Function / source | Runtime |
|---|---|---|
| SASRec | `train` / [`sasrec_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/sasrec_trainer.py) | Accelerate |
| HSTU | `train` / [`hstu_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/hstu_trainer.py) | Accelerate |
| RQ-VAE | `train` / [`rqvae_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/rqvae_trainer.py) | Accelerate |
| RQ-KMeans | `fit_rqkmeans` / [`rqkmeans_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/rqkmeans_trainer.py) | CPU / scikit-learn |
| TIGER | `train` / [`tiger_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/tiger_trainer.py) | single-process PyTorch |
| LCRec | `train` / [`lcrec_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/lcrec_trainer.py) | Accelerate |
| COBRA | `train` / [`cobra_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/cobra_trainer.py) | Accelerate |
| OneRec SFT | `train_sft` / [`onerec_sft_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/onerec_sft_trainer.py) | Accelerate, FSDP-aware save |
| OneRec GRPO | `train_grpo` / [`onerec_grpo_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/onerec_grpo_trainer.py) | Accelerate |
| OneRec CLM | `train_clm` / [`onerec_clm_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/onerec_clm_trainer.py) | Accelerate |
| RPG | `train_rpg` / [`rpg_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/rpg_trainer.py) | Accelerate |
| NoteLLM | `train` / [`notellm_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/notellm_trainer.py) | Accelerate |
| GRAM | `train_gram` / [`gram_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/gram_trainer.py) | Accelerate |

## Baselines

```python
# sasrec_trainer.py
train(
    epochs=200, batch_size=128, learning_rate=1e-3, weight_decay=0.0,
    max_seq_len=50, embed_dim=64, num_heads=2, num_blocks=2,
    ffn_dim=256, dropout=0.2, loss_type="bce",
    dataset_folder="dataset/amazon", split="beauty",
    do_eval=True, eval_every_epoch=1, eval_batch_size=256,
    patience=50, save_dir_root="out/sasrec/amazon/beauty",
    save_every_epoch=50, wandb_logging=False,
    wandb_project="sasrec_training", wandb_log_interval=100,
    amp=True, mixed_precision_type="bf16", seed=42,
)
```

SASRec constructs `AmazonSASRecDataset` internally. `loss_type="bce"` uses one sampled negative per position; any other string selects full CE. Item-level Recall@10 controls early stopping.

```python
# hstu_trainer.py
train(
    epochs=200, batch_size=128, learning_rate=1e-3, weight_decay=0.0,
    max_seq_len=50, embed_dim=64, num_heads=2, num_blocks=2,
    dropout=0.2, dqk=None, dv=None, use_position_bias=True,
    concat_ua=False, num_position_buckets=32, num_time_buckets=64,
    use_temporal_bias=True, loss_type="ce", num_negatives=128,
    ss_temperature=0.05, ss_l2_norm=True,
    ss_sampling_strategy="local", dataset_folder="dataset/amazon",
    split="beauty", dataset_type="amazon", do_eval=True,
    eval_every_epoch=10, eval_batch_size=256, patience=50,
    save_dir_root="out/hstu/amazon/beauty", save_every_epoch=50,
    wandb_logging=False, amp=True, mixed_precision_type="bf16", seed=42,
)
```

`dataset_type` must be `amazon` or `ml1m`, and `loss_type` must be `ce` or
`sampled_softmax`. For sampled softmax, `local` is Meta's local-table sampler:
each positive independently draws catalog IDs with replacement. `in_batch` is
a separate sampler, while `global` preserves GenRec's legacy shared,
deduplicated catalog-negative set. Unknown values fail fast.

The Amazon 2023 baseline configs still use `AmazonSASRecDataset`/`AmazonHSTUDataset` over converted 2014-compatible gzip files. Run `prepare_amazon2023()` first; current filename mappings cover `books` and `arts_crafts_and_sewing`.

## Quantizers

RQ-VAE's complete signature and artifacts are in [RQ-VAE API](rqvae.md#trainer). Exactly one of `epochs` and `iterations` must be set.

```python
fit_rqkmeans(
    dataset=AmazonItemDataset, dataset_folder="dataset/amazon",
    encoder_model_name="./models_hub/Qwen3-Embedding-8B",
    n_layers=3, codebook_size=8192,
    save_dir_root="out/onerec/amazon/{split}/rqkmeans",
    random_state=42, max_iter=100, batch_size=4096, seed=42,
)
```

The dataset must expose float NumPy `embeddings[N,D]`. This function writes `rqkmeans.pt` and `results.json`. Current configs override the Python default K=8192 to K=256.

## Semantic-ID recommenders

TIGER's complete contract is in [TIGER API](tiger.md#trainer).

```python
# lcrec_trainer.py (principal arguments)
train(
    epochs=4, batch_size=8, learning_rate=5e-5, weight_decay=0.01,
    warmup_ratio=0.01, gradient_accumulate_every=2, max_length=512,
    pretrained_path="Qwen/Qwen2.5-1.5B", use_lora=True,
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    num_codebooks=5, codebook_size=256, dataset=None,
    dataset_folder="dataset/amazon", max_seq_len=20, max_text_len=128,
    pretrained_rqvae_path="./out/lcrec/amazon/beauty/rqvae/checkpoint.pt",
    do_eval=True, eval_every_epoch=1, eval_batch_size=64,
    eval_beam_width=10, save_dir_root="out/lcrec/amazon/beauty",
    eval_only=False, checkpoint_path=None, seed=42, ...,
)
```

LCRec requires a Gin-injected dataset. `checkpoint_path` is an LCRec `save_pretrained()` directory, not an optimizer checkpoint. `eval_only=True` evaluates validation only.

```python
# cobra_trainer.py (principal arguments)
train(
    epochs=100, batch_size=32, learning_rate=1e-4, weight_decay=0.01,
    dataset_folder="dataset/amazon", dataset=None,
    save_dir_root="out/cobra/amazon/beauty",
    encoder_n_layers=1, encoder_hidden_dim=768, encoder_num_heads=8,
    encoder_vocab_size=32128, id_vocab_size=256, n_codebooks=3,
    d_model=384, max_len=1024, temperature=0.2, queue_size=1024,
    decoder_n_layers=8, decoder_num_heads=6, decoder_dropout=0.1,
    encoder_type="light", num_warmup_steps=500, max_seq_len=20,
    pretrained_rqvae_path="./out/rqvae/amazon/beauty/checkpoint.pt",
    encoder_model_name="sentence-transformers/sentence-t5-xl",
    resume_from_checkpoint=None, sparse_loss_weight=1.0,
    dense_loss_weight=1.0, use_dense=True, seed=42, ...,
)
```

COBRA also requires dataset injection. Its resume file is a COBRA dict containing model, optimizer, scheduler, and epoch. Only `encoder_type="pretrained"` selects `SentenceT5Encoder`; every other string falls back to `LightT5Encoder`. `eval_test_every_epoch` is retained for config compatibility; benchmark evaluation now selects on validation and runs a one-shot test from `best_model.pt`.

## OneRec

```python
train_sft(
    epochs=4, batch_size=8, learning_rate=3e-4, weight_decay=0.01,
    warmup_steps=20, warmup_ratio=0.0, adam_beta2=0.999,
    gradient_accumulate_every=2, max_length=512,
    pretrained_path="Qwen/Qwen2.5-1.5B", use_lora=True,
    num_codebooks=3, codebook_size=8192, dataset=None,
    dataset_folder="dataset/amazon2023", max_seq_len=20,
    do_eval=True, eval_batch_size=64, eval_beam_width=10,
    save_dir_root="out/onerec/amazon2023/{split}/sft",
    eval_only=False, checkpoint_path=None, early_stopping_patience=3,
    attention_dropout=0.0, label_smoothing=0.0,
    enabled_tasks=None, seed=42, ...,
)
```

Repository configs override the old Python K=8192/Amazon-2023 defaults to Amazon 2014, K=256, and RQ-KMeans. The function's `enabled_tasks` value is recorded but actual task selection is bound on the dataset.

```python
train_grpo(
    epochs=2, batch_size=4, learning_rate=1e-5, weight_decay=0.01,
    warmup_steps=10, gradient_accumulate_every=4,
    max_length=512, max_grad_norm=0.3, num_generations=16,
    temperature=1.0, beta=1e-3, clip_eps=0.2,
    reward_type="exact", use_beam_search=False, add_gt=False,
    num_codebooks=3, codebook_size=8192,
    sft_checkpoint_path="./out/onerec/amazon2023/{split}/sft/checkpoint_final",
    rl_dataset=None, eval_dataset=None,
    dataset_folder="dataset/amazon2023", max_seq_len=20,
    do_eval=True, eval_batch_size=4, eval_beam_width=10,
    save_dir_root="out/onerec/amazon2023/{split}/grpo", seed=42, ...,
)
```

`rl_dataset` is required. With no `eval_dataset`, `do_eval=True` still creates no evaluation loaders. Sampling uses OneRec's catalog `PrefixTrie`; valid reward types are `exact`, `hierarchical`, and `rank`.

```python
train_clm(
    epochs=5, batch_size=32, learning_rate=5e-5, weight_decay=0.01,
    warmup_steps=50, gradient_accumulate_every=2, max_length=256,
    pretrained_path="Qwen/Qwen2.5-1.5B", use_lora=False,
    num_codebooks=3, codebook_size=256, dataset=None,
    dataset_folder="dataset/amazon", max_seq_len=50,
    do_eval=True, eval_batch_size=64, eval_beam_width=10,
    save_dir_root="out/onerec/amazon/{split}/clm",
    eval_only=False, checkpoint_path=None,
    early_stopping_patience=3, seed=42, ...,
)
```

CLM falls back to `AmazonOneRecCLMDataset` when no class is injected. It adds code tokens and `<|item_sep|>` to the tokenizer, so checkpoints must preserve that tokenizer.

## RPG, NoteLLM, and GRAM

```python
train_rpg(
    n_embd=448, n_layer=2, n_head=4, n_inner=1024,
    n_codebook=32, codebook_size=256, temperature=0.07,
    embd_pdrop=0.5, attn_pdrop=0.5, resid_pdrop=0.0,
    num_beams=50, n_edges=50, propagation_steps=3, chunk_size=1024,
    epochs=150, batch_size=256, learning_rate=3e-4,
    weight_decay=0.01, warmup_steps=10000, max_grad_norm=1.0,
    max_seq_len=50, dataset_folder="dataset/amazon", split="beauty",
    encoder_model_name="sentence-transformers/sentence-t5-xl",
    sent_emb_pca=0, do_eval=True, eval_every_epoch=1,
    eval_batch_size=256, eval_with_graph=False, patience=20,
    save_dir_root="out/rpg/amazon/{split}", seed=42, ...,
)
```

RPG fixes `AmazonRPGDataset`. Item-level NDCG@10 selects the checkpoint; direct decoding is the headline result and optional graph metrics live under `results.json.extra`.

```python
# notellm_trainer.py
train(
    epochs=5, batch_size=4, gradient_accumulate_every=8,
    learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.05,
    pretrained_path="Qwen/Qwen2.5-1.5B", freeze_lm=False,
    use_lora=True, lora_r=16, lora_alpha=32, lora_dropout=0.05,
    dataset_folder="dataset/amazon", split="beauty",
    max_seq_len=50, max_text_length=256,
    eval_batch_size=32, catalog_batch_size=64, patience=2,
    enable_csft=True, csft_alpha=0.01,
    save_dir_root="out/notellm/amazon/{split}",
    amp=True, mixed_precision_type="bf16", seed=42, ...,
)
```

NoteLLM fixes `AmazonNoteLLMDataset`, combines symmetric GCL with optional CSFT,
and evaluates exact full-catalog embedding retrieval after excluding the query
item. Repeated catalog items are masked as GCL false negatives. It raises
`RuntimeError` if LoRA is requested without PEFT. The shipped config binds both
the trainer and dataset split, so the standard `--split sports` route is
consistent.

```python
train_gram(
    backbone="t5-small", max_his=20, use_position_embedding=True,
    epochs=30, batch_size=32, learning_rate=1e-3,
    dataset=None, dataset_folder="dataset/gram", split="Beauty",
    item_prompt_max_len=128, target_max_len=50,
    item_id_file="item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt",
    item_prompt="all_text", top_k_similar_item=10, cf_model="sasrec",
    num_beams=50, eval_batch_size=1, length_penalty=0.0,
    metrics="hit@5,hit@10,ndcg@5,ndcg@10",
    save_dir_root="out/gram/{split}", eval_only=False,
    checkpoint_path=None, seed=42, ...,
)
```

GRAM falls back to `AmazonGRAMDataset`. Every comma-separated metric must include `@K`; candidate-trie items come from `train_ds.all_items`. `checkpoint_path` is a raw state dict.

## Collators, evaluation, and GRPO rewards

There is no common helper protocol. Key functions are:

- `sasrec_trainer.evaluate(...)`, `hstu_trainer.evaluate(...)`, `rpg_trainer.evaluate(...)`, and `notellm_trainer.evaluate_full_catalog(...)` return model-specific item metrics.
- `cobra_collate_fn(...)`, `lcrec_collate_fn(...)`, `onerec_collate_fn(...)`, `rl_collate_fn(...)`, CLM collators, `GRAMCollator`, and `t5_collate(...)` each expose different dictionaries.
- LCRec's `ConstrainedDecodingHelper` masks codebook position ranges but does not guarantee a catalog SID tuple.
- OneRec GRPO's `PrefixTrie` performs catalog-tuple constraints. This does not apply to TIGER.

```python
compute_rule_reward(generated_sem_ids, target_sem_ids) -> list[float]
compute_hierarchical_reward(generated_sem_ids, target_sem_ids,
                            num_codebooks) -> list[float]
compute_rank_reward(generated_sem_ids, target_sem_ids,
                    num_codebooks) -> list[float]
```

Rule reward is exact 0/1. Hierarchical reward is `10**(m-1)` for a matching prefix of length `m`. Rank reward returns an all-zero group when no exact hit exists; otherwise it adds penalties to wrong candidates. Its `num_codebooks` parameter is currently unused.

## Shared utilities

[`trainer_utils.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/trainer_utils.py) provides:

```python
set_seed(seed: int, deterministic: bool = False) -> None
get_git_sha() -> str
setup_accelerator(split_batches=True, gradient_accumulate_every=1,
                  amp=False, mixed_precision_type="fp16")
setup_wandb(project, run_name=None, config=None, step_metrics=None) -> None
get_parameter_count(model) -> int
save_run_results(save_dir, model, split, seed, metrics,
                 config=None, extra=None) -> str
save_checkpoint(path, model, optimizer, scheduler=None,
                epoch=None, step=None, extra=None, logger=None) -> None
load_checkpoint(path, model, optimizer=None, scheduler=None,
                device=torch.device("cpu"), logger=None) -> dict
```

`TrainerConfig` is only a convenience dataclass; no trainer inherits or automatically consumes it. `save_run_results` overwrites `results.json` and includes model, split, seed, Git/source metadata, UTC timestamp, metrics, config, and optional execution/extra objects.

## Checkpoint formats are not interchangeable

| Trainer | Artifact | Optimizer-resumable? |
|---|---|---|
| SASRec / HSTU / RPG / GRAM / TIGER | raw model `state_dict` | No |
| COBRA | dict with epoch/model/optimizer/scheduler | Yes, via COBRA path |
| RQ-VAE | dict with epoch-or-iter/model/model_config/optimizer/scheduler | Yes |
| RQ-KMeans | `RqKmeans.save()` artifact | Not applicable |
| LCRec / OneRec SFT/GRPO/CLM | Hugging Face model/tokenizer directory | No |
| NoteLLM | trainable-state dict plus tokenizer and best metadata | No; reconstruct base model |

LCRec/SFT/CLM can warn and silently fall back to full-parameter training when LoRA lacks PEFT; NoteLLM refuses. RPG OPQ needs FAISS, RQ-KMeans needs scikit-learn, and LLM routes need Transformers and weights.

## CLI

```bash
python genrec/trainers/sasrec_trainer.py config/sasrec/amazon.gin --split beauty
python genrec/trainers/hstu_trainer.py config/hstu/amazon.gin --split beauty
python genrec/trainers/rqvae_trainer.py config/tiger/amazon/rqvae.gin --split beauty
python genrec/trainers/tiger_trainer.py config/tiger/amazon/tiger.gin --split beauty
python genrec/trainers/lcrec_trainer.py config/lcrec/amazon/lcrec.gin --split beauty
python genrec/trainers/cobra_trainer.py config/cobra/amazon/cobra.gin --split beauty
python genrec/trainers/rqkmeans_trainer.py config/onerec/amazon/rqkmeans.gin --split beauty
python genrec/trainers/onerec_sft_trainer.py config/onerec/amazon/sft.gin --split beauty
python genrec/trainers/onerec_grpo_trainer.py config/onerec/amazon/grpo.gin --split beauty
python genrec/trainers/onerec_clm_trainer.py config/onerec/amazon/clm.gin --split beauty
python -m genrec.trainers.rpg_trainer config/rpg/amazon.gin --split beauty
python genrec/trainers/notellm_trainer.py config/notellm/amazon.gin --split beauty
python genrec/trainers/gram_trainer.py config/gram/amazon/gram.gin --split Beauty
```

See [Configuration and CLI](configs.md) before adding a new entry point. A new trainer must declare its own Gin arguments, dataset/collator schema, distributed collectives, checkpoint format, and item-level evaluation semantics.
