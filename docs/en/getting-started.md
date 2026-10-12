# Getting started

This path produces one auditable SASRec run before introducing the dependencies of generative pipelines. Run commands from the repository root.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For CUDA, optional dependencies, and local model mirrors, see [Installation](installation.md).

## 2. Prepare Amazon 2014 Beauty

SASRec and HSTU read the raw gzip files directly, so prepare them explicitly:

```bash
python scripts/prepare_amazon2014.py \
  --splits beauty --kinds reviews meta
```

Expected files:

```text
dataset/amazon/raw/beauty/
├── reviews_Beauty_5.json.gz
└── meta_Beauty.json.gz
```

The preparation script verifies gzip CRC and SHA-256 and writes atomically. Existing files are revalidated.

## 3. Run a baseline

```bash
python genrec/trainers/sasrec_trainer.py \
  config/sasrec/amazon.gin --split beauty \
  --gin 'train.wandb_logging=False'
```

The default may train for up to 300 epochs with early stopping on validation Recall@10. For a code-path smoke test, isolate the output and shorten the run:

```bash
python genrec/trainers/sasrec_trainer.py \
  config/sasrec/amazon.gin --split beauty \
  --gin 'train.epochs=1' \
  --gin 'train.patience=1' \
  --gin 'train.wandb_logging=False' \
  --gin 'train.save_dir_root="out/_smoke/sasrec/beauty"'
```

Do not publish smoke-test metrics. A normal output directory contains `best_model.pt`, a timestamped log, and `results.json`; the JSON artifact, not a copied console number, is the aggregation input.

## 4. Choose a pipeline

### SASRec / HSTU

```bash
python genrec/trainers/sasrec_trainer.py config/sasrec/amazon.gin --split beauty
python genrec/trainers/hstu_trainer.py config/hstu/amazon.gin --split beauty
python genrec/trainers/hstu_trainer.py config/hstu/amazon_ss.gin --split beauty
```

HSTU cross-entropy and sampled-softmax runs are different experiments; keep their outputs and labels separate.

### RQ-VAE → TIGER / LCRec / COBRA

Each downstream model has its own RQ-VAE geometry. Train and load the matching checkpoint:

```bash
# TIGER
python genrec/trainers/rqvae_trainer.py config/tiger/amazon/rqvae.gin --split beauty
python genrec/trainers/tiger_trainer.py config/tiger/amazon/tiger.gin --split beauty

# LCRec
python genrec/trainers/rqvae_trainer.py config/lcrec/amazon/rqvae.gin --split beauty
python genrec/trainers/lcrec_trainer.py config/lcrec/amazon/lcrec.gin --split beauty

# COBRA
python genrec/trainers/rqvae_trainer.py config/cobra/amazon/rqvae.gin --split beauty
python genrec/trainers/cobra_trainer.py config/cobra/amazon/cobra.gin --split beauty
```

Check the Sentence-T5 path in `config/base.gin`. Do not, for example, pass TIGER's 3×256 tokenizer checkpoint to a COBRA 3×32 pipeline.

TIGER currently uses ordinary beam search. It does **not** use Trie-constrained decoding; catalog-invalid SIDs are removed during item mapping.

### RQ-KMeans → OneRec

```bash
# Stage 1: item embeddings + 3×256 residual K-Means
python genrec/trainers/rqkmeans_trainer.py \
  config/onerec/amazon/rqkmeans.gin --split beauty

# Stage 2a: multi-task SFT
accelerate launch --config_file config/accelerate_4gpu.yaml \
  -m genrec.trainers.onerec_sft_trainer \
  config/onerec/amazon/sft.gin --split beauty

# Stage 2b: alternative CLM training, not a required post-SFT stage
python -m genrec.trainers.onerec_clm_trainer \
  config/onerec/amazon/clm.gin --split beauty

# Stage 3: GRPO, which depends on an SFT checkpoint
accelerate launch --config_file config/accelerate_4gpu.yaml \
  -m genrec.trainers.onerec_grpo_trainer \
  config/onerec/amazon/grpo.gin --split beauty
```

The default pipeline expects local `Qwen3-Embedding-8B` and `Qwen3-1.7B` assets. Read the [OneRec page](models/onerec.md) before changing rewards, codebook size, collision policy, or checkpoint paths.

### RPG / NoteLLM / GRAM

```bash
python -m genrec.trainers.rpg_trainer \
  config/rpg/amazon.gin --split beauty

python -m genrec.trainers.notellm_trainer \
  config/notellm/amazon.gin --split beauty

python -m genrec.trainers.gram_trainer \
  config/gram/amazon/gram.gin --split Beauty
```

RPG requires FAISS. NoteLLM's default LoRA path requires PEFT; its config binds
`{split}` to both the trainer and dataset, so the standard `--split` flag is
sufficient. GRAM consumes paper-style preprocessed assets under `dataset/gram`;
the regular Amazon adapter does not build every lexical-index and similar-item
artifact.

## 5. Gin overrides

The first positional argument is a Gin config. `--split` first replaces literal `{split}` placeholders in its text; repeated `--gin` arguments are then parsed as overrides:

```bash
python genrec/trainers/hstu_trainer.py config/hstu/amazon.gin \
  --split sports \
  --gin 'train.seed=7' \
  --gin 'train.batch_size=64' \
  --gin 'train.wandb_logging=False'
```

The configurable is not always named `train`: OneRec uses `train_sft`, `train_clm`, and `train_grpo`; RPG uses `train_rpg`; GRAM uses `train_gram`. Copy the selector from the actual config.

## 6. Evaluation contract

The default protocol is `leave_one_out`; compatible Amazon adapters also support `global_temporal`. Separate results by protocol, seed, and metric level.

For semantic-ID generators, only item-level metrics are suitable as headlines:

```text
beam SIDs
  → training-only most-popular SID→item resolution
  → item deduplication
  → Recall@K / NDCG@K
```

SID-tuple exact match, NoteLLM in-batch recall, and sampled-negative metrics are diagnostics, not cross-model headline results. See [Evaluation system](architecture/evaluation-system.md).

## 7. Before a matrix run

```bash
python scripts/run_matrix.py --help
python scripts/validate_matrix_results.py --help
python scripts/aggregate_results.py --help
```

Dry-run the matrix before using a remote experiment cell. Runs on the primary 179 host must follow the [concurrency-safe reproduction runbook](179-reproduction-runbook.md).

## Next steps

- [Model overview](models/overview.md)
- [Architecture](architecture/index.md)
- [GenRec research landscape](landscape/index.md)
- [Extension guide](architecture/extension-guide.md)
- [Implementation and reproduction status](reference/model-status.md)
