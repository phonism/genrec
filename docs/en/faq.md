# Frequently asked questions

## Installation and environment

### Which Python and PyTorch versions are supported?

`pyproject.toml` requires Python `>=3.9` and PyTorch `>=2.0`. Choose a CUDA wheel compatible with the machine's driver; the primary experiment environment is Linux with NVIDIA GPUs.

### Why does `config/base.gin` point into `/root/workspace/models_hub`?

Those are local mirror paths on the primary 179 host. Override the `MODEL_HUB_*` Gin constants elsewhere. The config parser does not automatically expand a generic model-hub environment variable.

### Why is PEFT missing for NoteLLM?

The default config enables LoRA, while PEFT is in the `full` optional dependency group. Install `pip install -e ".[full]"` or explicitly disable LoRA; those are different experiment configurations.

### Why is FAISS missing for RPG?

Install `pip install -e ".[rpg]"`. The group currently pins `faiss-cpu==1.10.0` to match the reproduction environment.

## Data

### Is data downloaded automatically?

Not uniformly. `AmazonItemDataset` can download raw review/metadata on demand, while SASRec and HSTU adapters read local files directly. On shared hosts, use `scripts/prepare_amazon2014.py` for locking, CRC checks, and SHA-256 manifests.

### Why can Amazon 2023 still read `reviews_*_5.json.gz`?

`genrec.data.amazon2023` downloads 2023 JSONL, applies iterative k-core filtering, and emits gzip files compatible with the older adapters. Run the preparation step first; changing only `dataset_folder` does not perform the conversion. The current Amazon 2023 SASRec/HSTU configs deliberately reuse the compatible 2014-style adapters.

### Can integer item IDs be shared across models?

Do not assume so. SASRec, HSTU, and RPG use 1-based IDs with padding 0; TIGER, LCRec, OneRec, and COBRA generally use 0-based IDs; GRAM uses string IDs. Mapping can also depend on source record order.

### Is `leave_one_out` better than `global_temporal`?

They answer different questions. Leave-one-out supports traditional reproduction; global temporal more strongly prevents future interactions from entering training histories. Report them separately and never average across protocols.

## Configuration

### What does `--split` do?

`parse_config` replaces literal `{split}` text in the Gin file, parses repeated `--gin` overrides, and stores the split in process context for result metadata.

### Why does `--gin 'train.xxx=...'` have no effect?

Check the real configurable name. OneRec uses `train_sft`, `train_clm`, and `train_grpo`; RPG uses `train_rpg`; GRAM uses `train_gram`. Use the selector shown on the left side of the config binding.

### Which wins: Python defaults or Gin values?

Gin overrides Python defaults at execution time. Reproduction is defined by the resolved config, CLI overrides, and `results.json`, not by an example value in prose.

### Does NoteLLM need an extra split override?

No. Its config binds `{split}` to both the trainer and dataset class, so
`--split sports` updates the data, output path, and result provenance together.

## Models and tokenizers

### Are RQ-VAE and RQ-KMeans recommenders?

No. They tokenize item embeddings into semantic IDs. Evaluate their reconstruction, codebook health, and collisions; do not report standalone recommendation Recall for them.

### Why can K=8192 collapse?

In a catalog with only tens of thousands of items, later residual layers are shallow and an oversized MiniBatchKMeans codebook can develop dead clusters and a super-cluster. Current small-catalog OneRec configs use K=256 and include dead-cluster reinitialization. Per-layer usage is not the same as full-SID uniqueness.

### Does TIGER use a Trie?

No. The current `Tiger.generate()` performs ordinary beam search. The item mapper drops illegal or out-of-catalog SIDs. Any older claim of Trie-constrained TIGER decoding described an API that does not exist here.

### How do OneRec SFT, CLM, and GRPO relate?

RQ-KMeans is the shared prerequisite. SFT and CLM are alternative training-data organizations; CLM is not a mandatory stage after SFT. GRPO depends on an SFT checkpoint.

### Why do many GRPO steps have no gradient?

With low SFT top-1 accuracy and a finite group size, a group often contains no positive sample. Returning zero rewards in that case is a hard-sample skip, not a reward bug. Compare `add_gt`, larger groups, or denser rewards as isolated experiments rather than silently changing semantics.

### Why can COBRA be implemented yet underperform the paper?

Source code presence and paper-result reproduction are separate claims. Current item-level results remain below the paper; the [COBRA page](models/cobra.md) records the diagnosis and open gaps.

### Is NoteLLM a semantic-ID generator?

Not in this repository. It extracts the LLM hidden state immediately before
`[EMB]` and performs dense item retrieval. In-batch recall is diagnostic;
formal evaluation uses exact full-catalog retrieval.

## Metrics

### Why can SID Recall exceed item Recall?

Multiple items may share a complete SID. Tuple-level matching then treats any item in the bucket as the target. Headline metrics must resolve with training-only SID→item statistics and deduplicate item candidates.

### Why can deduplication change Recall nonlinearly?

Several beams may resolve to one item. Removing duplicates pulls deeper beams into the top K, so item@K is not a fixed scaling of SID@K.

### Can NoteLLM in-batch Recall be compared with SASRec?

No. Its candidate set is only the batch. Only the full-catalog `compute_full_catalog_metrics` path belongs in a common headline table.

### Are Recall@K and HR@K the same here?

They are numerically equal under the current one-target next-item protocol. Documentation uses “Recall” so the name remains meaningful if multi-target evaluation is added.

## Training and results

### How is the best checkpoint chosen?

Most recommenders use validation Recall@10, but inspect the trainer. For SID models, verify that selection uses item-level metrics rather than a legacy tuple-level metric.

### Why does a README number not match `best_model.pt`?

A smoke run may have overwritten a checkpoint, or the number may use a different seed, protocol, source snapshot, or metric level. Check exact provenance in `results.json` and logs rather than inferring from a filename.

### How should I run a three-seed matrix?

Dry-run `scripts/run_matrix.py`, submit through the 179 locked workflow, then validate and aggregate. Do not manually launch competing commands against the same GPU or experiment cell.

### May I kill an unfamiliar GPU process?

No. Inspect both the remote task list and GPU processes. `run-locked-179.sh` reacquires locks and refuses unsafe sharing; an unknown process may belong to another session.

## Documentation

### Why are some English pages missing?

Chinese is the canonical implementation and landscape documentation. The i18n site falls back to Chinese for pages without an English translation; this does not imply complete bilingual parity.

### How should I report a documentation error?

Give the page, exact statement, corresponding source or paper evidence, and verification date. Follow the [documentation policy](reference/documentation-policy.md).
