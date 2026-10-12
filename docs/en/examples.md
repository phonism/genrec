# Runnable examples

Trainer + Gin is the supported execution path. Handwritten training loops easily omit checkpoint selection, evaluation semantics, or provenance, so CLI examples come first. Direct Python snippets below are for API orientation only.

## Single-model smoke test

```bash
python genrec/trainers/sasrec_trainer.py \
  config/sasrec/amazon.gin --split beauty \
  --gin 'train.epochs=1' \
  --gin 'train.patience=1' \
  --gin 'train.wandb_logging=False' \
  --gin 'train.save_dir_root="out/_smoke/sasrec/beauty"'
```

A smoke test checks the code path, not model quality. Never reuse an `_smoke` checkpoint as a benchmark artifact.

## Multiple Gin overrides

```bash
python genrec/trainers/hstu_trainer.py \
  config/hstu/amazon.gin --split sports \
  --gin 'train.seed=7' \
  --gin 'train.loss_type="sampled_softmax"' \
  --gin 'train.ss_sampling_strategy="local"' \
  --gin 'train.num_negatives=256' \
  --gin 'train.save_dir_root="out/hstu_ss/amazon/sports/s7"'
```

For real sampled-softmax work, start from `config/hstu/amazon_ss.gin`; the long override list mainly demonstrates syntax.

## Change the evaluation protocol

```bash
python genrec/trainers/sasrec_trainer.py \
  config/sasrec/amazon.gin --split beauty \
  --gin 'AmazonSASRecDataset.evaluation_protocol="global_temporal"' \
  --gin 'train.seed=42' \
  --gin 'train.save_dir_root="out/sasrec/amazon/beauty/global_temporal/s42"'
```

Include the protocol in the output path so it cannot overwrite a leave-one-out `results.json`.

## Inspect tokenizer health

```bash
python scripts/verify_codebook_health.py \
  --kind rqkmeans \
  --checkpoint out/onerec/amazon/beauty/rqkmeans/rqkmeans.pt \
  --dataset-folder dataset/amazon \
  --split beauty \
  --encoder /data/models/Qwen3-Embedding-8B
```

Use `--kind rqvae` for RQ-VAE. Inspect per-layer usage, dead codes, maximum frequency, and full-SID collisions together.

## Dry-run a result matrix

```bash
python scripts/run_matrix.py --help
python scripts/run_matrix.py \
  --models sasrec hstu \
  --splits beauty sports toys \
  --seeds 42 43 44 \
  --dry-run

python scripts/validate_matrix_results.py --help
python scripts/aggregate_results.py --help
```

Treat each script's local `--help` as authoritative. The validator rejects headline artifacts with incomplete protocol, metric-level, or source provenance.

## Call SASRec directly

```python
import torch
from genrec.models.sasrec import SASRec

model = SASRec(
    num_items=100,
    max_seq_len=5,
    embed_dim=16,
    num_heads=2,
    num_blocks=2,
    ffn_dim=64,
    dropout=0.0,
    loss_type="ce",
)

input_ids = torch.tensor([[0, 0, 4, 8, 3]])
targets = torch.tensor([[0, 0, 8, 3, 9]])
logits, loss = model(input_ids, targets)
top5 = model.predict(input_ids, top_k=5)

print(logits.shape)  # [1, 5, 101]
print(float(loss), top5)
```

Use the trainer's full-dataloader evaluation for real metrics; this snippet has no data protocol or result writer.

## Call HSTU directly

```python
import torch
from genrec.models.hstu import HSTU

model = HSTU(
    num_items=100,
    max_seq_len=5,
    embed_dim=16,
    num_heads=2,
    num_blocks=1,
    dropout=0.0,
    use_temporal_bias=True,
)

items = torch.tensor([[0, 2, 5, 7, 8]])
times = torch.tensor([[0, 100, 200, 260, 400]])
logits, _ = model(items, times)
print(logits.shape)  # [1, 5, 101]
```

Sampled-softmax score normalization is constructed by the trainer; do not toggle one model argument and omit the associated normalization and temperature settings.

## Load RQ-VAE and generate SIDs

```python
import numpy as np
import pandas as pd
import torch
from genrec.models.rqvae import RqVae

df = pd.read_parquet(
    "dataset/amazon/processed/beauty/item_emb_sentence-t5-xl.parquet"
)
x = torch.tensor(np.stack(df["embedding"].values), dtype=torch.float32)

model = RqVae.from_pretrained(
    "out/tiger/amazon/beauty/rqvae/checkpoint_epoch_4999.pt",
    map_location="cpu",
).eval()

with torch.no_grad():
    sid = model.get_semantic_ids(x).sem_ids
print(sid.shape)  # [num_items, num_codebooks]
```

Use the exact restoration and semantic-ID methods shown above, and keep the item mapping with the SID array.

## Resolve SID collisions honestly

```python
import torch
from genrec.modules.metrics import SemanticIdItemMapper

catalog_sids = [[1, 2], [1, 2], [3, 4]]
training_sequences = [[0, 0, 1, 2], [1, 2, 2, 0]]
mapper = SemanticIdItemMapper.from_sequences(
    catalog_sids,
    training_sequences,
    train_holdout=2,
)

beam = torch.tensor([[[1, 2], [1, 2], [3, 4], [9, 9]]])
items = mapper.map_predictions(beam)
print(items)
```

Duplicate SIDs resolve to the same representative item and are deduplicated. Unknown SIDs are dropped and missing positions are filled with `-1`.

## Inspect `results.json`

```python
import json
from pathlib import Path

payload = json.loads(Path("out/sasrec/amazon/beauty/results.json").read_text())
for key in ["model", "split", "seed", "metrics", "config"]:
    print(key, payload.get(key))
```

The real schema also carries source provenance. Do not use this read-only snippet to write results; generation goes through `save_run_results`.

## OneRec stages

```bash
python genrec/trainers/rqkmeans_trainer.py \
  config/onerec/amazon/rqkmeans.gin --split beauty

accelerate launch --config_file config/accelerate_4gpu.yaml \
  -m genrec.trainers.onerec_sft_trainer \
  config/onerec/amazon/sft.gin --split beauty

accelerate launch --config_file config/accelerate_4gpu.yaml \
  -m genrec.trainers.onerec_grpo_trainer \
  config/onerec/amazon/grpo.gin --split beauty
```

Before GRPO, run `python scripts/check_grpo_readiness.py --help` to verify the SFT checkpoint, tokenizer, and reward sparsity.

## Next steps

- [Complete pipelines](getting-started.md)
- [Configuration system](architecture/configuration.md)
- [Evaluation system](architecture/evaluation-system.md)
- [Extension guide](architecture/extension-guide.md)
