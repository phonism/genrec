# Gin configuration and CLI

GenRec uses [Gin-config](https://github.com/google/gin-config). There are no `DatasetConfig`, `TextEncodingConfig`, or `TrainingConfig` objects and no YAML/JSON configuration loader. Configuration binds functions and classes decorated with `@gin.configurable`.

## `parse_config`

Source: [`genrec/modules/utils.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/utils.py)

```python
parse_config() -> tuple[str, str]
```

The command line is:

```text
trainer.py CONFIG.gin [--split beauty] [--gin BINDING ...]
```

`parse_config()` reads the Gin file, replaces every literal `{split}` with the selected split, parses the file, then parses the repeatable CLI bindings. It returns `(config_path, split)`. A missing config path or file exits through `argparse`/`FileNotFoundError`; Gin syntax and unknown selectors propagate Gin errors.

```bash
python genrec/trainers/hstu_trainer.py config/hstu/amazon.gin \
  --split sports \
  --gin 'train.epochs=100' \
  --gin 'train.use_temporal_bias=False'
```

The replacement is textual, not a safe formatting language: any unrelated `{split}` in prompt or regex text is replaced too.

## Bindings

Function parameters:

```gin
train.epochs = 200
train_sft.learning_rate = 8e-5
train_rpg.n_codebook = 32
fit_rqkmeans.codebook_size = 256
```

Class parameters:

```gin
AmazonSeqDataset.split = "{split}"
AmazonSeqDataset.evaluation_protocol = "leave_one_out"
```

An explicit constructor keyword supplied by a trainer wins over a class binding.

Callable references and macros:

```gin
include "config/base.gin"
import genrec.data.amazon

train.dataset = @AmazonSeqDataset
AmazonSeqDataset.encoder_model_name = %MODEL_HUB_SENTENCE_T5_XL
```

The imported module must register the configurable before `@Name` is resolved. Include paths assume that commands run from the repository root.

## Model-hub macros

[`config/base.gin`](https://github.com/phonism/genrec/blob/main/config/base.gin) defines five absolute-path macros: `MODEL_HUB_SENTENCE_T5_BASE`, `MODEL_HUB_SENTENCE_T5_XL`, `MODEL_HUB_QWEN3_EMBEDDING_8B`, `MODEL_HUB_QWEN3_1_7B`, and `MODEL_HUB_QWEN3_4B`.

`parse_config()` does not expand a `MODEL_HUB` environment variable or `${MODEL_HUB}`. Rebind a macro when running elsewhere:

```bash
--gin 'MODEL_HUB_QWEN3_1_7B="/data/models/Qwen3-1.7B"'
```

## Configuration families

| Path | Selector | Purpose |
|---|---|---|
| `config/sasrec/` | `train` | SASRec |
| `config/hstu/` | `train` | HSTU CE or sampled-softmax |
| `config/tiger/amazon/` | `train` | RQ-VAE and TIGER stages |
| `config/lcrec/` | `train` | RQ-VAE and LCRec |
| `config/cobra/` | `train` | RQ-VAE and sparse/mixed COBRA |
| `config/onerec/amazon/` | `fit_rqkmeans`, `train_sft`, `train_grpo`, `train_clm` | OneRec stages |
| `config/rpg/` | `train_rpg` | RPG, split-specific |
| `config/notellm/amazon.gin` | `train` | NoteLLM symmetric GCL + optional CSFT |
| `config/gram/amazon/gram.gin` | `train_gram` | GRAM |

The Amazon 2023 SASRec/HSTU configs still construct the 2014-style `AmazonSASRecDataset`/`AmazonHSTUDataset`. They consume compatible gzip files written by `prepare_amazon2023()`. Among Amazon 2023 categories, `DATASET_CONFIGS` currently maps `books` and `arts_crafts_and_sewing`; another 2023 category requires an additional filename mapping.

## Enum constants

`QuantizeForwardMode` is registered with `@gin.constants_from_enum`:

```gin
train.vae_codebook_mode = %genrec.models.rqvae.QuantizeForwardMode.GUMBEL_SOFTMAX
train.vae_codebook_last_layer_mode = %genrec.models.rqvae.QuantizeForwardMode.SINKHORN
```

Valid members are `GUMBEL_SOFTMAX`, `STE`, `ROTATION_TRICK`, and `SINKHORN`. `QuantizeDistance` is not registered in the same way.

## Common failures

- `No configurable matching ...`: missing `import`, misspelled selector, or the config was passed to the wrong trainer.
- An override has no effect: distinguish `train`, `train_sft`, `train_grpo`, and the other entry names.
- Paths still point at a different split: inspect hard-coded strings and `{split}` use.
- A list is split by the shell: quote the entire binding.
- A checkpoint loads but represents the wrong encoder/catalog: Gin checks shapes, not artifact provenance.
- `load_config_from_file()` is requested: that API does not exist here.
- A split appears inconsistent: inspect the resolved config and `results.json`.
  NoteLLM now binds both `train.split` and `AmazonNoteLLMDataset.split`, so the
  standard `--split sports` command updates both.

See [trainer signatures](trainers.md), [data routing](dataset-factory.md), and [configuration architecture](../architecture/configuration.md).
