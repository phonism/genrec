# API reference

This section indexes the current source code. It is not a stable public SDK. GenRec is a research repository: most models, data adapters, and trainers evolve independently, and there is no shared `BaseModel`, `BaseDataset`, `DatasetFactory`, or `BaseTrainer`.

## Navigation

| Need | Page | Source |
|---|---|---|
| Dataset classes and sample schemas | [Datasets](datasets.md) | [`genrec/data/`](https://github.com/phonism/genrec/tree/main/genrec/data) |
| Shared split and sequence contracts | [Sequence data contracts](base-dataset.md) | [`evaluation_protocol.py`](https://github.com/phonism/genrec/blob/main/genrec/data/evaluation_protocol.py), [`schemas.py`](https://github.com/phonism/genrec/blob/main/genrec/data/schemas.py) |
| Gin configuration and CLI overrides | [Configuration and CLI](configs.md) | [`parse_config`](https://github.com/phonism/genrec/blob/main/genrec/modules/utils.py), [`config/`](https://github.com/phonism/genrec/tree/main/config) |
| Layers, losses, and metrics | [Shared modules](modules.md) | [`genrec/modules/`](https://github.com/phonism/genrec/tree/main/genrec/modules) |
| Real preprocessing entry points | [Processing](processors.md) | data-module helpers and `scripts/` |
| Choosing or injecting a dataset | [Data routing](dataset-factory.md) | direct construction or Gin `dataset=@ClassName` |
| Train functions and artifacts | [Trainers](trainers.md) | [`genrec/trainers/`](https://github.com/phonism/genrec/tree/main/genrec/trainers) |
| RQ-VAE return types | [RQ-VAE API](rqvae.md) | [`genrec/models/rqvae.py`](https://github.com/phonism/genrec/blob/main/genrec/models/rqvae.py) |
| TIGER tokens and generation | [TIGER API](tiger.md) | [`genrec/models/tiger.py`](https://github.com/phonism/genrec/blob/main/genrec/models/tiger.py) |

## Model modules

There are eleven implementation modules under `genrec/models`. RQ-VAE and RQ-KMeans are tokenizer/quantizer components, not standalone recommenders.

| Module / main class | Model page |
|---|---|
| `sasrec.py` / `SASRec` | [SASRec](../models/sasrec.md) |
| `hstu.py` / `HSTU` | [HSTU](../models/hstu.md) |
| `rqvae.py` / `RqVae` | [RQ-VAE](../models/rqvae.md), [API](rqvae.md) |
| `rqkmeans.py` / `RqKmeans` | [RQ-KMeans](../models/rqkmeans.md) |
| `tiger.py` / `Tiger` | [TIGER](../models/tiger.md), [API](tiger.md) |
| `lcrec.py` / `LCRec` | [LCRec](../models/lcrec.md) |
| `cobra.py` / `Cobra` | [COBRA](../models/cobra.md) |
| `onerec.py` / `OneRec` | [OneRec](../models/onerec.md) |
| `rpg.py` / `RPGModel` | [RPG](../models/rpg.md) |
| `notellm.py` / `Query2Embedding` | [NoteLLM](../models/notellm.md) |
| `gram.py` / `GRAM` | [GRAM](../models/gram.md) |

`genrec.models` re-exports `RqVae`, `QuantizeForwardMode`, `Tiger`, `SASRec`, `HSTU`, `LCRec`, `Cobra`, and `GRAM`. Import the other classes from their implementation modules:

```python
from genrec.models.onerec import OneRec
from genrec.models.rpg import RPGModel
from genrec.models.notellm import Query2Embedding
from genrec.models.rqkmeans import RqKmeans
```

`genrec.data` only re-exports the shared sequence named tuples and `cycle`; dataset classes should also be imported from their concrete modules.

## Cross-model contracts that actually exist

- A dataset follows PyTorch's `__len__`/`__getitem__` protocol; its item schema must match the selected trainer's collator.
- Trainer entry points call `parse_config()` and accept one Gin file, `--split`, and repeatable `--gin` bindings.
- Several item/SID evaluators share `TopKAccumulator` and `SemanticIdItemMapper`.
- Most trainers write `results.json` through `save_run_results`, but checkpoint names and payloads are intentionally heterogeneous.

Do not assume that an argument supported by one trainer is supported by another. Inspect both the target `@gin.configurable` function and its `.gin` file.

## Static verification

```bash
rg -n '^class |^def |^@gin.configurable' genrec/models genrec/data genrec/modules genrec/trainers
rg -n 'train\.|train_[a-z]+\.' config
```

See also [model selection](../models/overview.md), [architecture](../architecture/index.md), and the [implementation status](../reference/model-status.md).
