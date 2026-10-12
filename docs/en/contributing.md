# Contributing

GenRec is an experimental repository for generative-recommendation reproduction and fair comparison, not a production service framework. A contribution must be more than runnable: another researcher must be able to identify the source, protocol, configuration, artifacts, and exact status of the claims it documents.

## Before you start

Read the material closest to your change:

- [Repository map](architecture/repository-map.md)
- [Extension guide](architecture/extension-guide.md)
- [Configuration system](architecture/configuration.md)
- [Evaluation system](architecture/evaluation-system.md)
- [Documentation policy](reference/documentation-policy.md)
- [Model-page template](models/template.md)
- [Verified command index](reference/command-index.md)

The repository has no universal model, dataset, processor, factory, or trainer inheritance hierarchy. Paper pipelines differ in inputs, targets, and decoding. Extend a neighboring implementation's real interfaces and reuse `genrec/modules/`, `genrec/trainers/trainer_utils.py`, and `genrec/modules/utils.py` where appropriate.

## Development environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,docs]"
```

GPU experiments also require a PyTorch build compatible with the host CUDA driver. Never commit private absolute paths, credentials, downloaded model weights, or experiment outputs.

## Workflow

1. Classify the change: implementation fix, protocol fix, reproduction result, or new model.
2. Locate the closest dataset, collator, model, trainer, and Gin config; write down input/output shapes and ID conventions.
3. Build the smallest end-to-end path. Explain paper behavior and repository deviations separately.
4. Add tests proportional to the risk, especially around protocol or tensor boundaries.
5. Run relevant formatting, tests, and documentation checks.
6. Preserve checkpoints, logs, `results.json`, and exact source provenance for new measurements.
7. Update the model page, architecture page, or command index whenever behavior changes.

Keep changes reviewable: do not mix unrelated formatting, generated data, and model behavior; do not commit `dataset/`, `models_hub/`, `out/`, W&B caches, or checkpoints; do not rewrite historical numbers into a new protocol; and document compatibility breaks in cache keys, checkpoint fields, or Gin selectors.

## Implementation conventions

### Data and IDs

Model-specific datasets live under `genrec/data/`. Document:

- raw-file and cache-directory contracts;
- raw IDs, contiguous internal IDs, padding IDs, and SID-token transforms;
- train/valid/test boundaries and split unit;
- `__getitem__` and collator fields, dtypes, shapes, and padding;
- every semantic parameter represented in cache keys;
- whether any valid/test statistic could leak into training.

Headline sequence-recommendation results must name `leave_one_out` or `global_temporal`. A changed candidate set, negative sampler, windowing rule, or SID→item resolver is a protocol change, not merely an optimization.

### Models and trainers

Put models in `genrec/models/`, reusable components in `genrec/modules/`, and CLI entry points in `genrec/trainers/`. Trainers are independent functions rather than subclasses of a common base, but contributions should still ensure:

- model calls, losses, scoring/generation APIs, and tensor shapes are traceable;
- datasets do not hardcode GPU movement;
- only the main process writes shared logs, checkpoints, and results;
- early stopping, best-checkpoint selection, and final test use consistent weights;
- SID predictions are resolved with training-only statistics, then deduplicated before item-level metrics;
- external encoders, tokenizers, and cached artifacts are declared prerequisites.

CLI entry points use `genrec.modules.utils.parse_config` with one Gin file and repeatable `--gin` overrides. Use the real top-level selector: `train_sft`, `train_clm`, `train_grpo`, `train_rpg`, or `train_gram` where applicable—not an invented universal `train` API.

### Configuration

Store default experiments under `config/<model>/...` and make them resolvable from the repository root. Verify imports, upstream/downstream paths, model/tokenizer geometry, explicit dataset protocol, literal `{split}` use, and every example config path. Do not document private one-off experiment files as public commands.

## Tests and static checks

Run the narrowest relevant tests first, then the full suite. Tests are flat `tests/test_*.py` files; do not cite a fictional generic test module.

```bash
pytest -q \
  tests/test_evaluation_protocol.py \
  tests/test_honest_metrics.py \
  tests/test_sid_collisions.py

pytest -q

ruff check genrec scripts tests
black --check genrec scripts tests
mypy genrec
git diff --check
```

Some tests require PyTorch, Transformers, FAISS, or local model assets. If a check cannot run, name the missing dependency and untested scope instead of claiming success.

For documentation changes:

```bash
mkdocs build --strict
rg -n 'TODO|to be completed|coming soon' docs/en
```

## Results and provenance

A log line is not a benchmark artifact. Before a result enters an aggregate table, its `results.json` must identify model, split, seed, evaluation protocol, item-level evaluation, and exact source provenance. SID recommenders must record training-only resolution, and Recall/NDCG values must be finite numbers in `[0, 1]`.

Validate before aggregating:

```bash
python scripts/validate_matrix_results.py out/matrix \
  --models sasrec hstu \
  --splits beauty sports toys \
  --protocols leave_one_out \
  --seeds 42 43 44

python scripts/aggregate_results.py out/matrix \
  --metrics Recall@5 Recall@10 NDCG@10 \
  --format md
```

`scripts/stamp_results_provenance.py` may annotate an artifact whose exact origin is known; it must not legitimize an unknown result. Do not mix SID-level, sampled-negative, or legacy-protocol figures into `BENCHMARKS.md`.

## Definition of done for a model page

A new model, important variant, or behavior that changes result semantics needs a corresponding model-page update. The page must:

- separate implementation status from reproduction status;
- distinguish paper, official-code, and GenRec facts;
- define inputs, outputs, representation, shapes, and special tokens;
- map the pipeline to real files, classes, functions, and configs;
- provide commands whose prerequisites and outputs are explicit;
- state checkpoint selection, candidate generation, constraints, deduplication, and metric level;
- disclose paper deviations, limitations, and negative evidence;
- use live relative links, configs, and script names.

An unimplemented model may be documented, but it must be labeled unimplemented and must not present fictional GenRec commands. Follow the [model-page template](models/template.md).

## Pull-request handoff

State what changed and why it affects—or does not affect—prior results; list the exact checks run and those skipped; name dataset, model-weight, and upstream-checkpoint prerequisites; provide protocol, seed, artifact path, and source provenance for measurements; and include a migration note for public interfaces, caches, or checkpoints.

When code, configs, tests, and documentation disagree, do not silently choose the most convenient version. Record the conflict under the [documentation policy](reference/documentation-policy.md) and make code behavior, default configuration, and status claims independently verifiable.
