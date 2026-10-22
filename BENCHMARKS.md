# GenRec Reproducibility Matrix

Last updated: 2026-10-22. The completed snapshot contains **162 main cells**:
9 training/evaluation pipelines × Beauty/Sports/Toys × 2 protocols × 3 seeds.
The recovery queue finished **84/84 jobs** on 2026-10-21; it includes tokenizer
and smoke jobs and is not the denominator of the full matrix. Workers and the
remote monitor exited after completion.

Per-seed metrics, recipe labels, original artifact paths and SHA256 checksums:
[main CSV](results/matrix_20261022.csv),
[supplemental CSV](results/ablations_20261022.csv), and
[machine-readable summary](results/matrix_20261022.summary.json).

## What the completed runs support

- The matrix covers **eight recommender types and nine pipelines**: SASRec,
  HSTU, TIGER, LCRec, COBRA, RPG, NoteLLM, and OneRec (SFT and GRPO).
- SASRec has the highest LOO mean R@10 on Beauty/Sports and essentially ties
  NoteLLM on Toys after rounding. NoteLLM's I2I query/candidate semantics differ;
  the table is not evidence of a universally strongest model.
- RPG is competitive on Toys (LOO N@10 0.0472 versus SASRec 0.0441). Larger
  autoregressive models do not consistently beat the discriminative baseline.
- Global-temporal R@10 is lower than LOO for every model/domain here. This
  shows that conclusions depend on the split; it does not isolate the cause
  of the gap or make the remaining transductive inputs disappear.
- GRPO does not show a consistent isolated improvement: 16/18 selected policies
  are SFT, and evaluation prompts vary between passes.
- Beauty's four-code OneRec ablation improves R@10 from 0.0658 to 0.0727, but
  only that domain/protocol was tested. The main recipe stays at three codes.
  TIGER/COBRA paper audits use separate recipes and do not establish that every
  original paper result was reproduced. No significance claim is made here.

## Reproduction entry points

Use `scripts/run_matrix.py --preset amazon2014_20261022` for this snapshot;
[`config/matrix/amazon2014_20261022.json`](config/matrix/amazon2014_20261022.json)
connects each CSV recipe to its config and prerequisite paths. The README has
seed-42 tokenizer preparation commands and both 81-cell protocol dry-runs.
Generic runner defaults and historical tuning scripts target other recipes.
The preset uses LCRec's four-code/4B/history-20 recipe, COBRA optimized light,
RPG's domain-specific configs, and OneRec's three-code SFT/GRPO paths.
It reproduces the recipe, not bit-identical historical results: pretrained
weights, tokenizer checkpoints, cached embeddings, and raw result files are
not distributed in this Git tree; the source/prompt limitations below remain.

Check all published table values without PyTorch:

```bash
python scripts/check_benchmarks.py
# Optional: also verify a recovered primary/ + ablations/ raw-results mirror.
python scripts/check_benchmarks.py --artifacts out/matrix_20261022
```

## Protocol and interpretation

- Amazon Reviews 2014, 5-core; seeds 42/43/44. Sequence history is 50 except
  **LCRec's paper recipe, which uses 20**. This is not a history-length-matched
  matrix; per-run lengths are recorded in the CSV. Cells are mean ± **population**
  standard deviation. Each main row has all three seeds, with no duplicate `(model, split, protocol, seed)`.
- `leave_one_out`: final interaction is test, penultimate is validation.
  `global_temporal`: global 80%/90% time boundaries; metadata, catalog filtering
  and some SID construction remain transductive. Keep the protocols separate.
- All headline metrics use item IDs and the full catalog, without sampled
  negatives. SID models resolve predictions with training-only popularity and
  deduplicate items; SID diagnostics and COBRA sparse-only diagnostics are not
  headline metrics. Beam search approximates top-K for autoregressive models.
- NoteLLM is the GCL+CSFT Amazon I2I adaptation: the query is the last history
  item and its own catalog ID is excluded. Its candidate/query semantics differ
  from sequence models. These are repository adaptations, not a claim that all
  original paper results or settings have been reproduced.
- GRAM, OneRec CLM, Home, and cross-dataset HSTU sampled-softmax are outside this
  matrix. RQ-VAE/RQ-KMeans are tokenizer stages, not recommender rows.

## Leave-one-out: 27 complete rows

| Model | Dataset | n | R@1 | R@5 | R@10 | N@10 |
| --- | --- | --- | --- | --- | --- | --- |
| SASRec (CE) | Beauty | 3 | 0.0081 ± 0.0006 | 0.0542 ± 0.0002 | 0.0858 ± 0.0002 | 0.0420 ± 0.0002 |
| HSTU (CE) | Beauty | 3 | 0.0150 ± 0.0006 | 0.0461 ± 0.0008 | 0.0727 ± 0.0008 | 0.0392 ± 0.0004 |
| TIGER (matrix recipe) | Beauty | 3 | — | 0.0361 ± 0.0010 | 0.0568 ± 0.0010 | 0.0301 ± 0.0009 |
| LCRec (4B, seq20, gb128) | Beauty | 3 | 0.0205 ± 0.0001 | 0.0509 ± 0.0011 | 0.0729 ± 0.0004 | 0.0431 ± 0.0001 |
| COBRA (optimized light) | Beauty | 3 | 0.0084 ± 0.0005 | 0.0290 ± 0.0012 | 0.0454 ± 0.0006 | 0.0240 ± 0.0005 |
| RPG (direct) | Beauty | 3 | 0.0197 ± 0.0015 | 0.0506 ± 0.0006 | 0.0732 ± 0.0007 | 0.0427 ± 0.0006 |
| NoteLLM (GCL+CSFT, I2I) | Beauty | 3 | 0.0213 ± 0.0006 | 0.0502 ± 0.0003 | 0.0693 ± 0.0004 | 0.0423 ± 0.0003 |
| OneRec SFT (3 SID) | Beauty | 3 | 0.0147 ± 0.0009 | 0.0433 ± 0.0016 | 0.0658 ± 0.0016 | 0.0364 ± 0.0012 |
| OneRec + GRPO (selected) | Beauty | 3 | 0.0145 ± 0.0007 | 0.0437 ± 0.0018 | 0.0661 ± 0.0024 | 0.0365 ± 0.0014 |
| SASRec (CE) | Sports | 3 | 0.0046 ± 0.0004 | 0.0313 ± 0.0004 | 0.0497 ± 0.0005 | 0.0241 ± 0.0003 |
| HSTU (CE) | Sports | 3 | 0.0070 ± 0.0006 | 0.0234 ± 0.0009 | 0.0384 ± 0.0011 | 0.0200 ± 0.0009 |
| TIGER (matrix recipe) | Sports | 3 | — | 0.0207 ± 0.0012 | 0.0342 ± 0.0017 | 0.0174 ± 0.0010 |
| LCRec (4B, seq20, gb128) | Sports | 3 | 0.0094 ± 0.0003 | 0.0260 ± 0.0007 | 0.0399 ± 0.0005 | 0.0223 ± 0.0004 |
| COBRA (optimized light) | Sports | 3 | 0.0054 ± 0.0001 | 0.0179 ± 0.0006 | 0.0290 ± 0.0007 | 0.0153 ± 0.0004 |
| RPG (direct) | Sports | 3 | 0.0108 ± 0.0004 | 0.0290 ± 0.0005 | 0.0426 ± 0.0007 | 0.0244 ± 0.0005 |
| NoteLLM (GCL+CSFT, I2I) | Sports | 3 | 0.0082 ± 0.0002 | 0.0237 ± 0.0003 | 0.0351 ± 0.0003 | 0.0197 ± 0.0001 |
| OneRec SFT (3 SID) | Sports | 3 | 0.0061 ± 0.0002 | 0.0222 ± 0.0002 | 0.0341 ± 0.0007 | 0.0180 ± 0.0002 |
| OneRec + GRPO (selected) | Sports | 3 | 0.0062 ± 0.0002 | 0.0220 ± 0.0002 | 0.0341 ± 0.0005 | 0.0180 ± 0.0001 |
| SASRec (CE) | Toys | 3 | 0.0058 ± 0.0005 | 0.0610 ± 0.0005 | 0.0917 ± 0.0007 | 0.0441 ± 0.0001 |
| HSTU (CE) | Toys | 3 | 0.0134 ± 0.0006 | 0.0438 ± 0.0009 | 0.0686 ± 0.0007 | 0.0366 ± 0.0008 |
| TIGER (matrix recipe) | Toys | 3 | — | 0.0288 ± 0.0009 | 0.0477 ± 0.0004 | 0.0242 ± 0.0003 |
| LCRec (4B, seq20, gb128) | Toys | 3 | 0.0201 ± 0.0003 | 0.0504 ± 0.0024 | 0.0705 ± 0.0041 | 0.0420 ± 0.0019 |
| COBRA (optimized light) | Toys | 3 | 0.0052 ± 0.0000 | 0.0252 ± 0.0006 | 0.0412 ± 0.0007 | 0.0202 ± 0.0005 |
| RPG (direct) | Toys | 3 | 0.0206 ± 0.0003 | 0.0569 ± 0.0005 | 0.0820 ± 0.0003 | 0.0472 ± 0.0002 |
| NoteLLM (GCL+CSFT, I2I) | Toys | 3 | 0.0233 ± 0.0006 | 0.0674 ± 0.0003 | 0.0917 ± 0.0005 | 0.0539 ± 0.0004 |
| OneRec SFT (3 SID) | Toys | 3 | 0.0123 ± 0.0004 | 0.0386 ± 0.0008 | 0.0591 ± 0.0004 | 0.0321 ± 0.0006 |
| OneRec + GRPO (selected) | Toys | 3 | 0.0116 ± 0.0003 | 0.0393 ± 0.0004 | 0.0604 ± 0.0013 | 0.0323 ± 0.0005 |

## Global temporal: 27 complete rows

| Model | Dataset | n | R@1 | R@5 | R@10 | N@10 |
| --- | --- | --- | --- | --- | --- | --- |
| SASRec (CE) | Beauty | 3 | 0.0005 ± 0.0003 | 0.0092 ± 0.0008 | 0.0166 ± 0.0010 | 0.0072 ± 0.0004 |
| HSTU (CE) | Beauty | 3 | 0.0026 ± 0.0006 | 0.0090 ± 0.0001 | 0.0163 ± 0.0006 | 0.0080 ± 0.0003 |
| TIGER (matrix recipe) | Beauty | 3 | — | 0.0121 ± 0.0001 | 0.0206 ± 0.0013 | 0.0101 ± 0.0006 |
| LCRec (4B, seq20, gb128) | Beauty | 3 | 0.0036 ± 0.0006 | 0.0130 ± 0.0022 | 0.0208 ± 0.0025 | 0.0108 ± 0.0016 |
| COBRA (optimized light) | Beauty | 3 | 0.0028 ± 0.0007 | 0.0096 ± 0.0007 | 0.0175 ± 0.0007 | 0.0088 ± 0.0005 |
| RPG (direct) | Beauty | 3 | 0.0030 ± 0.0006 | 0.0105 ± 0.0003 | 0.0173 ± 0.0004 | 0.0088 ± 0.0003 |
| NoteLLM (GCL+CSFT, I2I) | Beauty | 3 | 0.0060 ± 0.0005 | 0.0180 ± 0.0008 | 0.0266 ± 0.0013 | 0.0148 ± 0.0006 |
| OneRec SFT (3 SID) | Beauty | 3 | 0.0042 ± 0.0004 | 0.0156 ± 0.0005 | 0.0239 ± 0.0014 | 0.0125 ± 0.0005 |
| OneRec + GRPO (selected) | Beauty | 3 | 0.0045 ± 0.0001 | 0.0161 ± 0.0009 | 0.0245 ± 0.0018 | 0.0129 ± 0.0005 |
| SASRec (CE) | Sports | 3 | 0.0013 ± 0.0009 | 0.0106 ± 0.0009 | 0.0189 ± 0.0008 | 0.0087 ± 0.0007 |
| HSTU (CE) | Sports | 3 | 0.0033 ± 0.0006 | 0.0111 ± 0.0005 | 0.0192 ± 0.0007 | 0.0098 ± 0.0005 |
| TIGER (matrix recipe) | Sports | 3 | — | 0.0119 ± 0.0008 | 0.0205 ± 0.0021 | 0.0106 ± 0.0007 |
| LCRec (4B, seq20, gb128) | Sports | 3 | 0.0029 ± 0.0004 | 0.0125 ± 0.0009 | 0.0207 ± 0.0007 | 0.0105 ± 0.0003 |
| COBRA (optimized light) | Sports | 3 | 0.0038 ± 0.0006 | 0.0133 ± 0.0009 | 0.0221 ± 0.0006 | 0.0113 ± 0.0003 |
| RPG (direct) | Sports | 3 | 0.0035 ± 0.0005 | 0.0123 ± 0.0004 | 0.0182 ± 0.0002 | 0.0098 ± 0.0001 |
| NoteLLM (GCL+CSFT, I2I) | Sports | 3 | 0.0032 ± 0.0001 | 0.0109 ± 0.0014 | 0.0192 ± 0.0005 | 0.0096 ± 0.0001 |
| OneRec SFT (3 SID) | Sports | 3 | 0.0028 ± 0.0001 | 0.0136 ± 0.0011 | 0.0220 ± 0.0007 | 0.0109 ± 0.0004 |
| OneRec + GRPO (selected) | Sports | 3 | 0.0029 ± 0.0003 | 0.0133 ± 0.0007 | 0.0223 ± 0.0004 | 0.0110 ± 0.0002 |
| SASRec (CE) | Toys | 3 | 0.0008 ± 0.0003 | 0.0096 ± 0.0013 | 0.0179 ± 0.0005 | 0.0078 ± 0.0003 |
| HSTU (CE) | Toys | 3 | 0.0021 ± 0.0003 | 0.0098 ± 0.0004 | 0.0165 ± 0.0005 | 0.0079 ± 0.0002 |
| TIGER (matrix recipe) | Toys | 3 | — | 0.0115 ± 0.0004 | 0.0184 ± 0.0020 | 0.0096 ± 0.0004 |
| LCRec (4B, seq20, gb128) | Toys | 3 | 0.0037 ± 0.0009 | 0.0125 ± 0.0014 | 0.0209 ± 0.0011 | 0.0108 ± 0.0009 |
| COBRA (optimized light) | Toys | 3 | 0.0025 ± 0.0006 | 0.0113 ± 0.0006 | 0.0190 ± 0.0007 | 0.0093 ± 0.0003 |
| RPG (direct) | Toys | 3 | 0.0027 ± 0.0004 | 0.0134 ± 0.0003 | 0.0226 ± 0.0004 | 0.0110 ± 0.0000 |
| NoteLLM (GCL+CSFT, I2I) | Toys | 3 | 0.0042 ± 0.0003 | 0.0175 ± 0.0005 | 0.0263 ± 0.0009 | 0.0137 ± 0.0003 |
| OneRec SFT (3 SID) | Toys | 3 | 0.0036 ± 0.0003 | 0.0131 ± 0.0008 | 0.0208 ± 0.0003 | 0.0107 ± 0.0004 |
| OneRec + GRPO (selected) | Toys | 3 | 0.0034 ± 0.0002 | 0.0127 ± 0.0008 | 0.0208 ± 0.0005 | 0.0107 ± 0.0004 |

## OneRec SFT → validation-selected GRPO pipeline

The main OneRec recipe remains **3×256 RQ-KMeans codes**. The 4-code suffix
variant is a separate Beauty LOO ablation. GRPO uses the matching SFT seed and
protocol, group size 8, hierarchical reward, and includes the starting SFT
weights in validation selection.

**16 of 18 GRPO cells selected the original SFT policy (`best_epoch=-1`).**
Only Toys LOO seeds 42/43 selected a trained GRPO epoch (0/1). Consequently the
table measures the validation-selected pipeline; it does not establish a
consistent isolated benefit from RL.

OneRec/LCRec evaluation currently samples prompt templates in `__getitem__`.
Evaluation prompts are not frozen across passes, so even a fallback to the
same SFT weights can differ from the standalone SFT test. Do not attribute
those differences to GRPO. Fixed prompts, matched baseline evaluation and
validation re-selection are required before claiming a small RL effect.

| Dataset | Protocol | SFT R@10 | Selected pipeline R@10 | RL epochs selected |
| --- | --- | --- | --- | --- |
| Beauty | leave_one_out | 0.0658 ± 0.0016 | 0.0661 ± 0.0024 | 0/3 |
| Beauty | global_temporal | 0.0239 ± 0.0014 | 0.0245 ± 0.0018 | 0/3 |
| Sports | leave_one_out | 0.0341 ± 0.0007 | 0.0341 ± 0.0005 | 0/3 |
| Sports | global_temporal | 0.0220 ± 0.0007 | 0.0223 ± 0.0004 | 0/3 |
| Toys | leave_one_out | 0.0591 ± 0.0004 | 0.0604 ± 0.0013 | 2/3 |
| Toys | global_temporal | 0.0208 ± 0.0003 | 0.0208 ± 0.0005 | 0/3 |

## Supplemental three-seed rows

| Model | Dataset | Protocol | n | R@1 | R@5 | R@10 | N@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HSTU (corrected Meta-local sampled softmax) | Beauty | leave_one_out | 3 | 0.0045 ± 0.0005 | 0.0465 ± 0.0011 | 0.0754 ± 0.0021 | 0.0353 ± 0.0011 |
| OneRec SFT (4-code suffix ablation) | Beauty | leave_one_out | 3 | 0.0168 ± 0.0002 | 0.0483 ± 0.0009 | 0.0727 ± 0.0013 | 0.0405 ± 0.0005 |

The 4-code row is kept separate from 3-code SFT/GRPO. It changes the learned
representation and cannot be treated as an evaluation-only correction.

## Source provenance and recipe boundaries

The snapshot mixes base Git versions `fb4bd1a`, `e52cc02`, and `6da9328`.
Exact base bundle/Git fields are retained per seed in the CSV. October runs
restored the recovery bundle with SHA256
`8bfb3b9ec036b99d08dbffff53a1d0b70154e4de5fde20cdd3bd21f64ed4b412`.
Those fields do not encode subsequent uncommitted patches: the summary records
the verified source-file hashes; source commit `c621411` preserves
the LCRec decoder/optimizer fixes, effective batch 128 recipe, GPU-lock fix and
GRPO `defaultdict` import fix. Do not describe this as one identical code bundle.

- LCRec uses four-level USM, Qwen3-4B, Sentence-T5-XL, six tasks, four epochs,
  history 20, micro-batch 4 × accumulation 32 = effective batch 128, beam 20,
  eval batch 4.
  This is the Amazon-2014 adaptation, distinct from the old 1.7B/five-code seed.
- TIGER main rows use the original matrix recipe. Beauty LOO R@10 is 0.0568;
  the archived paper-aligned Beauty row below is a different recipe and is not
  substituted into this matrix.
- COBRA uses optimized light across all three domains. Its separate Beauty
  paper-alignment audit remains below; old pre-fix matrix rows are excluded.
- RPG main metrics and validation checkpoint selection use direct item scores.
  Sports/Toys LOO seeds have different graph-only `num_beams/n_edges/steps`;
  their graph diagnostics are not aggregated as a common recipe.
- NoteLLM Beauty LOO seed42 is the earlier GCL+CSFT recovery run; seeds43/44
  are October runs with matching training settings. Old GCL-only rows are excluded.
- Raw `results.json` metadata and checksums pass the strict validator; that is
  a structural/protocol check, not proof of paper parity or deterministic prompts.
  Full CPU regression suite after recovery: **97 passed**;
  `mkdocs build --strict` also passes.

## Archived September audits (outside the 162-cell snapshot)

The following rows retain their previously published provenance and scope.
They were not re-run as part of the October completion and are not pooled with
the main matrix. Their source paths and interpretation are preserved below.

| Model | Dataset | Protocol | n | R@5 | R@10 | N@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| COBRA (corrected legacy ckpt) | Beauty | global temporal | 3 | 0.0094 ± 0.0005 | 0.0159 ± 0.0012 | 0.0082 ± 0.0004 |
| COBRA (corrected legacy ckpt) | Beauty | leave one out | 3 | 0.0202 ± 0.0008 | 0.0350 ± 0.0008 | 0.0173 ± 0.0005 |
| COBRA (paper-aligned, validation-selected) | Beauty | leave one out | 3 | 0.0285 ± 0.0013 | 0.0460 ± 0.0013 | 0.0239 ± 0.0009 |
| TIGER (paper-aligned) | Beauty | leave one out | 3 | 0.0413 ± 0.0017 | 0.0655 ± 0.0014 | 0.0354 ± 0.0012 |

The pre-fix COBRA rows were LOO R@10 `0.0098 ± 0.0008` and temporal R@10
`0.0056 ± 0.0010`. They are retained only as historical A/B baselines: the
right-padding generation bug prevented later SID positions from being decoded.
The corrected raw evaluations and three-seed aggregate are under
`out/recovery/cobra/beauty/`; `summary_final.json` is the authoritative recovery
and optimization aggregate. Relative to the corrected legacy checkpoints, the
optimized light model improves LOO R@10 by 29.7% and temporal R@10 by 10.2%.
The later paper-aligned COBRA recipe was selected entirely on validation
(validation R@10 `0.0644 ± 0.0008`) before a one-shot test. Its test R@10 is
`0.0460 ± 0.0013`, exposing a large validation-to-test generalization gap and
remaining 36.6% relative gap to the paper's `0.0725`; see
`docs/zh/cobra-paper-alignment-20260903.md`.

RPG's paper graph parameters (Beauty: beam 10, 100 edges, 3 propagation steps)
were also selected against direct decoding on validation. They reduced all
three seeds, so direct decoding remains the headline. The strict re-evaluation
reproduced R@10 `0.0732 ± 0.0007` and N@10 `0.0427 ± 0.0006`; see
`docs/zh/rpg-paper-alignment-20260903.md`.
