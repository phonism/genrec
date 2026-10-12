# GenRec Reproducibility Matrix

Last updated: 2026-09-03. This file reports only results produced by the current
honest, item-level evaluation path. Legacy SID-tuple metrics are excluded.

## Protocol

- Dataset: Amazon Reviews 2014, 5-core, full-catalog evaluation.
- Splits: Beauty, Sports and Toys; maximum sequence length 50.
- Seeds: 42, 43 and 44. Values are mean ± population standard deviation.
- `leave_one_out`: the final interaction is test and the penultimate interaction
  is validation for every retained user.
- `global_temporal`: interactions are split at global 80%/90% time boundaries.
  It prevents future interactions from entering train histories, but catalog
  filtering and item metadata remain transductive.
- Headline evaluation level: item IDs. LOO and temporal results are kept in
  separate rows and must not be averaged together.

## P1: Discriminative protocol baselines

All 36 planned cells completed on host 179. Every row has three seeds and every
`results.json` records `evaluation_level=item`, its protocol, and source bundle
SHA256 `79876966435a89861d91d8c8e36ece31facec00a1c16a111d2da2664870f9abf`.
The additional HSTU sampled-softmax row uses corrected Meta-local negatives and
source bundle SHA256
`2b3f2bdf21847671aa1f9f6215aadd04cfe36d0589e6d74af43fed77e2181c51`.

| Model | Dataset | Protocol | n | R@1 | R@5 | R@10 | N@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| HSTU | Beauty | global temporal | 3 | 0.0026 ± 0.0006 | 0.0090 ± 0.0001 | 0.0163 ± 0.0006 | 0.0080 ± 0.0003 |
| HSTU | Beauty | leave one out | 3 | 0.0150 ± 0.0006 | 0.0461 ± 0.0008 | 0.0727 ± 0.0008 | 0.0392 ± 0.0004 |
| HSTU (sampled softmax, corrected local sampler) | Beauty | leave one out | 3 | 0.0045 ± 0.0005 | 0.0465 ± 0.0011 | 0.0754 ± 0.0021 | 0.0353 ± 0.0011 |
| HSTU | Sports | global temporal | 3 | 0.0033 ± 0.0006 | 0.0111 ± 0.0005 | 0.0192 ± 0.0007 | 0.0098 ± 0.0005 |
| HSTU | Sports | leave one out | 3 | 0.0070 ± 0.0006 | 0.0234 ± 0.0009 | 0.0384 ± 0.0011 | 0.0200 ± 0.0009 |
| HSTU | Toys | global temporal | 3 | 0.0021 ± 0.0003 | 0.0098 ± 0.0004 | 0.0165 ± 0.0005 | 0.0079 ± 0.0002 |
| HSTU | Toys | leave one out | 3 | 0.0134 ± 0.0006 | 0.0438 ± 0.0009 | 0.0686 ± 0.0007 | 0.0366 ± 0.0008 |
| SASRec | Beauty | global temporal | 3 | 0.0005 ± 0.0003 | 0.0092 ± 0.0008 | 0.0166 ± 0.0010 | 0.0072 ± 0.0004 |
| SASRec | Beauty | leave one out | 3 | 0.0081 ± 0.0006 | 0.0542 ± 0.0002 | 0.0858 ± 0.0002 | 0.0420 ± 0.0002 |
| SASRec | Sports | global temporal | 3 | 0.0013 ± 0.0009 | 0.0106 ± 0.0009 | 0.0189 ± 0.0008 | 0.0087 ± 0.0007 |
| SASRec | Sports | leave one out | 3 | 0.0046 ± 0.0004 | 0.0313 ± 0.0004 | 0.0497 ± 0.0005 | 0.0241 ± 0.0003 |
| SASRec | Toys | global temporal | 3 | 0.0008 ± 0.0003 | 0.0096 ± 0.0013 | 0.0179 ± 0.0005 | 0.0078 ± 0.0003 |
| SASRec | Toys | leave one out | 3 | 0.0058 ± 0.0005 | 0.0610 ± 0.0005 | 0.0917 ± 0.0007 | 0.0441 ± 0.0001 |

## P2: Verified generative rows

The table below contains only complete three-seed rows. All 36 finished cells
pass the strict artifact validator and record item-level evaluation. TIGER/RPG
use source bundle SHA256
`099208d1fffd786610acdcadb37750f791a7ab93c8422c4cb168137f70167d87`.
The legacy COBRA checkpoints were trained with SHA256
`59421f3488c9b6bd8f58b6619f5fe8e9144ad050b4467801bb958d8b707d0df4`
and re-evaluated at `max_seq_len=50`, beam 64, without catalog-prefix
constraints, using the corrected generation path in SHA256
`c17f9d715e4d1f3909007c7c96b3db53a01b5f1762cdc960560fb84eb8523748`.
The optimized COBRA rows were also trained and evaluated with that corrected
bundle, using the paper's one-layer light encoder and a two-layer decoder.
The final COBRA paper-alignment and strict RPG re-evaluations use source bundle
SHA256 `2f68e0eef7dde005e28d6bf26125fa9d969152504991eaa6ece60b76a366b486`.

| Model | Dataset | Protocol | n | R@5 | R@10 | N@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| COBRA (corrected legacy ckpt) | Beauty | global temporal | 3 | 0.0094 ± 0.0005 | 0.0159 ± 0.0012 | 0.0082 ± 0.0004 |
| COBRA (corrected legacy ckpt) | Beauty | leave one out | 3 | 0.0202 ± 0.0008 | 0.0350 ± 0.0008 | 0.0173 ± 0.0005 |
| COBRA (optimized light) | Beauty | global temporal | 3 | 0.0096 ± 0.0007 | 0.0175 ± 0.0007 | 0.0088 ± 0.0005 |
| COBRA (optimized light) | Beauty | leave one out | 3 | 0.0290 ± 0.0012 | 0.0454 ± 0.0006 | 0.0240 ± 0.0005 |
| COBRA (paper-aligned, validation-selected) | Beauty | leave one out | 3 | 0.0285 ± 0.0013 | 0.0460 ± 0.0013 | 0.0239 ± 0.0009 |
| RPG | Beauty | global temporal | 3 | 0.0105 ± 0.0003 | 0.0173 ± 0.0004 | 0.0088 ± 0.0003 |
| RPG | Beauty | leave one out | 3 | 0.0506 ± 0.0006 | 0.0732 ± 0.0007 | 0.0427 ± 0.0006 |
| RPG | Sports | global temporal | 3 | 0.0123 ± 0.0004 | 0.0182 ± 0.0002 | 0.0098 ± 0.0001 |
| TIGER | Beauty | global temporal | 3 | 0.0121 ± 0.0001 | 0.0206 ± 0.0013 | 0.0101 ± 0.0006 |
| TIGER (paper-aligned) | Beauty | leave one out | 3 | 0.0413 ± 0.0017 | 0.0655 ± 0.0014 | 0.0354 ± 0.0012 |
| TIGER | Sports | global temporal | 3 | 0.0119 ± 0.0008 | 0.0205 ± 0.0021 | 0.0106 ± 0.0007 |
| TIGER | Sports | leave one out | 3 | 0.0207 ± 0.0012 | 0.0342 ± 0.0017 | 0.0174 ± 0.0010 |
| TIGER | Toys | global temporal | 3 | 0.0115 ± 0.0004 | 0.0184 ± 0.0020 | 0.0096 ± 0.0004 |
| TIGER | Toys | leave one out | 3 | 0.0288 ± 0.0009 | 0.0477 ± 0.0004 | 0.0242 ± 0.0003 |

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

## Remaining matrix

- P2: TIGER is complete (18/18), including the Beauty paper-aligned LOO row.
  COBRA Beauty has corrected, optimized-light, and validation-selected
  paper-alignment rows. RPG Beauty LOO has a strict validation/test graph audit.
  COBRA Sports/Toys and unfinished RPG cells remain paused.
- P3: unfinished LCRec, OneRec-SFT and GRPO matrix work remains paused.
- LCRec Beauty LOO currently has one legacy adaptation seed (Qwen3-1.7B, five ordinary
  RQ codebooks): item R@10 `0.0654`, versus the collision-prone SID diagnostic
  `0.0673`. Its raw SID collision rate is 0.49%. This is not a paper row: the
  paper uses four index levels, uniform semantic mapping, a 7B LLaMA backbone,
  max sequence length 20, and different Amazon subsets. The new four-level USM
  recipe is pending a clean rerun after matching the reference's first-batch
  K-Means, Xavier-normal MLP initialization, and mean-reduced reconstruction/VQ
  losses; both pre-fix attempts are excluded.
- No OneRec recommender checkpoint/result artifact is present on host 179;
  only the Beauty/Sports/Toys RQ-KMeans tokenizer artifacts are present. The
  May-2026 Beauty v2b number therefore remains an archived experiment note,
  not a completed current matrix cell.
- NoteLLM has an exact full-catalog item evaluator that excludes each I2I query
  item from its candidate pool. The historical Amazon run is explicitly labeled
  `gcl_only_lora_amazon_adaptation`; the current trainer adds symmetric GCL,
  CSFT, and exact-item false-negative masking, so that old seed is not evidence
  for the corrected recipe.
  Beauty currently has one seed: LOO R@10 `0.0777`, temporal R@10 `0.0281`;
  incomplete rows remain outside the numeric table.

- HSTU's clean Meta-local sampled-softmax rerun is complete for seeds 42/43/44.
  Validation-only checkpoint selection followed by one-shot item-level test
  gives R@5 `0.0465 ± 0.0011`, R@10 `0.0754 ± 0.0021`, and N@10
  `0.0353 ± 0.0011`. All earlier runs with the incorrectly shared/in-batch
  `local` sampler stay excluded.

RQ-VAE and RQ-KMeans are tokenizer stages, not recommender benchmark rows. Their
codebook utilization, collision rate, seed and artifact checksum are recorded
as dependencies of downstream results.
