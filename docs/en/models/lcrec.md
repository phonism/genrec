# LCRec: adapting a causal LLM with collaborative semantic IDs

| Field | Current repository |
|---|---|
| Paper | [Adapting Large Language Models by Integrating Collaborative Semantics](https://arxiv.org/abs/2311.09049), ICDE 2024 |
| Task | next-item generative retrieval plus text/SID alignment tasks |
| Item representation | main config: 5×256; paper-alignment config: 4×256 plus USM |
| Backbone | Hugging Face causal LM; current Gin selects local Qwen3-1.7B |
| Training | response-only causal LM, full-parameter by default |
| Decoding | position-specific codebook masks with KV-cached beam search |
| Headline | item Recall/NDCG after training-only SID resolution and deduplication |
| Checkpoint | Hugging Face model/tokenizer directory; no optimizer/scheduler/epoch |

The wrapper uses `AutoModelForCausalLM` and `AutoTokenizer`; it is not hard-coded to Qwen. Python's Qwen2.5 fallback and old docstrings are not the current experiment setting. The repository's [`reference/LC-Rec/`](https://github.com/phonism/genrec/tree/main/reference/LC-Rec) is reference code, while this implementation is an independent GenRec adaptation.

## Two stages and vocabulary

Stage 1 trains LCRec's dedicated RQ-VAE. The legacy main recipe uses input 768, latent 64, hidden 512/256/128 and five 256-code layers. `rqvae_paper_recipe.gin` follows the reference indexer geometry with latent 32, hidden 2048/1024/512/256/128/64 and four 256-code layers. It also matches the released `run.sh` behavior: first-batch K-Means on 1,024 items, Xavier-normal MLP weights, and mean-reduced reconstruction/per-layer VQ losses. Stage 2 trains the causal LM on user sequences. The catalog order, Sentence-T5-XL embedding space, and RQ-VAE geometry must remain identical.

For each configured layer, `LCRec.add_codebook_tokens()` adds position-specific special tokens such as `<C0_0>`. A complete SID contains five tokens in the legacy recipe or four in the paper recipe. Model input/output embeddings are resized, so the saved tokenizer is a required checkpoint artifact.

`uniform_semantic_mapping=True` performs the official-style iterative Sinkhorn reassignment on collision groups in the final codebook. True duplicates or a fixed point may still collide, and the adapter reports the resulting unique count instead of silently treating tuples as item identities.

## Tasks and objective

`AmazonLCRecDataset` implements six tasks:

| Task | Mapping | Enabled in current main config? |
|---|---|---|
| `seqrec` | history SIDs to next SID | Yes |
| `item2index` | item text to SID | Yes |
| `index2item` | SID to item text | Yes |
| `fusionseqrec` | history SIDs to item text | No |
| `itemsearch` | synthetic query/history to SID | No |
| `preferenceobtain` | history to category-preference text | No |

The optional tasks use local metadata heuristics and are not equivalent to the paper's data generation simply because all six names exist. The legacy main config enables three tasks; `lcrec_paper_recipe.gin` enables all six. Validation/test datasets generate `seqrec` only.

The collator concatenates `prompt + response + EOS` and sets prompt/padding labels to `-100`:

\[
\mathcal L=-\sum_{t\in response+EOS}\log p_\theta(x_t\mid x_{<t}).
\]

`train.max_length=512` controls actual LM truncation. Dataset `max_text_len` is currently stored but does not truncate LCRec metadata.

## Generation and evaluation

The class has **no `LCRec.generate()` method**. Its production trainer uses:

```python
sem_ids, scores = model.generate_constrained_beam(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_mask=helper.position_mask,
    code_map=helper.code_map,
    num_codebooks=5,
    beam_width=10,
    topk=10,
)
```

`generate_topk` is an older per-example alternative. `ConstrainedDecodingHelper` allows only `<Cj_*>` at code position `j` and EOS after the final code. This guarantees positional token type, **not membership in the catalog**; it is not a catalog PrefixTrie.

The item mapper resolves each SID using only training-prefix popularity, breaks ties by smaller item ID, drops absent tuples, and removes duplicate items. Item-level Recall/NDCG@1/5/10 drives validation-only model selection. The trainer saves `checkpoint_best`, reloads that state on every rank, then evaluates test exactly once. Pre-resolution SID tuple metrics remain diagnostics under `results.json.extra`.

| Layer | Source | Contract |
|---|---|---|
| wrapper | [`lcrec.py`](https://github.com/phonism/genrec/blob/main/genrec/models/lcrec.py) | HF forward and custom beam methods |
| data | [`amazon_lcrec.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_lcrec.py) | task/prompt/response/target dictionaries |
| collator/helper/trainer | [`lcrec_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/lcrec_trainer.py) | response mask, code-position mask, Accelerate loop |
| resolver | [`metrics.py`](https://github.com/phonism/genrec/blob/main/genrec/modules/metrics.py) | SID candidates to deduplicated items |

## Run and artifacts

```bash
python -m genrec.trainers.rqvae_trainer \
  config/lcrec/amazon/rqvae.gin --split beauty

python -m genrec.trainers.lcrec_trainer \
  config/lcrec/amazon/lcrec.gin --split beauty

# Paper-structure adaptation: 4-code indexer, USM, six tasks, history 20,
# beam 20, and Qwen3-4B as the locally available LLaMA-7B substitute.
python -m genrec.trainers.rqvae_trainer \
  config/lcrec/amazon/rqvae_paper_recipe.gin --split beauty
python -m genrec.trainers.lcrec_trainer \
  config/lcrec/amazon/lcrec_paper_recipe.gin --split beauty
```

Stage 2 currently reads `out/lcrec/amazon/<split>/rqvae/checkpoint_epoch_1999.pt`. The RQ-VAE run continues past that checkpoint, so an alternative epoch requires an explicit path override.

There is no `lcrec_debug.gin`. For a smoke run after real model/tokenizer assets exist, override epochs/sample limits on the standard config.

The current main config uses Qwen3-1.7B, 10 epochs, batch 16 with accumulation 4, learning rate `2e-5`, 1% warmup, 50 history items, 512 LM tokens, full-parameter bf16 training, beam 10, and per-epoch save/eval.

Output contains `checkpoint_epoch_<zero-based epoch>/`, `checkpoint_best/`, `checkpoint_final/`, and `results.json`. `checkpoint_best` is selected only on validation Recall@10; `results.json` contains its one-shot test metrics. Final still means last. `eval_only=True` currently evaluates validation only and does not write a new results artifact. LoRA without PEFT only warns and falls back, so verify trainable-parameter counts.

## Reproduction boundary

The paper-alignment recipe matches four codebooks, USM, six tasks, history 20, four epochs, effective batch 16, learning rate `5e-5`, and beam 20. It still substitutes Sentence-T5-XL and Qwen3-4B for the release's LLaMA-derived embeddings/LLaMA-7B and runs Amazon-2014 Beauty rather than the paper datasets. Historical SID-tuple results are not current item-level headlines; publishable values must come from verified artifacts in `BENCHMARKS.md`.

See [RQ-VAE](rqvae.md), [trainer API](../api/trainers.md#semantic-id-recommenders), and [evaluation architecture](../architecture/evaluation-system.md).
