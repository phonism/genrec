# Dataset API

Every data class inherits directly from PyTorch `Dataset`; there is no repository-wide base class or factory. Construction often performs download, parsing, embedding/SID cache initialization, and sample generation, so it may be expensive.

Common names: `split` is usually an Amazon category; `train_test_split` is usually `train|valid|test`; `max_seq_len` counts history items, not necessarily text tokens. Adapters that expose `evaluation_protocol` accept `leave_one_out` or `global_temporal`.

## Amazon 2014 core datasets

Source: [`genrec/data/amazon.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon.py)

```python
AmazonItemDataset(
    root="dataset/amazon", split="beauty", train_test_split="all",
    encoder_model_name="sentence-transformers/sentence-t5-base",
    force_regenerate=False,
)
```

`__getitem__() -> list[float]`; public `embeddings` has shape `[N,D]` and `dim` is `D`. `train_test_split` is `all|train|eval`.

```python
AmazonSeqDataset(
    root="dataset/amazon", split="beauty", train_test_split="train",
    max_seq_len=20, pretrained_rqvae_path="...",
    encoder_model_name="...", rqvae_input_dim=768,
    rqvae_embed_dim=32, rqvae_hidden_dims=[512, 256, 128, 64],
    rqvae_codebook_size=256, rqvae_n_layers=3,
    subsample=True, add_disambiguation=True, dedup_sinkhorn=False,
    evaluation_protocol="leave_one_out",
)
```

It returns `SeqData(user_id, item_ids, target_ids, target_item)`. History codes are flattened. `sem_ids_list`, `sequences`, and `evaluation_protocol` are consumed by the TIGER trainer. Encoder or RQ-VAE geometry mismatches normally fail at checkpoint loading or matrix operations.

## Discriminative and retrieval adapters

`AmazonSASRecDataset` in [`amazon_sasrec.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_sasrec.py) accepts `root`, category `split`, `train_test_split`, `max_seq_len`, `min_seq_len`, protocol, and temporal ratios. It returns:

```python
{"history": list[int], "target": int}
```

`num_items` excludes padding. `sasrec_collate_fn` left-pads and returns `[B,L]` `input_ids`/next-token `targets`; when `num_items > 0` it also samples one Python-RNG negative per non-padding position. `sasrec_eval_collate_fn` returns `[B,L]` inputs and `[B]` targets.

`AmazonHSTUDataset` in [`amazon_hstu.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_hstu.py) has a similar constructor and returns `history`, `timestamps`, and `target`. Its train/eval collators return the corresponding `[B,L]` timestamp tensor. `MovieLens1MHSTUDataset` in [`ml1m_hstu.py`](https://github.com/phonism/genrec/blob/main/genrec/data/ml1m_hstu.py) has the same item schema but no Amazon evaluation-protocol argument.

`AmazonNoteLLMDataset` in [`amazon_notellm.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_notellm.py) returns adjacent-item pairs:

```python
{"query": int, "target": int}
```

It exposes `item_texts: dict[int,str]`; `notellm_collate_fn` returns Python lists `queries` and `targets`.

## LLM and semantic-ID adapters

`AmazonLCRecDataset` in [`amazon_lcrec.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_lcrec.py) loads an RQ-VAE and returns task dictionaries:

```python
{
  "task": "seqrec|item2index|index2item|fusionseqrec|itemsearch|preferenceobtain",
  "prompt": str,
  "response": str,
  "target_item": int,          # normally absent only for preferenceobtain
  "target_sem_ids": list[int],
}
```

Its collator is in `lcrec_trainer.py`. Unknown tasks raise `ValueError`.

OneRec datasets are in [`amazon_onerec.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_onerec.py) and [`amazon_onerec_clm.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_onerec_clm.py):

- `AmazonOneRecSFTDataset` supports `tokenizer_type="rqvae"|"rqkmeans"` and tasks `sidsft|title2sid|sid2title|fusionseqrec`; every sample includes task, prompt, response, target item, and target SID.
- `AmazonOneRecRLDataset` has no `train_test_split`; it emits RL prompt/target/task dictionaries for `rl_collate_fn`.
- `AmazonOneRecCLMDataset` emits `{"item_seq": list[int], "task": "clm"}` for training and adds target/SID for validation/test.

`AmazonCobraDataset` in [`amazon_cobra.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_cobra.py) returns:

```python
{
  "input_ids": list[int],                 # flattened history SID
  "encoder_input_ids": Tensor[T,L],
  "target_sem_ids": list[int],
  "target_encoder_input_ids": Tensor[1,L],
  "target_item": int,
}
```

## RPG and GRAM

`AmazonRPGDataset` in [`amazon_rpg.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_rpg.py) accepts Amazon paths/protocol, sequence limits, `n_codebook`, power-of-two `codebook_size`, text encoder, optional PCA, and FAISS settings. Training samples contain item `input_ids` and position labels; evaluation uses one target item. `item_id2tokens: Tensor[num_items,C]` initializes the model. First-time OPQ generation requires FAISS.

`AmazonGRAMDataset` in [`amazon_gram.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_gram.py) consumes externally prepared GRAM files and returns:

```python
{"input": list[str], "output": str, "user_id": str}
```

`GRAMCollator(tokenizer, item_prompt_max_len=128, target_max_len=50, max_item_num=20)` produces `item_text_ids/item_text_masks [B,N,L]`, target tensors `[B,Y]`, and user IDs. Missing required files raise `FileNotFoundError`.

## Amazon 2023 and legacy P5

`Amazon2023ItemDataset` and `Amazon2023SASRecDataset` look under `root/processed/<split>`. The current `prepare_amazon2023()` instead writes a converted Amazon-2014-compatible layout under `root/raw/<category>`. Consequently, the current SASRec/HSTU Amazon 2023 configs use the ordinary 2014 adapters, not these two classes.

[`p5_amazon.py`](https://github.com/phonism/genrec/blob/main/genrec/data/p5_amazon.py) provides the legacy PyG/Polars path: `AmazonReviews`, `P5AmazonReviewsItemDataset`, and `P5AmazonReviewsSeqDataset`. The RQ-VAE trainer's Python default still names the P5 item adapter, but current downstream Gin configs inject `AmazonItemDataset` instead. The legacy path requires PyG, Polars, and SentenceTransformers.

See [sequence contracts](base-dataset.md), [data routing](dataset-factory.md), and [processing](processors.md).
