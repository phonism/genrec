# Processing entry points

## There is no processors package

The source tree has no `genrec.data.processors`, `TextProcessor`, `SequenceProcessor`, `DataAugmentor`, or `PreprocessingPipeline`. Processing lives in concrete dataset constructors, helpers in the same modules, and command-line scripts. Cache identity is also adapter-specific.

## Amazon 2014 helpers

Source: [`genrec/data/amazon.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon.py)

```python
download_file(url: str, dest_path: str) -> None
parse_gzip_json(path: str) -> Iterator[dict]
```

`download_file` skips an existing destination and otherwise streams the response. `parse_gzip_json` parses one gzip line at a time; for legacy compatibility it falls back to Python `eval()` after JSON failure. Use it only for trusted raw files.

Download, item mapping, metadata text, and embedding-cache construction otherwise occur inside concrete Amazon adapters. Protocol-aware sequence selection is shared through [`select_protocol_sequences`](base-dataset.md#evaluation-protocol-selector), but each model builds samples itself.

## Amazon 2023 download and k-core filtering

Source: [`genrec/data/amazon2023.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon2023.py)

```python
download_with_progress(url, dest_path, desc="Downloading") -> None
download_amazon2023(category, output_dir, download_meta=True,
                    source="huggingface") -> tuple[str, str | None]
parse_jsonl(path, max_lines=None) -> Iterator[dict]
apply_k_core_filter(reviews_path, k=5, max_reviews=None) \
    -> tuple[dict, dict[str, int], dict[str, int]]
process_metadata(meta_path, item_id_map) -> dict[int, dict]
save_processed_data(user_reviews, user_id_map, item_id_map,
                    item_meta, output_dir, category, k_core) -> str
prepare_amazon2023(category="Books", k_core=5,
                   output_dir="dataset/amazon2023", max_reviews=None,
                   source="huggingface") -> str
```

`apply_k_core_filter` repeatedly removes users/items below `k` until stable. Unknown categories raise `ValueError`; malformed JSONL lines are skipped.

```bash
python -m genrec.data.amazon2023 --category Books --k_core 5
python -m genrec.data.amazon2023 --list_categories
```

`prepare_amazon2023()` writes converted, Amazon-2014-compatible files under `output_dir/raw/<category>/`. It does **not** populate the `root/processed/<split>/` layout expected by the separate `Amazon2023ItemDataset` and `Amazon2023SASRecDataset` classes, and it does not train a semantic tokenizer.

## Text helpers

NoteLLM, [`genrec/data/amazon_notellm.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_notellm.py):

```python
build_item_text(meta: dict, fallback: str) -> str
format_item_prompt(item_text: str, item_token: str = "[EMB]") -> str
```

The former flattens title, brand, categories, and description and truncates to 4,000 characters. The latter removes duplicate markers and guarantees exactly one marker at the end.

GRAM, [`genrec/data/amazon_gram.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_gram.py):

```python
read_lines(path: str) -> list[str]
parse_dict_from_lines(lines: list[str]) -> dict[str, str]
load_similar_items(path: str, top_k: int = 10) -> dict[str, list[str]]
```

Missing files raise `FileNotFoundError`. These functions consume externally prepared text/lexical-ID artifacts; the semantic-to-lexical generator is not included.

## Semantic tokenizer helpers

RPG has private OPQ helpers in [`amazon_rpg.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_rpg.py):

```python
_build_meta_sentence(meta: dict) -> str
_encode_sentence_embeddings(sentences, model_name, batch_size=256) -> np.ndarray
_generate_opq_semantic_ids(sent_embs, train_mask, n_codebook,
                           codebook_size, pca_dim=0, use_gpu=False,
                           gpu_id=0, num_threads=32) -> np.ndarray
```

`codebook_size` must be a power of two. FAISS is imported lazily and missing FAISS raises `ImportError`. Prefer constructing `AmazonRPGDataset` over depending on these private names.

The other tokenizer calls are [`RqVae.get_semantic_ids`](rqvae.md) for tensors and `RqKmeans.assign` for NumPy arrays. Encoder, checkpoint geometry, and catalog must be managed as one artifact set.

## Cache behavior

There is no shared `CacheManager`. Some Amazon/RPG/NoteLLM caches use `fcntl.flock`; several writers commit a PID-suffixed temporary file with `os.replace`. Cache keys do not consistently include metadata hashes, model revisions, and every hyperparameter. Inspect the target adapter before rebuilding data after a text-template or encoder change.

Useful diagnostics include [`verify_codebook_health.py`](https://github.com/phonism/genrec/blob/main/scripts/verify_codebook_health.py), [`report_sid_collisions.py`](https://github.com/phonism/genrec/blob/main/scripts/report_sid_collisions.py), [`dump_sem_ids.py`](https://github.com/phonism/genrec/blob/main/scripts/dump_sem_ids.py), and [`validate_dedup.py`](https://github.com/phonism/genrec/blob/main/scripts/validate_dedup.py).

See [Datasets](datasets.md), [configuration](configs.md), and [RQ-VAE](rqvae.md).
