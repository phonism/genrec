# 数据预处理入口

## 先澄清：没有 `genrec.data.processors`

当前源码不存在 `TextProcessor`、`SequenceProcessor`、`DataAugmentor` 或 `PreprocessingPipeline`。预处理逻辑分布在具体 dataset 构造函数、同文件 helper 和 `scripts/` 中；缓存身份也由各 adapter 自己定义。

本页保留旧导航文件名，只提供真实入口索引。若需要跨数据集统一 preprocessing，应新增经过测试的模块，而不是调用旧文档中的虚构类。

## Amazon 2014

源文件：[`genrec/data/amazon.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon.py)

```python
download_file(url: str, dest_path: str) -> None
parse_gzip_json(path: str) -> Iterator[dict]
```

- `download_file` 若目标已存在会直接返回；否则流式下载到目标路径。网络/HTTP/写盘异常向上传播。
- `parse_gzip_json` 按行解析 gzip JSON；JSON 失败时会退回 Python `eval()`，两者都失败才跳过该行。因此只能用于可信 raw 文件，不要让不受信任内容进入这条兼容路径。
- raw 下载、item mapping、metadata text 和 embedding cache 主要在 `AmazonItemDataset` / `AmazonSeqDataset.__init__()` 的私有方法中完成，不是独立公共 processor。

SASRec/HSTU/RPG/NoteLLM 的数据模块各自再次实现 protocol-aware sequence sample builder。共享的只有 [`select_protocol_sequences`](base-dataset.md#shared-protocol-functions)。

## Amazon 2023 下载与 k-core { #amazon-2023-download-k-core }

源文件：[`genrec/data/amazon2023.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon2023.py)

```python
download_with_progress(
    url: str, dest_path: str, desc: str = "Downloading"
) -> None

download_amazon2023(
    category: str,
    output_dir: str,
    download_meta: bool = True,
    source: str = "huggingface",
) -> tuple[str, str | None]

parse_jsonl(
    path: str, max_lines: int | None = None
) -> Iterator[dict]

apply_k_core_filter(
    reviews_path: str,
    k: int = 5,
    max_reviews: int | None = None,
) -> tuple[dict[str, list[dict]], dict[str, int], dict[str, int]]

process_metadata(
    meta_path: str,
    item_id_map: dict[str, int],
) -> dict[int, dict]

save_processed_data(
    user_reviews, user_id_map, item_id_map,
    item_meta, output_dir, category, k_core,
) -> str

prepare_amazon2023(
    category: str = "Books",
    k_core: int = 5,
    output_dir: str = "dataset/amazon2023",
    max_reviews: int | None = None,
    source: str = "huggingface",
) -> str
```

`apply_k_core_filter` 迭代删除 user/item interaction 少于 `k` 的记录，直到数量稳定。未知 category 在 `download_amazon2023` 抛 `ValueError`。`parse_jsonl` 跳过 JSON decode 失败的行，不会为每个坏行抛错。

CLI：

```bash
python -m genrec.data.amazon2023 --category Books --k_core 5
python -m genrec.data.amazon2023 --list_categories
```

`prepare_amazon2023()` 实际把转换结果写到 `output_dir/raw/<category>/` 并返回该目录，用于 `AmazonSASRecDataset` / `AmazonHSTUDataset` 的 2014-compatible 路径；名称中的 “processed” 指已转换数据，不是 `output_dir/processed/...` 目录。另一方面，`Amazon2023ItemDataset` / `Amazon2023SASRecDataset` 构造函数查找 `root/processed/<split>/`，当前函数不会填充那套目录，两条路径不可混用。它也不会自动训练 semantic tokenizer。

## 文本格式化与 embedding

### NoteLLM

源文件：[`genrec/data/amazon_notellm.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_notellm.py)

```python
build_item_text(meta: dict, fallback: str) -> str
format_item_prompt(item_text: str, item_token: str = "[EMB]") -> str
```

`build_item_text` 按 title、brand、categories、description 固定顺序展平，字符串最多 4000 字符；缺字段时返回 `item: <fallback>`。`format_item_prompt` 会先从内容中移除已有 marker，再保证末尾恰有一个 `[EMB]`。

### GRAM 文本文件

源文件：[`genrec/data/amazon_gram.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_gram.py)

```python
read_lines(path: str) -> list[str]
parse_dict_from_lines(lines: list[str]) -> dict[str, str]
load_similar_items(path: str, top_k: int = 10) -> dict[str, list[str]]
```

`read_lines` 对缺文件抛 `FileNotFoundError`。`parse_dict_from_lines` 只按第一个空格拆 key/value，不验证重复 key 或 lexical-ID 唯一性。GRAM 的 semantic-to-lexical 生成器不在仓库；这些函数只读取外部预处理结果。

### RPG OPQ

源文件：[`genrec/data/amazon_rpg.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_rpg.py)

以下是实现 helper，名称以下划线开头，不承诺公共 API：

```python
_build_meta_sentence(meta: dict) -> str
_encode_sentence_embeddings(
    sentences: list[str], model_name: str, batch_size: int = 256
) -> np.ndarray          # [N,D], float32
_generate_opq_semantic_ids(
    sent_embs: np.ndarray, train_mask: np.ndarray,
    n_codebook: int, codebook_size: int,
    pca_dim: int = 0, use_gpu: bool = False,
    gpu_id: int = 0, num_threads: int = 32,
) -> np.ndarray          # [N,C]
```

`codebook_size` 必须是 2 的幂；否则 assert。FAISS 只在函数调用时 import，未安装会抛 `ImportError`。正常用法是构造 `AmazonRPGDataset`，不应单独依赖私有 helper 名称。

### RQVAE / RQ-KMeans item tokenization

两类 tokenizer 不是“processor 类”：

- [`RqVae.get_semantic_ids`](rqvae.md)：`Tensor[N,D] -> sem_ids Tensor[N,C]`。
- [`RqKmeans.assign`](../models/rqkmeans.md)：`np.ndarray[N,D] -> RqKmeansOutput.sem_ids[N,C]`。

下游 dataset 在构造时加载 checkpoint 并把 item embedding 转 SID。encoder、checkpoint 几何和 catalog 必须作为同一 artifact 组管理。

## 旧 P5 preprocessing

源文件：[`genrec/data/p5_amazon.py`](https://github.com/phonism/genrec/blob/main/genrec/data/p5_amazon.py)

`PreprocessingMixin` 包含 `_process_genres`、`_remove_low_occurrence`、`_encode_text_feature`、`_rolling_window`、`_ordered_train_test_split`、`_df_to_tensor_dict` 和 `_generate_user_history`。它们全是以下划线开头的内部 static method，服务 `AmazonReviews(InMemoryDataset)` 的 PyG/Polars 管线，不是通用 processor API。

## Cache 与并发行为

没有共享 `CacheManager`。当前实现采用以下局部策略：

- Amazon/RPG/NoteLLM 的部分 cache 使用 `fcntl.flock` 防止同机重复生成；Windows 不保证可用。
- 多个 writer 常用 `.part.<pid>` + `os.replace` 原子提交。
- cache key 通常包含 split 和 encoder basename，但未必包含 metadata hash、model revision 或所有超参。
- RQVAE/RQ-KMeans checkpoint 的 provenance 不能仅靠 cache 文件名恢复。

修改文本模板、metadata 或 encoder 后，应先确认目标 adapter 的具体 cache identity，再定点重建；不要假定统一 `force_reload` 参数在所有类上存在。

## 诊断脚本

| 脚本 | 输入 / 输出 |
|---|---|
| [`verify_codebook_health.py`](https://github.com/phonism/genrec/blob/main/scripts/verify_codebook_health.py) | RQVAE/RQ-KMeans checkpoint → usage/collision 报告 |
| [`report_sid_collisions.py`](https://github.com/phonism/genrec/blob/main/scripts/report_sid_collisions.py) | `.npy/.json/text` SID → prefix collision/Markdown/JSON |
| [`dump_sem_ids.py`](https://github.com/phonism/genrec/blob/main/scripts/dump_sem_ids.py) | tokenizer checkpoint → SID artifact |
| [`validate_dedup.py`](https://github.com/phonism/genrec/blob/main/scripts/validate_dedup.py) | RQVAE collision dedup 诊断 |

脚本是 CLI，不由 `genrec.data` 重导出。

## 历史 API 迁移

| 旧文档调用 | 当前实现 |
|---|---|
| `TextProcessor.encode_texts(...)` | 具体 item dataset 的 embedding cache，或直接调用选定 encoder |
| `SequenceProcessor.create_training_samples(...)` | 每个模型 dataset 的 `_generate_samples()` 私有实现 |
| `DataAugmentor` | 无通用实现；训练 dataset 内可能做滑窗/随机 subsample |
| `PreprocessingPipeline.process(...)` | 构造具体 dataset 或运行 Amazon 2023 CLI |
| `visualize_embeddings(...)` | 无此库函数；使用 `scripts/` 中现有诊断或外部绘图库 |

## 相关文档

- [数据集 API](datasets.md)
- [配置与 CLI](configs.md)
- [RQ-VAE API](rqvae.md)
- [RQ-KMeans 说明](../models/rqkmeans.md)
