# 数据集 API

所有数据类直接继承 PyTorch `Dataset`；没有统一的仓库基类或 factory。构造函数通常执行下载/读取、embedding/SID cache 初始化和样本生成，可能很昂贵。这里列出当前可调用类、关键构造参数和 `__getitem__()` schema；完整默认值以链接的源文件为准。

## 共享命名约定

- `root`：数据根目录。
- `split`：Amazon category，例如 `beauty`；GRAM 使用大小写敏感的 `Beauty` 风格目录。
- `train_test_split`：通常是 `train|valid|test`；item tokenizer 数据集还使用 `all|train|eval`。
- `max_seq_len`：历史 item 数；对文本 adapter 不一定等于 tokenizer token 长度。
- `evaluation_protocol`：支持它的 Amazon 2014 adapter 接受 `leave_one_out|global_temporal`。
- SASRec/HSTU/RPG 使用 0 作为 padding，真实 item ID 通常从 1 开始；RQVAE/TIGER 旧路径的内部映射需以具体 dataset 为准。

## Amazon 2014 通用数据

源文件：[`genrec/data/amazon.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon.py)

### `AmazonItemDataset`

```python
AmazonItemDataset(
    root="dataset/amazon",
    split="beauty",
    train_test_split="all",       # all | train | eval
    encoder_model_name="sentence-transformers/sentence-t5-base",
    force_regenerate=False,
)
```

`__getitem__(idx) -> list[float]`，长度为 encoder 输出维度。公开属性包括 `embeddings: np.ndarray[N,D]`、`dim`。文本 embedding cache 会按 split/encoder 写入 processed 目录。未知 Amazon category 抛 `ValueError`；下载或 metadata 文件错误向上传播。

### `AmazonSeqDataset`

```python
AmazonSeqDataset(
    root="dataset/amazon", split="beauty",
    train_test_split="train", max_seq_len=20,
    subsample=True, add_disambiguation=True, dedup_sinkhorn=False,
    pretrained_rqvae_path="...",
    encoder_model_name="...",
    rqvae_input_dim=768, rqvae_embed_dim=32,
    rqvae_hidden_dims=[512, 256, 128, 64],
    rqvae_codebook_size=256, rqvae_n_layers=3,
    evaluation_protocol="leave_one_out",
    temporal_train_ratio=0.8, temporal_valid_ratio=0.1,
)
```

`__getitem__() -> SeqData(user_id, item_ids, target_ids, target_item)`。`item_ids` 为历史 SID 展平列表，`target_ids` 长度为 codebook 层数。公开属性 `sem_ids_list`、`sequences`、`evaluation_protocol` 被 TIGER evaluator 使用。RQ-VAE 几何或 encoder 与 checkpoint 不匹配时通常在 `load_state_dict`/矩阵运算处失败。

## 判别式与向量检索 adapter

### `AmazonSASRecDataset`

源文件：[`amazon_sasrec.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_sasrec.py)

```python
AmazonSASRecDataset(
    root="dataset/amazon", split="beauty",
    train_test_split="train", max_seq_len=50, min_seq_len=5,
    evaluation_protocol="leave_one_out",
    temporal_train_ratio=0.8, temporal_valid_ratio=0.1,
)
```

样本：`{"history": list[int], "target": int}`。属性 `num_items` 是不含 padding 的 catalog 数。

```python
sasrec_collate_fn(batch, max_seq_len=50, num_items=0)
# -> input_ids [B,L], targets [B,L]
# num_items > 0 时另有 negatives [B,L]

sasrec_eval_collate_fn(batch, max_seq_len=50)
# -> input_ids [B,L], targets [B]
```

训练 collator 左 padding；`num_items>0` 时用 Python RNG 每个非 padding 位置采一个不等于 positive 的负 item。

### `AmazonHSTUDataset`

源文件：[`amazon_hstu.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_hstu.py)

构造参数与 SASRec 基本相同。样本为 `history`、`timestamps`、`target`。

```python
hstu_collate_fn(batch, max_seq_len=50)
# -> input_ids [B,L], targets [B,L], timestamps [B,L]

hstu_eval_collate_fn(batch, max_seq_len=50)
# -> input_ids [B,L], targets [B], timestamps [B,L]
```

### `MovieLens1MHSTUDataset`

源文件：[`ml1m_hstu.py`](https://github.com/phonism/genrec/blob/main/genrec/data/ml1m_hstu.py)

```python
MovieLens1MHSTUDataset(
    root="dataset/ml1m", split="ml1m",
    train_test_split="train", max_seq_len=200, min_seq_len=5,
)
```

返回与 `AmazonHSTUDataset` 相同的三字段 dict，但没有统一 Amazon `evaluation_protocol` 参数。

### `AmazonNoteLLMDataset`

源文件：[`amazon_notellm.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_notellm.py)

```python
AmazonNoteLLMDataset(
    root="dataset/amazon", split="beauty",
    train_test_split="train", max_seq_len=50, min_seq_len=5,
    evaluation_protocol="leave_one_out",
    temporal_train_ratio=0.8, temporal_valid_ratio=0.1,
)
```

样本：`{"query": int, "target": int}`。它继承 SASRec adapter 的序列加载，但只产生相邻 item I2I 对；`item_texts: dict[int,str]` 提供 prompt 文本。

```python
build_item_text(meta: dict, fallback: str) -> str
format_item_prompt(item_text: str, item_token="[EMB]") -> str
notellm_collate_fn(batch) -> {"queries": list[int], "targets": list[int]}
```

## LLM/SID adapter

### `AmazonLCRecDataset`

源文件：[`amazon_lcrec.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_lcrec.py)

主要构造参数：`root`、`split`、`train_test_split`、`max_seq_len`、`max_text_len`、`pretrained_rqvae_path`、encoder/RQVAE 几何、`enabled_tasks`、`task_sample_weights` 和协议参数。

`__getitem__()` 按任务返回：

```python
{
  "task": "seqrec|item2index|index2item|fusionseqrec|itemsearch|preferenceobtain",
  "prompt": str,
  "response": str,
  # 除 preferenceobtain 外通常还有：
  "target_item": int,
  "target_sem_ids": list[int],
}
```

未知任务抛 `ValueError`。collation 在 [`lcrec_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/lcrec_trainer.py) 的 `lcrec_collate_fn`，不在数据模块。

### `AmazonOneRecSFTDataset`

源文件：[`amazon_onerec.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_onerec.py)

构造参数同时支持 `tokenizer_type="rqvae"|"rqkmeans"`，并分别接受两类 checkpoint/几何参数。其他关键参数为 `enabled_tasks`、`max_text_len` 和协议 ratios。

样本 schema 与 LCRec 类似，任务是 `sidsft|title2sid|sid2title|fusionseqrec`，固定返回 `task`、`prompt`、`response`、`target_item`、`target_sem_ids`。未知 tokenizer/task 抛 `ValueError`。

### `AmazonOneRecRLDataset`

同一源文件。没有 `train_test_split` 参数；它只构造 RL 训练 prompts。默认 `rl_tasks` 为 `sidsft|title2sid|title_seqrec`，`__getitem__()` 直接返回带 `prompt`、`target`/`target_sem_ids` 和 task 元数据的内部 sample dict。字段以 [`rl_collate_fn`](https://github.com/phonism/genrec/blob/main/genrec/trainers/onerec_grpo_trainer.py) 为消费契约。

### `AmazonOneRecCLMDataset`

源文件：[`amazon_onerec_clm.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_onerec_clm.py)

构造参数与 OneRec SFT 的 tokenizer 部分相同。训练样本为 `{"item_seq": list[int], "task": "clm"}`；valid/test 另含 `target`、`target_sem_ids`，task 为 `seqrec`。token 展平由 CLM trainer 的 mapper/collator 完成。

### `AmazonCobraDataset`

源文件：[`amazon_cobra.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_cobra.py)

关键构造参数为 RQVAE checkpoint/几何、text encoder、`max_text_len` 和协议。样本：

```python
{
  "input_ids": list[int],                 # 历史 SID，长度 T*C
  "encoder_input_ids": Tensor[T,L],       # 历史 item 文本 token
  "target_sem_ids": list[int],            # 长度 C
  "target_encoder_input_ids": Tensor[1,L],
  "target_item": int,
}
```

## RPG 与 GRAM { #rpg-and-gram }

### `AmazonRPGDataset`

源文件：[`amazon_rpg.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_rpg.py)

```python
AmazonRPGDataset(
    root="dataset/amazon", split="beauty",
    train_test_split="train", max_seq_len=50, min_seq_len=3,
    n_codebook=32, codebook_size=256,
    encoder_model_name="sentence-transformers/sentence-t5-xl",
    sent_emb_pca=0, sent_emb_batch_size=256,
    opq_use_gpu=False, opq_gpu_id=0, faiss_num_threads=32,
    evaluation_protocol="leave_one_out",
    temporal_train_ratio=0.8, temporal_valid_ratio=0.1,
)
```

训练样本：`input_ids: list[item_id]` 与逐位置 `labels`；评测 labels 为单 item 列表。`item_id2tokens: Tensor[num_items,C]` 供模型构造。

```python
rpg_train_collate_fn(...)  # input_ids/labels/attention_mask [B,L], seq_lens [B]
rpg_eval_collate_fn(...)   # input_ids/attention_mask [B,L], seq_lens [B], labels [B,1]
```

`codebook_size` 非 2 的幂会在 OPQ 生成时 assert；FAISS 未安装会在首次生成 SID 时 `ImportError`。

### `AmazonGRAMDataset` / `GRAMCollator`

源文件：[`amazon_gram.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon_gram.py)

```python
AmazonGRAMDataset(
    root="dataset/gram", split="Beauty", train_test_split="train",
    max_seq_len=20, max_text_len=128,
    item_id_file="item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt",
    item_prompt="all_text", top_k_similar_item=10,
    cf_model="sasrec", his_sep=" ; ",
    skip_empty_his=True, reverse_history=True,
)
```

样本：`{"input": list[str], "output": str, "user_id": str}`；`input[0]` 是用户粗粒度 prompt，其余为逐 item passage。缺任一必需文件抛 `FileNotFoundError`。

```python
GRAMCollator(tokenizer, item_prompt_max_len=128,
             target_max_len=50, max_item_num=20)(batch)
# -> item_text_ids/item_text_masks [B,N,L]
#    target_ids/target_masks [B,Y], user_ids list[str]
```

## Amazon 2023 与旧 P5

源文件：[`amazon2023.py`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon2023.py)

- `Amazon2023ItemDataset(...)`：`__getitem__ -> list[float]`；split 为 `root/processed/<split>` 目录名，如 `books_5core`。注意 `prepare_amazon2023()` 当前写的是 `root/raw/<category>` compatibility layout，不会创建此类所需的 `processed` 目录。
- `Amazon2023SASRecDataset(...)`：返回 `history`、`timestamps`、`target`，但没有统一 `global_temporal` 参数。当前 SASRec/HSTU trainer 没有选择这个类；Amazon 2023 Gin 走的是 `prepare_amazon2023()` 生成的 2014-compatible raw 文件与 `AmazonSASRecDataset` / `AmazonHSTUDataset`。
- `prepare_amazon2023(category="Books", k_core=5, output_dir="dataset/amazon2023", max_reviews=None, source="huggingface")`：下载、迭代 k-core、写 processed artifact。
- `download_amazon2023(...)` 对未知 category 抛 `ValueError`；processed 目录/文件缺失抛 `FileNotFoundError`。

源文件 [`p5_amazon.py`](https://github.com/phonism/genrec/blob/main/genrec/data/p5_amazon.py) 提供旧 PyG/P5 数据链：`AmazonReviews`、`P5AmazonReviewsItemDataset`、`P5AmazonReviewsSeqDataset`。当前 RQ-VAE 配置仍使用 item dataset；它额外依赖 PyG、Polars、SentenceTransformers。P5 sequence dataset 返回 `SeqData`，但属于兼容路径，不是统一基类。

## 相关文档

- [数据集基础契约](base-dataset.md)
- [数据选择与注入](dataset-factory.md)
- [预处理入口](processors.md)
- [模型总览](../models/overview.md)
