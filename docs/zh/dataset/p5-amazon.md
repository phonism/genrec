# Amazon 2014 与 P5 兼容层

仓库历史文档把主数据路径统称为“P5 Amazon”，但当前代码实际上有两套实现：新的 Amazon 2014 模型专用 adapters，以及保留在 `p5_amazon.py` 的旧 PyTorch Geometric/P5 兼容管线。正式 benchmark 使用前者。

## 当前主路径：Amazon Reviews 2014 5-core

原始文件来自 Stanford SNAP Amazon Product Data：

```text
dataset/amazon/raw/<split>/reviews_<Category>_5.json.gz
dataset/amazon/raw/<split>/meta_<Category>.json.gz
```

`_5` 文件已经是来源方提供的 5-core 版本。当前 `DATASET_CONFIGS` 包含：

| `--split` | review 文件 | meta 文件 |
|---|---|---|
| `beauty` | `reviews_Beauty_5.json.gz` | `meta_Beauty.json.gz` |
| `sports` | `reviews_Sports_and_Outdoors_5.json.gz` | `meta_Sports_and_Outdoors.json.gz` |
| `toys` | `reviews_Toys_and_Games_5.json.gz` | `meta_Toys_and_Games.json.gz` |
| `clothing` | `reviews_Clothing_Shoes_and_Jewelry_5.json.gz` | `meta_Clothing_Shoes_and_Jewelry.json.gz` |
| `home` | `reviews_Home_and_Kitchen_5.json.gz` | `meta_Home_and_Kitchen.json.gz` |

`books` 与 `arts_crafts_and_sewing` 条目用于读取 Amazon 2023 转换后的兼容文件，不应误认为 Stanford 2014 同名原始类别。

## review 与 metadata 字段

模型 adapters 实际使用的最小字段：

| 来源 | 字段 | 用途 |
|---|---|---|
| review | `asin` | item key |
| review | `reviewerID` | user key |
| review | `unixReviewTime` | 序列排序与 temporal protocol |
| metadata | `asin` | 与 item mapping 对齐 |
| metadata | `title`, `price`, `salesRank`, `brand`, `categories` | 当前 `AmazonItemDataset` 的文本 embedding |

rating 在当前 next-item adapters 中不是监督目标。缺少 metadata 的 review item 可能无法进入 item embedding 表，因此必须检查 embedding 行数与序列 mapping 是否一致。

## 下载与校验

```bash
python scripts/prepare_amazon2014.py \
  --splits beauty sports toys \
  --kinds reviews meta
```

脚本做四件事：跨会话排他锁、下载到隐藏临时文件、读到 gzip EOF 验证 CRC、写 SHA-256 manifest。不要用浏览器中断下载后的文件直接覆盖最终路径。

`AmazonItemDataset` 也能在文件缺失时下载，但显式准备脚本更适合共享实验机，因为 provenance 和并发行为更清楚。

## 文本 embedding

当前 metadata 模板在 `AmazonItemDataset._generate_embeddings_unlocked` 中定义，近似为：

```text
'title':...
 'price':...
 'salesRank':...
 'brand':...
 'categories':...
```

SentenceTransformer 以 batch 32 编码，结果保存为每行：

| 列 | 含义 |
|---|---|
| `ItemID` | 当前 mapping 中的整数 ID |
| `embedding` | Python list 形式的向量 |

缓存路径包含 encoder basename，例如 `item_emb_sentence-t5-xl.parquet`。不同模型：TIGER/LCRec/COBRA/RPG 常用 Sentence-T5，OneRec 的 RQ-KMeans 使用 Qwen3-Embedding-8B。

## 用户序列

各 adapter 都从 review 重建 mapping 并按 timestamp 排序，但训练样本不同：

- SASRec/HSTU：每用户一条训练序列，collate 内 shift 多位置监督；
- TIGER/LCRec/GRAM：常见 sliding prefixes；
- OneRec：按任务生成自然语言/SID prompt；
- RPG：1-based item IDs 和右 padding；
- NoteLLM：把相邻 item 组成 item-to-item retrieval 对。

所以不能把一个 adapter pickled 的整数序列直接交给另一个 adapter。

## 旧 `p5_amazon.py`

真实存在的类是：

| 类 | 用途 |
|---|---|
| `AmazonReviews` | PyG `InMemoryDataset`，构建异构 user–item 图与历史 |
| `P5AmazonReviewsItemDataset` | 从 PyG processed data 读取 item embedding |
| `P5AmazonReviewsSeqDataset` | 加载 RQ-VAE，把历史 item 转为 SID `SeqData` |

旧页面曾使用 `P5AmazonItemDataset`、`P5AmazonSequenceDataset` 等名称；这些类在当前仓库不存在。

该文件导入 `torch_geometric`，使用独立 processed artifact 和固定/历史参数，当前下游 Gin 已改为明确绑定 `AmazonItemDataset` 或模型专用 adapter。只有复现旧 checkpoint 时才应选择 P5 兼容层，并记录其数据版本。

## 最小读取示例

训练 item tokenizer：

```python
from genrec.data.amazon import AmazonItemDataset

items = AmazonItemDataset(
    root="dataset/amazon",
    split="beauty",
    train_test_split="all",
    encoder_model_name="/data/models/sentence-t5-xl",
)
print(len(items), items.dim)
```

检查序列基线：

```python
from genrec.data.amazon_sasrec import AmazonSASRecDataset

test = AmazonSASRecDataset(
    root="dataset/amazon",
    split="beauty",
    train_test_split="test",
    max_seq_len=50,
)
print(len(test), test.num_items, test[0])
```

dataset 构造可能触发大文件读取或模型编码；交互式示例只适合已准备好缓存的环境。

## 引用与许可

- [Amazon Product Data (2014)](https://jmcauley.ucsd.edu/data/amazon/)
- [P5: Recommendation as Language Processing](https://arxiv.org/abs/2203.13366)

使用数据时遵守原数据集条款。本仓库的 MIT 许可证不重新授权第三方数据或预训练模型。
