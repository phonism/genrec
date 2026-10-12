# 数据集总览

GenRec 没有一个所有模型继承的“万能 Dataset 基类”。每个模型 adapter 直接实现 `torch.utils.data.Dataset`，共享的是原始 Amazon 文件、协议函数和若干 schema。阅读代码时应从 trainer 传入的具体 dataset class 开始。

## 数据源

| 数据源 | 入口 | 主要用途 | 状态 |
|---|---|---|---|
| Amazon Reviews 2014 5-core | `genrec/data/amazon.py` 及各 `amazon_<model>.py` | 主要 benchmark；Beauty/Sports/Toys/Home/Clothing | 核心路径 |
| Amazon Reviews 2023 | `genrec/data/amazon2023.py` | 下载 JSONL、迭代 5-core、转换为兼容 gzip | 预处理 + 部分模型 config |
| MovieLens 1M | `genrec/data/ml1m_hstu.py` | HSTU 小规模对齐/消融 | 仅 HSTU adapter |
| GRAM 论文式预处理数据 | `genrec/data/amazon_gram.py` | lexical IDs、item text、CF similar items | 独立目录，不能由普通 Amazon adapter 直接替代 |
| 旧 P5 / PyG 管线 | `genrec/data/p5_amazon.py` | 早期通用 item/sequence 数据实验 | 保留兼容，不是当前 benchmark 主路径 |

## 模型—adapter 矩阵

| 模型 | Dataset | item ID 约定 | 关键输出 |
|---|---|---|---|
| SASRec | `AmazonSASRecDataset` | 1-based，0 padding | history、target |
| HSTU | `AmazonHSTUDataset` | 1-based，0 padding | history、timestamp、target |
| RQ-VAE / RQ-KMeans | `AmazonItemDataset` | parquet 行与序列 mapping 对齐 | continuous item embedding |
| TIGER | `AmazonSeqDataset` | 0-based item；SID 单独编码 | `SeqData` + raw `target_item` |
| LCRec | `AmazonLCRecDataset` | 0-based | 多任务 prompt/response、SID catalog |
| OneRec SFT/GRPO | `AmazonOneRecSFTDataset`, `AmazonOneRecRLDataset` | 0-based | 四类 SFT task 或 RL prompt |
| OneRec CLM | `AmazonOneRecCLMDataset` | 0-based | 每用户 flat token stream |
| COBRA | `AmazonCobraDataset` | 0-based | sparse codes + dense text tokens |
| RPG | `AmazonRPGDataset` | 1-based，0 padding | item sequence + OPQ tokens |
| NoteLLM | `AmazonNoteLLMDataset` | 继承 SASRec 的 1-based ID | item-to-item query/target + item text |
| GRAM | `AmazonGRAMDataset` | 原始字符串 ID + lexical ID | user passage + per-item passages |

!!! danger "不要跨 adapter 复用整数 item ID"
    同一个 ASIN 在不同 adapter 中可能是 0-based 或 1-based，映射又由原始文件首次出现顺序构造。checkpoint、SID 数组和预测 dump 必须与生成它们的 mapping 同源；“数字相同”不代表同一 item。

## Amazon 2014 文件

核心目录：

```text
dataset/amazon/
├── raw/
│   └── beauty/
│       ├── reviews_Beauty_5.json.gz
│       └── meta_Beauty.json.gz
└── processed/
    └── beauty/
        ├── item_emb_sentence-t5-xl.parquet
        └── item_emb_Qwen3-Embedding-8B.parquet
```

安全准备：

```bash
python scripts/prepare_amazon2014.py \
  --splits beauty sports toys --kinds reviews meta
```

该脚本目前显式覆盖 Beauty、Sports、Toys。其他 `DATASET_CONFIGS` 类别可由 `AmazonItemDataset` 的下载逻辑处理，或按相同文件名手工准备；正式 run 应保存 checksum manifest。

## Amazon 2023 转换

```bash
python -m genrec.data.amazon2023 --list_categories
python -m genrec.data.amazon2023 \
  --category Arts_Crafts_and_Sewing \
  --k_core 5 \
  --output_dir dataset/amazon2023
```

预处理会下载 review/meta JSONL，反复过滤直到 user 和 item 都满足 k-core，再写成现有 Amazon adapters 可读取的兼容文件。`--max_reviews` 只用于测试，不得把截断数据与完整数据 benchmark 混用。

## 两种序列协议

共享函数 `select_protocol_sequences` 位于 `genrec/data/evaluation_protocol.py`：

- `leave_one_out`：每位用户最后一项 test、倒数第二项 valid；
- `global_temporal`：用全局 timestamp quantile 建立 train/valid cutoffs，避免晚于 cutoff 的交互进入相应历史。

具体 adapter 再把协议输出变成 sliding windows 或一用户一样本。协议相同不意味着训练 sample augmentation 相同。

`global_temporal` 仍是 transductive catalog 设置：metadata、5-core 和 SID 构建可以看到完整 item 集，它不是严格 cold-item inductive protocol。

## 文本 embedding 缓存

`AmazonItemDataset` 把 metadata 格式化为 title、price、salesRank、brand、categories 文本，经 SentenceTransformer 编码并保存 parquet。生成过程使用：

- `<parquet>.lock` 排他文件锁；
- `<parquet>.part.<pid>` 临时文件；
- 完成后 `os.replace` 原子发布。

RPG 有自己的 embedding/OPQ 缓存和 training-item-only 拟合约束；OneRec 使用 Qwen3-Embedding-8B。不要仅凭 embedding 维度判断两个缓存可交换。

## 数据质量检查

```bash
python scripts/verify_dataset.py --skip_rqvae
python scripts/verify_dataset.py \
  --rqvae_path out/tiger/amazon/beauty/rqvae/checkpoint_epoch_4999.pt
python scripts/prepare_item_embeddings.py --help
```

`scripts/diagnose_dataset.py` 仍硬编码旧 checkpoint，`scripts/check_data_leakage.py` 仍依赖仓库外的绝对路径；两者属于历史现场诊断脚本，不是可移植的通用入口。正式泄漏检查应以统一 protocol、target 时间边界和矩阵验证器为准。

最低检查项：

1. raw 文件 checksum 与 record count；
2. user/item 数与 5-core 条件；
3. 每个 split 的 target 时间边界；
4. adapter 的 item mapping 与 embedding/SID 行数一致；
5. train/valid/test sample 数；
6. SID collision、codebook health；
7. embedding cache 的 encoder 名称、revision、维度和 hash。

## 深入阅读

- [Amazon / P5 文件说明](p5-amazon.md)
- [自定义数据集](custom.md)
- [代码级数据流水线](../architecture/data-pipeline.md)
- [评测系统](../architecture/evaluation-system.md)
- [领域常用数据集](../landscape/datasets.md)
