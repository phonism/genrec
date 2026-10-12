# 命令索引

本页是从仓库根目录运行 GenRec 的入口清单。命令、Gin selector、配置路径和依赖已按 **2026-08-31 的当前代码**核验；它回答“入口是否存在、要先准备什么”，不把“代码可启动”误写成“论文已复现”。模型的复现结论以[实现与复现状态](model-status.md)为准，参数与张量契约见[架构文档](../architecture/index.md)。

## 状态约定

| 状态 | 含义 |
|---|---|
| 主路径 | 当前训练/评测闭环，并写入项目规定的输出 |
| 前置阶段 | tokenizer、量化器或上游资产，不作为独立推荐器比较 |
| 实验性 | 入口存在，但论文对齐、数据覆盖或复现证据仍有限 |
| 诊断 | 只用于检查、smoke 或局部重评，不产生正式 benchmark |
| 受限 | 代码或配置存在，但有已知契约缺口；按备注处理后才能使用 |
| 外部资产 | 仓库只消费产物，没有对应的生成入口 |

所有 Gin trainer 都接受一个位置参数配置文件、可选的 `--split`，以及可重复的 `--gin`：

```bash
python -m genrec.trainers.sasrec_trainer \
  config/sasrec/amazon.gin \
  --split beauty \
  --gin 'train.seed=42'
```

`--split` 会在 Gin 解析前做字面 `{split}` 替换。它不是环境变量展开，也不会自动选择数据版本。顶层函数不是 `train` 时，要用真实 selector：`fit_rqkmeans`、`train_sft`、`train_grpo`、`train_clm`、`train_rpg` 或 `train_gram`。

## Baselines

### Amazon 2014

| 模型 | 命令 | 配置 | 前置依赖 | 状态 |
|---|---|---|---|---|
| SASRec | `python -m genrec.trainers.sasrec_trainer config/sasrec/amazon.gin --split beauty` | `config/sasrec/amazon.gin` | Amazon 2014 raw；无 tokenizer | 主路径；full-catalog item-level |
| HSTU-CE | `python -m genrec.trainers.hstu_trainer config/hstu/amazon.gin --split beauty` | `config/hstu/amazon.gin` | Amazon 2014 raw；无 tokenizer | 主路径；full-catalog item-level |
| HSTU sampled softmax | `python -m genrec.trainers.hstu_trainer config/hstu/amazon_ss.gin --split beauty` | `config/hstu/amazon_ss.gin` | 同上 | 主路径；`local` 为每个正例独立从本地完整 item table 采样，不是 in-batch；不能与旧 global 日志混报 |

`beauty`、`sports`、`toys`、`clothing`、`home` 均在 `genrec.data.amazon.DATASET_CONFIGS` 中注册；原子下载脚本只覆盖前三项，其他 split 会由 dataset 构造器按需下载。

### Amazon 2023

先用本页“数据准备”中的 Amazon 2023 命令生成 `dataset/amazon2023/raw/<split>/`，再运行：

```bash
python -m genrec.trainers.sasrec_trainer \
  config/sasrec/amazon2023.gin --split books

python -m genrec.trainers.hstu_trainer \
  config/hstu/amazon2023.gin --split arts_crafts_and_sewing
```

| 配置 | 前置依赖 | 状态 |
|---|---|---|
| `config/sasrec/amazon2023.gin` | `books` 已在通用 Amazon 映射中注册 | 主路径配置存在；当前只显式注册 Books 与 Arts/Crafts 两类文件名 |
| `config/hstu/amazon2023.gin` | `arts_crafts_and_sewing` 已注册 | 主路径配置存在；评测协议由 `AmazonHSTUDataset` 绑定 |

### MovieLens-1M

HSTU trainer 有数据适配器，但没有独立 Gin 文件。下面的覆盖项因此是命令的一部分，复现实验必须完整记录：

```bash
python -m genrec.trainers.hstu_trainer config/hstu/amazon.gin \
  --split ml1m \
  --gin 'train.dataset_type="ml1m"' \
  --gin 'train.dataset_folder="dataset/ml1m"' \
  --gin 'train.save_dir_root="out/hstu/ml1m"'
```

`MovieLens1MHSTUDataset` 会把 GroupLens 的 `ml-1m.zip` 下载到 `dataset/ml1m/`，执行迭代 5-core 后复用 HSTU collator。状态为主路径适配器，但它没有 Amazon 的 `global_temporal` configurable，结果协议当前记录为默认 `leave_one_out`。

## Tokenizers 与 item 表示

### RQ-VAE

TIGER、LCRec 和 COBRA 的 RQ-VAE geometry 不同，checkpoint 不能互换：

```bash
# TIGER：3 层、K=256、embed_dim=32
python -m genrec.trainers.rqvae_trainer \
  config/tiger/amazon/rqvae.gin --split beauty

# LCRec：5 层、K=256、embed_dim=64
python -m genrec.trainers.rqvae_trainer \
  config/lcrec/amazon/rqvae.gin --split beauty

# LCRec 论文配方：4 层、K=256、USM 下游映射
python -m genrec.trainers.rqvae_trainer \
  config/lcrec/amazon/rqvae_paper_recipe.gin --split beauty

# COBRA：3 层、K=32、embed_dim=32
python -m genrec.trainers.rqvae_trainer \
  config/cobra/amazon/rqvae.gin --split beauty

# COBRA 论文公开字段：title/price/category/description
python -m genrec.trainers.rqvae_trainer \
  config/cobra/amazon/rqvae_paper_attributes.gin --split beauty
```

| 消费方 | 配置 | 默认消费路径 | 前置依赖 | 状态 |
|---|---|---|---|---|
| TIGER | `config/tiger/amazon/rqvae.gin` | `out/tiger/amazon/<split>/rqvae/checkpoint_epoch_4999.pt` | Sentence-T5-XL item embeddings | 前置阶段 |
| LCRec legacy | `config/lcrec/amazon/rqvae.gin` | `out/lcrec/amazon/<split>/rqvae/checkpoint_epoch_1999.pt` | Sentence-T5-XL item embeddings | 5 层 adaptation；不是论文 4 层配方 |
| LCRec paper recipe | `config/lcrec/amazon/rqvae_paper_recipe.gin` | `out/lcrec/amazon/<split>/rqvae_paper_recipe/checkpoint_best_health.pt` | Sentence-T5-XL item embeddings | 4 层 indexer；按 full-tuple collision 选点，下游启用 USM；5% raw layer usage 仅用于拦截 one-code collapse（LC-Rec 论文无 80% 门槛） |
| COBRA | `config/cobra/amazon/rqvae.gin` | `out/cobra/amazon/<split>/rqvae/checkpoint_epoch_4999.pt` | Sentence-T5-XL item embeddings | 前置阶段 |
| COBRA paper fields | `config/cobra/amazon/rqvae_paper_attributes.gin` | `out/cobra/amazon/<split>/rqvae_paper_attributes/checkpoint_best_health.pt` | Sentence-T5-XL item embeddings | 独立字段缓存；下游有 ≥80% raw layer usage 门槛 |

Amazon 2023 的 LCRec tokenizer 入口是：

```bash
python -m genrec.trainers.rqvae_trainer \
  config/lcrec/amazon2023/rqvae.gin --split books
```

### RQ-KMeans

OneRec 当前 Amazon 配置使用 3 层、K=256 的残差 MiniBatchKMeans：

```bash
python -m genrec.trainers.rqkmeans_trainer \
  config/onerec/amazon/rqkmeans.gin --split beauty
```

| 输出 | 前置依赖 | 状态 |
|---|---|---|
| `out/onerec/amazon/<split>/rqkmeans/rqkmeans.pt` | Qwen3-Embedding-8B item embeddings、scikit-learn | 前置阶段；CPU 拟合，不是推荐模型 |

### RPG OPQ 与 GRAM lexical ID

| 表示 | 入口 | 前置依赖 | 状态 |
|---|---|---|---|
| RPG OPQ | 无独立 trainer；首次构造 `AmazonRPGDataset` 时在数据适配器内拟合/缓存 | Sentence-T5-XL、FAISS（`pip install -e ".[rpg]"`） | 前置阶段；缓存与 `n_codebook`、PCA、协议等参数相关 |
| GRAM 层次 lexical ID | 仓库无生成脚本 | `dataset/gram/<Split>/` 下的官方风格 ID、序列、文本和相似物品文件 | 外部资产；必须另验 ID 唯一性 |

### 码本健康与碰撞

```bash
python scripts/verify_codebook_health.py \
  --kind rqkmeans \
  --checkpoint out/onerec/amazon/beauty/rqkmeans/rqkmeans.pt \
  --dataset-folder dataset/amazon \
  --split beauty \
  --encoder models_hub/Qwen3-Embedding-8B

python scripts/dump_sem_ids.py \
  --kind rqkmeans \
  --checkpoint out/onerec/amazon/beauty/rqkmeans/rqkmeans.pt \
  --parquet dataset/amazon/processed/beauty/item_emb_Qwen3-Embedding-8B.parquet \
  --out out/_sid_dumps/onerec_beauty.npy

python scripts/report_sid_collisions.py --markdown \
  --sem-ids 'OneRec-K256=out/_sid_dumps/onerec_beauty.npy'
```

三个脚本都是诊断入口；`dump_sem_ids.py` 要求已经存在与 tokenizer 对齐的 parquet，`report_sid_collisions.py` 只读 SID 数组。

## Generative recommenders

### TIGER、LCRec 与 COBRA

```bash
# 先完成各自匹配的 RQ-VAE
python -m genrec.trainers.tiger_trainer \
  config/tiger/amazon/tiger.gin --split beauty

python -m genrec.trainers.lcrec_trainer \
  config/lcrec/amazon/lcrec.gin --split beauty

python -m genrec.trainers.cobra_trainer \
  config/cobra/amazon/cobra.gin --split beauty
```

| 模型 | 配置/变体 | 额外前置依赖 | 状态 |
|---|---|---|---|
| TIGER | `config/tiger/amazon/tiger.gin` | 匹配的 RQ-VAE `checkpoint_epoch_4999.pt` | 主路径；单进程 trainer，当前实现用仓库内纯 PyTorch T5，普通 beam 后做 item 解析 |
| LCRec | `config/lcrec/amazon/lcrec.gin` | RQ-VAE epoch 1999、Qwen3-1.7B | 主路径；历史 checkpoint headline 待按 item-level 重评 |
| COBRA | `cobra.gin`、`cobra_v2.gin`、`cobra_sparse_only.gin` | K=32 RQ-VAE；baseline 的 pretrained dense encoder 还需 Sentence-T5-base | 主路径与消融入口均存在；已实现但未复现论文结果 |

Amazon 2023 LCRec 使用同目录的成对配置：

```bash
python -m genrec.trainers.lcrec_trainer \
  config/lcrec/amazon2023/lcrec.gin --split books
```

### OneRec

当前完整顺序是 RQ-KMeans → SFT 或 CLM；GRPO 只能从匹配的 SFT checkpoint 启动：

```bash
# instruction SFT
python -m genrec.trainers.onerec_sft_trainer \
  config/onerec/amazon/sft.gin --split beauty

# OpenOneRec 风格的替代 CLM；不是 SFT 之后的 stage
python -m genrec.trainers.onerec_clm_trainer \
  config/onerec/amazon/clm.gin --split beauty

# GRPO 前先做完整依赖体检
python scripts/check_grpo_readiness.py \
  config/onerec/amazon/grpo.gin --split beauty

python -m genrec.trainers.onerec_grpo_trainer \
  config/onerec/amazon/grpo.gin --split beauty
```

| 入口 | 前置依赖 | 默认输出/读取 | 状态 |
|---|---|---|---|
| `onerec_sft_trainer` | RQ-KMeans、Qwen3-1.7B | 写 `.../sft/checkpoint_best` | 主路径 |
| `onerec_clm_trainer` | 同一 RQ-KMeans、Qwen3-1.7B | valid Recall@10 严格改善时写 `.../clm/checkpoint_best` | 实验性替代训练目标；全零 smoke 不保证产生 best 目录 |
| `onerec_grpo_trainer` | `.../sft/checkpoint_best`、RQ-KMeans、有效 PrefixTrie | 读 SFT，写 `.../grpo/` | 主路径 RL stage；不能脱离对应 SFT 报增益 |

多卡可使用：

```bash
accelerate launch --config_file config/accelerate_4gpu.yaml \
  -m genrec.trainers.onerec_sft_trainer \
  config/onerec/amazon/sft.gin --split beauty
```

`config/accelerate_4gpu.yaml` 固定为物理卡 `3,4,5,6` 和 4 processes；只有分配到这四张卡时才能原样复制，否则应制作/传入与实际资源一致的 Accelerate 配置。

### RPG

Beauty、Sports、Toys 的配置不同，不应让 Sports/Toys 默用 Beauty 配方：

```bash
python -m genrec.trainers.rpg_trainer \
  config/rpg/amazon.gin --split beauty

python -m genrec.trainers.rpg_trainer \
  config/rpg/amazon_sports.gin --split sports

python -m genrec.trainers.rpg_trainer \
  config/rpg/amazon_toys.gin --split toys
```

状态为主路径。直接全目录 item-level 结果是 headline；`eval_with_graph=True` 的图传播结果是附加近似指标。`scripts/run_matrix.py` 已按 split 路由这三份配置。

### NoteLLM

```bash
python -m genrec.trainers.notellm_trainer \
  config/notellm/amazon.gin --split beauty
```

前置依赖是 Qwen3-1.7B，LoRA 分支还需要 PEFT。当前 trainer 实现对称 GCL、category CSFT、同物品伪负例屏蔽和 exact full-catalog I2I retrieval；它不是用户历史序列生成器，状态为实验性完整链路。

config 同时绑定 trainer 与 dataset 的 `{split}`，因此标准 `--split sports` 会一致更新数据、输出目录与结果 provenance。

真实 backbone 的受限 smoke 入口为：

```bash
python scripts/smoke_notellm.py \
  --pretrained-path models_hub/Qwen3-1.7B \
  --dataset-root dataset/amazon \
  --split beauty
```

### GRAM

```bash
python -m genrec.trainers.gram_trainer \
  config/gram/amazon/gram.gin --split Beauty
```

前置目录 `dataset/gram/Beauty/` 必须已有 `user_sequence.txt`、`item_plain_text.txt`、相似物品文件和配置指定的层次 ID 文件；仓库不会从 Amazon raw 生成它们。状态为实验性完整链路，未注册进 `run_matrix.py`。当前 GRAM `results.json` 未写 `extra.evaluation_level`，因此会被严格矩阵验证器拒绝，修复结果契约前不能进入正式聚合。

## Evaluation 与诊断

正式 test 指标默认由 trainer 的训练闭环产生；评测过程中使用的 checkpoint、best-epoch 选择和最终 `results.json` 才是一套完整记录。以下 eval-only selector 确实存在，但主要用于 checkpoint 检查，不会替代正式结果 provenance：

```bash
# LCRec：checkpoint_path 是 save_pretrained 目录
python -m genrec.trainers.lcrec_trainer \
  config/lcrec/amazon/lcrec.gin --split beauty \
  --gin 'train.eval_only=True' \
  --gin 'train.checkpoint_path="out/lcrec/amazon/beauty/checkpoint_epoch_9"'

# OneRec SFT：当前 eval-only 分支评 valid，不写 results.json
python -m genrec.trainers.onerec_sft_trainer \
  config/onerec/amazon/sft.gin --split beauty \
  --gin 'train_sft.eval_only=True' \
  --gin 'train_sft.checkpoint_path="out/onerec/amazon/beauty/sft/checkpoint_best"'

# OneRec CLM：同样消费 save_pretrained 目录
python -m genrec.trainers.onerec_clm_trainer \
  config/onerec/amazon/clm.gin --split beauty \
  --gin 'train_clm.eval_only=True' \
  --gin 'train_clm.checkpoint_path="out/onerec/amazon/beauty/clm/checkpoint_best"'

# GRAM：checkpoint 是 .pt state_dict；eval-only 评 valid 与 test
python -m genrec.trainers.gram_trainer \
  config/gram/amazon/gram.gin --split Beauty \
  --gin 'train_gram.eval_only=True' \
  --gin 'train_gram.checkpoint_path="out/gram/Beauty/checkpoint_best.pt"'
```

评测相关的轻量检查：

```bash
# 纯 NumPy，自校验 SID 碰撞公式
python scripts/report_sid_collisions.py --selftest

# 数据协议和“诚实指标”测试
pytest -q \
  tests/test_evaluation_protocol.py \
  tests/test_honest_metrics.py \
  tests/test_sid_collisions.py \
  tests/test_notellm_full_catalog.py
```

`scripts/eval_cobra_itemlevel.py` 含固定的 `/root/...` 路径且与当前 trainer 的 item-level 说明已不同；`scripts/eval_rpg_graph.py` 的 `eval_rpg_graph.*` 参数不会自动继承 `train_rpg.*` Gin 绑定。这两个脚本当前只算历史诊断入口，不应原样用于 benchmark。

## Data

### Amazon Reviews 2014

并发环境优先使用原子下载与校验脚本：

```bash
python scripts/prepare_amazon2014.py \
  --root dataset/amazon \
  --splits beauty sports toys
```

它下载 reviews 与 metadata、完整读取 gzip 以验证 CRC，并写 SHA-256 manifest。脚本只枚举 Beauty/Sports/Toys；通用 dataset 对 Clothing/Home 仍有自动下载逻辑，但没有这条脚本的 manifest 流程。

预先生成与校验 item embedding 缓存：

```bash
python scripts/prepare_item_embeddings.py \
  --root dataset/amazon \
  --splits beauty sports toys \
  --encoder models_hub/sentence-t5-xl
```

该命令通过 `AmazonItemDataset` 创建 parquet 并为每个缓存写 manifest。encoder 必须与下游 tokenizer 配置一致；OneRec 应改用 Qwen3-Embedding-8B 路径。

### Amazon Reviews 2023

```bash
# 查看可下载 category
python -m genrec.data.amazon2023 --list_categories

# 下载、迭代 k-core，并转为通用 Amazon adapter 所需的 gzip 文件
python -m genrec.data.amazon2023 \
  --category Books \
  --k_core 5 \
  --output_dir dataset/amazon2023 \
  --source huggingface
```

输出位于 `dataset/amazon2023/raw/books/`，可供当前 `AmazonItemDataset`、SASRec、HSTU 和 LCRec 适配器按 `split=books` 读取。注意同一模块中较早的 `Amazon2023ItemDataset`/`Amazon2023SeqDataset` 类查找的是 `processed/<split>/`；当前提供的配置没有选用这两个类，不能把它们与 CLI 输出直接混用。

### GRAM

GRAM 没有仓库内 raw→训练文件转换命令。复制外部预处理产物前，应记录来源和 checksum，并核对 `config/gram/amazon/gram.gin` 的 `item_id_file`、`cf_model` 与目录大小写。

## Matrix 与结果

### 规划和执行

先 dry-run，检查每个 cell 的 trainer、split-specific config、save dir 和 Gin 覆盖：

```bash
python scripts/run_matrix.py \
  --models sasrec hstu \
  --splits beauty sports toys \
  --seeds 42 43 44 \
  --evaluation-protocol leave_one_out \
  --save-dir-root 'out/matrix/{protocol}/{model}/{split}/seed{seed}' \
  --dry-run
```

去掉 `--dry-run` 才会串行启动。当前 registry 是 `sasrec`、`hstu`、`tiger`、`lcrec`、`cobra`、`rpg`、`notellm`、`onerec_sft`、`onerec_grpo`、`onerec_clm`；不包含 tokenizer 和 GRAM。生成式模型的上游 tokenizer 不由 matrix 自动训练。`onerec_grpo` 会要求同模板下对应 SFT cell 的 `checkpoint_best`。

### 验证、provenance 与聚合

```bash
python scripts/validate_matrix_results.py out/matrix \
  --models sasrec hstu \
  --splits beauty sports toys \
  --protocols leave_one_out global_temporal \
  --seeds 42 43 44

python scripts/stamp_results_provenance.py out/matrix \
  --bundle-name "$BUNDLE_NAME" \
  --sha256 "$BUNDLE_SHA256" \
  --git-sha "$GIT_SHA"

python scripts/aggregate_results.py out/matrix \
  --metrics Recall@5 Recall@10 NDCG@10 \
  --format md \
  --output /tmp/genrec-matrix.md
```

其中三个环境变量必须取自真实上传记录，不能手填猜测。顺序必须是“确认真实 source bundle → 必要时 stamp → strict validate → aggregate”。聚合器会按 `(model, split, protocol, evaluation_level)` 分组，但本身比验证器宽松；不能只因聚合命令成功就认定结果合规。stamp 只适用于来源已经确定的旧产物。

## Remote 179

权威流程见[179 复现运行手册](../179-reproduction-runbook.md)。这里保留入口，不记录会变化的 GPU 空闲状态。

### 上传与应用

```bash
# 本地只打包，检查归档名与 SHA
./upload.sh --dry-run

# 本地上传源码归档与 apply helper
./upload.sh
```

上传不会直接改远端代码；使用输出中带精确 archive 和 SHA-256 的 apply 命令。远端 apply helper 会取得独占源码锁、拒绝训练期间替换，并记录最后应用的 bundle。它使用 `rsync -a` 而没有 `--delete`，所以本地已删除的旧文件可能仍残留远端；涉及删除/重命名时必须额外核对远端树。

### 锁定任务

每次提交前都要同时检查 179 的 running task 列表和 GPU 进程；仅看空闲显存不够。MCP 任务声明的物理卡必须与 wrapper 的 `--gpus` 完全相同：

```bash
scripts/run-locked-179.sh \
  --gpus 2 \
  --cell sasrec.beauty.loo.s42.train \
  -- venv/bin/python -m genrec.trainers.sasrec_trainer \
  config/sasrec/amazon.gin --split beauty \
  --gin 'train.seed=42' \
  --gin 'train.save_dir_root="out/matrix/leave_one_out/sasrec/beauty/seed42"'
```

CPU-only 数据 cell 也要序列化：

```bash
scripts/run-locked-179.sh \
  --gpus none \
  --cell data.amazon2014.raw.prepare \
  -- venv/bin/python scripts/prepare_amazon2014.py \
  --splits beauty sports toys
```

任务名使用 `genrec:<model>:<split>:<protocol>:s<seed>:<purpose>`。不要杀死未知进程、复用其他会话的 cell，或绕过 wrapper 直接共享 GPU。

## 提交前核验

入口或配置变化时，本页也必须更新。最低检查：

```bash
# CLI/配置目标存在
test -f config/sasrec/amazon.gin
python scripts/run_matrix.py \
  --models sasrec --splits beauty --seeds 42 --dry-run

# 结果协议脚本的纯测试
pytest -q \
  tests/test_run_matrix_protocol.py \
  tests/test_validate_matrix_results.py \
  tests/test_aggregate_protocol.py \
  tests/test_stamp_results_provenance.py

# 文档链接与导航
mkdocs build --strict
```

命令“存在”不等于训练已经在当前机器完成；耗时路径的状态应回到[模型状态表](model-status.md)和对应模型页记录，不在本索引里写滚动中的任务进度。
