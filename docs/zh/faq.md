# 常见问题

## 安装与环境

### 支持哪些 Python / PyTorch 版本？

`pyproject.toml` 声明 Python `>=3.9`、PyTorch `>=2.0`。具体 CUDA wheel 由本机驱动决定，不要从文档复制固定 CUDA 版本。主要实验环境是 Linux + NVIDIA GPU。

### 为什么 `config/base.gin` 指向 `/root/workspace/models_hub`？

这是 179 复现实例的本地镜像路径。其他机器用 `--gin 'MODEL_HUB_...="..."'` 覆盖。不要把训练直接指向慢速 AFS。

### 为什么 NoteLLM 报 `peft` 缺失？

默认 config 开启 LoRA，而 PEFT 属于 `full` 可选依赖。安装 `pip install -e ".[full]"`，或明确关闭 LoRA；后者是不同实验配置。

### 为什么 RPG 报 FAISS 缺失？

安装 `pip install -e ".[rpg]"`。当前 optional group 固定 `faiss-cpu==1.10.0` 以匹配复现环境。

## 数据

### 数据会自动下载吗？

不统一。`AmazonItemDataset` 能按需下载 raw review/meta；SASRec/HSTU adapter 直接读取文件。共享实验机推荐显式运行 `scripts/prepare_amazon2014.py`，获得锁、CRC 与 SHA-256 manifest。

### Amazon 2023 为什么仍读取 `reviews_*_5.json.gz`？

`genrec.data.amazon2023` 先下载 2023 JSONL、做迭代 k-core，再输出兼容旧 adapter 的 gzip 文件。必须先运行准备命令；只换 `dataset_folder` 不会自动完成转换。

### 可以跨模型复用整数 item ID 吗？

不可以默认复用。SASRec/HSTU/RPG 是 1-based + 0 padding；TIGER/LCRec/OneRec/COBRA 多为 0-based；GRAM 使用字符串 ID。mapping 还依赖原始记录顺序。

### `leave_one_out` 与 `global_temporal` 哪个更好？

它们回答不同问题。LOO 便于复现传统工作；global temporal 更能抑制未来交互进入训练历史。主表应并列报告或明确选择，绝不能跨协议平均。

## 配置

### `--split` 做了什么？

`parse_config` 先把 Gin 文件文本中的 `{split}` 替换成 CLI 值，再解析多条 `--gin` override，并把 split 放进进程上下文供 `results.json` 使用。

### 为什么 `--gin 'train.xxx=...'` 不生效？

检查可配置函数名。OneRec 是 `train_sft/train_clm/train_grpo`，RPG 是 `train_rpg`，GRAM 是 `train_gram`。以 config 左侧 binding 为准。

### Python 默认值和 Gin 不一样，哪个算数？

执行时 Gin 覆盖 Python 默认；复现实验以实际 config、CLI overrides 和 `results.json` 为准。文档中的示例值不是 provenance。

## 模型

### RQ-VAE、RQ-KMeans 是推荐模型吗？

不是。它们把 item embedding 转为 SID，是下游生成式推荐器的 tokenizer。其指标是重构、码本健康和 collision，不单独报推荐 Recall。

### 为什么 K=8192 会塌缩？

在一万级 item 的目录中，后层 residual 很浅，超大 MiniBatchKMeans 难以稳定覆盖大量 cluster，可能出现 dead clusters 与 super-cluster。本项目小目录基线使用 K=256，并有 dead-cluster reinit。不要把每层 usage 与完整 SID 唯一性混同。

### TIGER 是否使用 Trie？

当前代码没有。`Tiger.generate()` 是普通 beam search，非法/目录外 SID 在 item mapper 中被丢弃。旧概览曾写 Trie-constrained，是文档与实现不一致，现已在模型页明确。

### OneRec 的 SFT、CLM、GRPO 是什么关系？

RQ-KMeans 是共同前置；SFT 与 CLM 是两种训练数据组织方式，CLM 不是 SFT 后必跑阶段；GRPO 依赖 SFT checkpoint。

### GRPO 为什么很多 step 没梯度？

SFT top-1 命中率低、group size 有限时，一个 group 没有任何正样本很常见。`if not has_match: return zeros` 是 hard-sample skip，不应当作 reward bug 删除。可比较 `add_gt`、更大 group 或 dense reward，但要保留隔离对照。

### COBRA 已实现为何结果低？

“有代码”与“复现论文结果”不同。当前 item-level 结果显著低于论文；已排除多类 tokenizer/解码/信息流问题，详细诊断保留在 [COBRA](models/cobra.md)。

### NoteLLM 是生成 SID 模型吗？

不是当前实现。它抽取 `[EMB]` 前一个 token 的 LLM hidden state，做 dense item retrieval。in-batch recall 只能诊断，正式 trainer 使用 exact full-catalog item retrieval。

## 指标

### 为什么 SID Recall 比 item Recall 高？

多个 item 可能共享完整 SID。SID tuple 命中会把同桶任意 item 当成目标，产生结构性膨胀。headline 必须使用 training-only resolver 解析到唯一 item，并去重候选。

### 为什么 item dedup 后 Recall 可能反而变化？

多个 beam 可解析到同一 item。去重会从更深 beam 拉入新 item，所以 item@K 不是简单对 SID@K 乘一个固定系数；两者数值不可直接比较。

### NoteLLM 的 in-batch Recall 能和 SASRec 比吗？

不能。候选只来自 batch，难度远低于完整目录。只有 `compute_full_catalog_metrics` 路径可进入同口径主表。

### Recall@K 与 HR@K 是否相同？

当前 next-item 协议每个样本一个 target，所以数值相同；文档统一写 Recall，以便扩展到多目标时保持语义。

## 训练与结果

### 最佳 checkpoint 如何选择？

多数 recommender 用 valid Recall@10；具体 trainer 可能不同。生成式模型必须确认选择依据已是 item-level，而非旧 SID-level。

### 为什么 README 数字和本地 `best_model.pt` 对不上？

checkpoint 可能被 smoke/debug run 覆盖，或数字来自不同 seed/protocol/metric level。优先检查 `results.json` 的 exact source provenance 与日志，不根据文件名猜实验。

### 如何跑 3-seed matrix？

使用 `scripts/run_matrix.py` dry-run，再通过 179 locked workflow 提交；完成后用 validator 和 aggregator。不要并行手敲命令占用同一 GPU/cell。

### 可以杀掉陌生 GPU 进程吗？

不可以。先同时检查远程 task list 和 GPU processes；未知进程可能属于其他会话。`run-locked-179.sh` 会在锁后复检并拒绝共享。

## 文档

### 为什么有些英文页比中文旧？

当前中文是实现与领域全景的 canonical 文档；i18n 对缺失英文页使用 fallback。不要据此声称双语内容完全同步。

### 如何报告文档错误？

给出页面、具体陈述、对应代码/论文证据和核验日期。模型页按[文档维护规范](reference/documentation-policy.md)更新。
