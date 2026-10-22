# GenRec

[English](README.md) | [简体中文](README.zh-CN.md)

面向生成式推荐的 PyTorch 研究框架，使用 Gin 管理实验配置。项目提供模型实现、数据处理、语义 ID、训练和评测链路，便于做可追溯的复现实验与方法比较。

## 复现结果

当前已完成 Amazon Reviews 2014 的 **Beauty、Sports、Toys** 主矩阵：**8 种推荐器、9 条训练流水线 × 3 个数据集 × 2 种切分协议 × 3 个种子，共 162 个实验、54 条三种子汇总行**。OneRec 的 SFT 与 GRPO 分别算一条流水线。

完整指标、均值 ± 总体标准差、配方边界与来源见 [BENCHMARKS.md](BENCHMARKS.md)；[逐种子 CSV](results/matrix_20261022.csv) 保留了原始结果路径和 SHA256，[汇总 JSON](results/matrix_20261022.summary.json) 可供程序读取。

使用这些数字时，需要保留以下边界：

- 主指标都是 **item-level** Recall/NDCG，评测覆盖完整商品目录，不使用抽样负例。SID 模型先按训练集流行度解析到确定的 item，再去重；不会把同一 SID 下的全部商品都算作命中。自回归模型的 top-K 由 beam search 近似获得。
- 下表为 **leave-one-out 测试集**上 seed 42/43/44 的均值，按 validation 选择模型；全局时间切分和标准差见 BENCHMARKS。历史长度通常为 50，LCRec 的论文适配配方为 20，因此并非严格等历史长度对照。
- `global_temporal` 使用全局 80%/90% 时间边界；商品目录、元数据过滤和部分 SID 构建仍使用全量信息，不能称为完全无泄漏的归纳式评测。
- OneRec 主结果使用 **3×256 码本**，四码 Beauty 消融单独报告。GRPO 把初始 SFT 也纳入 validation 选择，**18 个实验中有 16 个回选 SFT**。OneRec/LCRec 的评测模板仍随机采样，细小差异不能当作隔离的 RL 收益。
- NoteLLM 是 GCL+CSFT 的 **I2I 适配**：取历史末项作为查询，并排除查询商品自身。它的查询和候选语义与序列推荐不同。
- 本次矩阵验证的是仓库中的适配实现。TIGER/COBRA 的独立论文对齐实验与历史 SID 指标分开保留，不代表所有原论文效果都已复现。

### Amazon 2014 Beauty

| 方法 | R@5 | R@10 | N@5 | N@10 |
| --- | --- | --- | --- | --- |
| SASRec (CE) | 0.0542 | 0.0858 | 0.0317 | 0.0420 |
| HSTU (CE) | 0.0461 | 0.0727 | 0.0306 | 0.0392 |
| TIGER (matrix recipe) | 0.0361 | 0.0568 | 0.0234 | 0.0301 |
| LCRec (4B, seq20, gb128) | 0.0509 | 0.0729 | 0.0360 | 0.0431 |
| COBRA (optimized light) | 0.0290 | 0.0454 | 0.0187 | 0.0240 |
| RPG (direct) | 0.0506 | 0.0732 | 0.0354 | 0.0427 |
| NoteLLM (GCL+CSFT, I2I) | 0.0502 | 0.0693 | 0.0362 | 0.0423 |
| OneRec SFT (3 SID) | 0.0433 | 0.0658 | 0.0291 | 0.0364 |
| OneRec + GRPO (selected) | 0.0437 | 0.0661 | 0.0293 | 0.0365 |

### Amazon 2014 Sports

| 方法 | R@5 | R@10 | N@5 | N@10 |
| --- | --- | --- | --- | --- |
| SASRec (CE) | 0.0313 | 0.0497 | 0.0182 | 0.0241 |
| HSTU (CE) | 0.0234 | 0.0384 | 0.0152 | 0.0200 |
| TIGER (matrix recipe) | 0.0207 | 0.0342 | 0.0131 | 0.0174 |
| LCRec (4B, seq20, gb128) | 0.0260 | 0.0399 | 0.0178 | 0.0223 |
| COBRA (optimized light) | 0.0179 | 0.0290 | 0.0117 | 0.0153 |
| RPG (direct) | 0.0290 | 0.0426 | 0.0200 | 0.0244 |
| NoteLLM (GCL+CSFT, I2I) | 0.0237 | 0.0351 | 0.0161 | 0.0197 |
| OneRec SFT (3 SID) | 0.0222 | 0.0341 | 0.0142 | 0.0180 |
| OneRec + GRPO (selected) | 0.0220 | 0.0341 | 0.0141 | 0.0180 |

### Amazon 2014 Toys

| 方法 | R@5 | R@10 | N@5 | N@10 |
| --- | --- | --- | --- | --- |
| SASRec (CE) | 0.0610 | 0.0917 | 0.0342 | 0.0441 |
| HSTU (CE) | 0.0438 | 0.0686 | 0.0287 | 0.0366 |
| TIGER (matrix recipe) | 0.0288 | 0.0477 | 0.0181 | 0.0242 |
| LCRec (4B, seq20, gb128) | 0.0504 | 0.0705 | 0.0356 | 0.0420 |
| COBRA (optimized light) | 0.0252 | 0.0412 | 0.0151 | 0.0202 |
| RPG (direct) | 0.0569 | 0.0820 | 0.0391 | 0.0472 |
| NoteLLM (GCL+CSFT, I2I) | 0.0674 | 0.0917 | 0.0460 | 0.0539 |
| OneRec SFT (3 SID) | 0.0386 | 0.0591 | 0.0256 | 0.0321 |
| OneRec + GRPO (selected) | 0.0393 | 0.0604 | 0.0255 | 0.0323 |

Beauty 的补充实验：修正后的 HSTU Meta-local sampled softmax，R@10 = 0.0754 ± 0.0021；OneRec 四码后缀消融，R@10 = 0.0727 ± 0.0013。两者均为独立的三种子结果，不替换主矩阵行。

### Amazon 2014 Home

Home 的完整复现正在准备，目标沿用主矩阵的 9 条流水线、两种切分协议和三个种子，共 54 个实验，使用独立的 [Home 配方清单](config/matrix/amazon2014_home_20261022.json)。RPG 直接迁移 Beauty 配方，不以 Home test 调参。英文 README 中的 Home 表仍是历史快照，尚未纳入这次核验的 162 个实验；在新结果完成前，不用旧 SID 数字补表。

### 当前结论

SASRec 在 LOO 的 R@10 上仍是强基线，生成模型没有在各数据集上全面胜出。RPG 在 Toys 上有竞争力；NoteLLM 的 I2I 结果需结合任务差异理解。所有模型在全局时间切分下的 R@10 都低于对应 LOO 结果，结论依赖数据协议。GRPO 尚未显示稳定的隔离收益；OneRec 四码在 Beauty 的改善也不能直接推广到其他域。本次没有做显著性结论。

## 模型与组件

仓库共有 **9 个推荐模型和 2 个 tokenizer 组件**；GRAM 未进入上述主矩阵。

| 模型 / 组件 | 路线 | 当前实现 |
| --- | --- | --- |
| SASRec | 判别式序列推荐 | softmax attention，单 item ID，全目录排序 |
| HSTU | 判别式序列推荐 | SiLU attention、U gate、时间偏置；CE / sampled softmax |
| TIGER | 自回归生成推荐 | T5 + RQ semantic ID，beam search |
| LCRec | LLM 生成推荐 | 协同语义 ID + 指令微调；主矩阵为 Qwen3-4B / 四码 USM |
| COBRA | 稀疏与稠密结合 | 分层 SID + dense refinement / BeamFusion |
| RPG | 并行生成 | GPT-2 + 多个并行 head + OPQ；主表使用 direct item score |
| NoteLLM | LLM 检索 | GCL+CSFT、LoRA，Amazon I2I 适配 |
| OneRec | LLM 生成推荐 | RQ-KMeans → SFT → GRPO；另有 CLM 路径 |
| GRAM | 词法 ID 生成推荐 | 多粒度融合；尚未进入主矩阵 |
| RQ-VAE / RQ-KMeans | tokenizer | 把商品 embedding 转成离散 SID，本身不承担推荐任务 |

## 安装

Python 3.9+，依赖与可选组件定义在 [pyproject.toml](pyproject.toml)。

```bash
git clone https://github.com/phonism/genrec.git
cd genrec
pip install -e .

# 复现完整矩阵所需的 RPG / LoRA 依赖
pip install -e ".[rpg]" "peft>=0.10.0"

# 如需 Triton 等其他可选组件
pip install -e ".[full]"
```

LLM 与文本编码路径默认从仓库根目录的 `models_hub/` 读取，见 [config/base.gin](config/base.gin)。按所用流水线准备 `sentence-t5-xl`、`Qwen3-Embedding-8B`、`Qwen3-1.7B` 和 `Qwen3-4B`。大模型完整参数训练需相应显存；LCRec 主配方是在 80GB GPU 上验证的。

## 复现已发布的矩阵

请从仓库根目录运行。对应主表的入口是 **`--preset amazon2014_20261022`**，配方映射在 [配置清单](config/matrix/amazon2014_20261022.json) 中。仓库保留的通用配置与历史调参脚本包含其他配方，不能直接与这次主表互换。

每个数据域先准备一个 **seed 42 的 tokenizer**，供推荐器的 42/43/44 三个种子共同使用，以隔离推荐器训练的随机性：

```bash
# 使用已提交的干净源码；为新实验记录源码 archive 与校验和。
mkdir -p out
GENREC_REPRO_REV=$(git rev-parse HEAD)
export GENREC_SOURCE_BUNDLE_NAME="genrec-source-${GENREC_REPRO_REV}.tar.gz"
git archive --format=tar.gz --output="out/${GENREC_SOURCE_BUNDLE_NAME}" HEAD
export GENREC_SOURCE_BUNDLE_SHA256="$(python -c 'import hashlib,os,pathlib; print(hashlib.sha256(pathlib.Path("out", os.environ["GENREC_SOURCE_BUNDLE_NAME"]).read_bytes()).hexdigest())')"

for split in beauty sports toys; do
  python -m genrec.trainers.rqvae_trainer config/tiger/amazon/rqvae.gin \
    --split "$split" --gin 'train.seed=42' --gin 'train.wandb_logging=False'
  python -m genrec.trainers.rqvae_trainer config/lcrec/amazon/rqvae_paper_recipe.gin \
    --split "$split" --gin 'train.seed=42'
  python -m genrec.trainers.rqvae_trainer config/cobra/amazon/rqvae.gin \
    --split "$split" --gin 'train.seed=42' --gin 'train.wandb_logging=False'
  python -m genrec.trainers.rqkmeans_trainer config/onerec/amazon/rqkmeans.gin \
    --split "$split" --gin 'fit_rqkmeans.seed=42'
done

# 每种协议各 81 个实验。先审阅命令，去掉 --dry-run 后才会串行训练。
for protocol in leave_one_out global_temporal; do
  python scripts/run_matrix.py --preset amazon2014_20261022 \
    --models sasrec hstu tiger lcrec cobra rpg notellm onerec_sft onerec_grpo \
    --splits beauty sports toys --seeds 42 43 44 \
    --evaluation-protocol "$protocol" \
    --save-dir-root 'out/reproduce_20261022/{protocol}/{model}/{split}/seed{seed}' \
    --dry-run
done
```

runner 会固定 LCRec 的 4B/四码/seq20 配方、COBRA optimized-light 和 RPG 各域配置，关闭 W&B 登录，分开协议/种子输出，并让 GRPO 加载同种子、同协议的 SFT `checkpoint_best`。每个实验按单卡串行执行；共享实验机需遵守项目的 GPU 锁与调度规则。

| 流水线 | 前置产物 |
| --- | --- |
| TIGER | `out/tiger/amazon/{split}/rqvae/checkpoint_epoch_4999.pt` |
| LCRec | `out/lcrec/amazon/{split}/rqvae_paper_recipe/checkpoint_best_health.pt` |
| COBRA | `out/cobra/amazon/{split}/rqvae/checkpoint_epoch_4999.pt` |
| OneRec SFT / GRPO | `out/onerec/amazon/{split}/rqkmeans/rqkmeans.pt`；GRPO 另需匹配的 SFT checkpoint |
| RPG | dataset adapter 内生成 OPQ 缓存 |

这些命令用于复现配方，不保证不同硬件或重新训练 tokenizer 后的数字逐位相同。模型权重、embedding 缓存、tokenizer checkpoint 和原始结果文件未随 Git 发布；CSV 保留原始结果路径与校验和。历史源码版本混用和随机评测模板等限制见 BENCHMARKS。

### 检查数字一致性

无需导入 PyTorch 即可核对中英文 README、BENCHMARKS、双语文档首页与 CSV/汇总 JSON：

```bash
python scripts/check_benchmarks.py

# 如已恢复 primary/ 与 ablations/ 原始结果目录，可进一步核对 JSON 与 SHA256。
python scripts/check_benchmarks.py --artifacts out/matrix_20261022
```

### 单模型起步

```bash
# 无外部 tokenizer 依赖的基线
python -m genrec.trainers.sasrec_trainer config/sasrec/amazon.gin \
  --split beauty --gin 'train.wandb_logging=False'

# HSTU；sampled softmax 用独立的 amazon_ss.gin，不能与 CE 混为同一配方。
python -m genrec.trainers.hstu_trainer config/hstu/amazon.gin \
  --split beauty --gin 'train.wandb_logging=False'

# 覆盖超参数或本地模型路径
python -m genrec.trainers.lcrec_trainer config/lcrec/amazon/lcrec_paper_recipe.gin \
  --split beauty --gin "MODEL_HUB_QWEN3_4B='/path/to/Qwen3-4B'"
```

所有 Gin trainer 都接受配置文件位置参数、可选的 `--split` 和可重复的 `--gin`。Amazon 2014 可选 `beauty/sports/toys/clothing/home`；Amazon 2023 使用各自目录下的专用配置。多阶段模型需先准备上表中的匹配产物。

## 项目结构

```text
genrec/
├── genrec/
│   ├── models/        # 推荐模型与量化器
│   ├── trainers/      # 各模型及 OneRec SFT/GRPO/CLM 训练器
│   ├── data/          # Amazon、MovieLens 与模型专用 adapter
│   └── modules/       # attention、encoder、loss、metrics、PrefixTrie 等
├── config/            # Gin 配置与矩阵配方清单
├── scripts/           # 数据准备、训练启动、诊断与结果核验
├── tests/             # 回归检查
├── results/           # 已发布的逐种子指标与汇总
└── docs/              # 中英文 MkDocs 文档
```

`dataset/`、`models_hub/`、`out/` 为本地产物目录，不进入 Git。各流水线的数据 adapter 与模型直接协作，公共训练与评测工具负责统一协议和结果契约。

## 文档与贡献

[中文文档](https://phonism.github.io/genrec/)包含[快速开始](docs/zh/getting-started.md)、[模型总览](docs/zh/models/overview.md)、[命令索引](docs/zh/reference/command-index.md)和[实现与复现状态](docs/zh/reference/model-status.md)。提交改动前请参考[贡献指南](docs/zh/contributing.md)。

## 引用

如果本项目对研究有帮助，可以引用仓库：

```bibtex
@software{genrec2025,
  title = {GenRec: A Model Zoo for Generative Recommendation},
  author = {Qi Lu},
  year = {2025},
  url = {https://github.com/phonism/genrec}
}
```

各模型的原始论文与官方代码入口见[英文 README](README.md#references)和模型文档。项目采用 [MIT License](LICENSE)。
