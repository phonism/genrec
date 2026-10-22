# 实现与复现状态

这张表回答三个不同的问题：仓库里是否有代码、代码能否组成完整训练/评测链路、论文结果是否已经在当前评测口径下复现。三者不能互相替代。

!!! warning "状态口径"
    “已实现”只表示仓库中存在模型、数据适配器和训练入口，不表示实现与论文逐行等价，也不表示已经复现论文数字。生成式模型的主结果必须是解析到真实 item、去重后的 item-level 指标；SID 元组命中只可作为诊断。

## 状态标签

| 标签 | 含义 |
|---|---|
| **完整链路** | 模型、数据适配器、训练器和配置均存在 |
| **组件** | 为下游模型服务，不应作为独立推荐器比较 |
| **实验性** | 有可运行入口，但覆盖、对齐或复现实验证据仍有限 |
| **已复现** | 在仓库规定协议下有可追溯结果；不等于复现所有论文设置 |
| **待重评** | 历史 checkpoint 的 headline 指标使用旧口径，需要用当前 item-level 代码重跑 |
| **未复现** | 已运行但未达到论文可比结果，或尚无足够证据 |
| **未实现** | 仅在领域目录中收录，仓库没有训练链路 |

## 仓库内模型

| 名称 | 角色 | 代码链路 | 主表示 / 输出 | 当前主评测 | 复现状态 |
|---|---|---|---|---|---|
| [SASRec](../models/sasrec.md) | 判别式序列基线 | 完整链路 | 单 item ID / 全目录 logits | item-level full catalog | 已复现基线 |
| [HSTU](../models/hstu.md) | 判别式序列基线 | 完整链路 | 单 item ID / 全目录 logits | item-level full catalog | 已复现基线；CE 与 sampled-softmax 分开报告 |
| [RQ-VAE](../models/rqvae.md) | SID tokenizer | 组件 | 多层残差量化码 | 重构、码本健康、碰撞率 | 下游依赖，不单列推荐 Recall |
| [RQ-KMeans](../models/rqkmeans.md) | SID tokenizer | 组件 | 多层 residual K-Means 码 | 码本健康、碰撞率 | OneRec 前置阶段 |
| [TIGER](../models/tiger.md) | 自回归生成式检索 | 完整链路 | RQ-VAE SID / T5 beam | item-level SID→item 解析与去重 | 原矩阵配方 18/18 完成；Beauty 论文对齐行单独保留 |
| [LCRec](../models/lcrec.md) | LLM + 协同语义 | 完整链路 | legacy 5 层；矩阵为 4 层 RQ-VAE + USM / Qwen3-4B | item-level SID→item 解析与去重 | Qwen3-4B、有效 batch 128 的 Amazon 适配 18/18 完成；不等于原论文设置 |
| [OneRec](../models/onerec.md) | LLM 生成式推荐 | 完整链路：RQ-KMeans、SFT、CLM、GRPO | 主矩阵 3 层 SID；SFT/GRPO 评测与 RL rollout 使用 catalog Trie | item-level SID→item 解析与去重 | SFT、GRPO 各 18/18 完成；16 格 GRPO 选回 SFT；4 码仅 Beauty 消融 |
| [COBRA](../models/cobra.md) | 稀疏—稠密级联生成 | 完整链路 | RQ-VAE 码 + dense vector | item-level BeamFusion | optimized light 18/18 完成；尚不能宣称复现论文结果 |
| [RPG](../models/rpg.md) | 并行生成 | 完整链路 | OPQ tokens / 并行 heads 后直接 item 打分 | item-level direct full catalog | 18/18 完成；graph metric 仅作 extra，不能覆盖 direct headline |
| [NoteLLM](../models/notellm.md) | LLM 可检索表示 | 实验性完整链路 | `[EMB]` 前一个 token 的向量 / dense retrieval | exact full catalog I2I retrieval | GCL+CSFT、同物品伪负例屏蔽的 Amazon I2I 适配 18/18 完成；非完整论文复现 |
| [GRAM](../models/gram.md) | 多粒度生成式检索 | 实验性完整链路 | FiD 风格多 passage / Trie lexical-ID generation | lexical-ID exact match；唯一 item 语义尚未验证 | 缺上游 lexical-ID 预处理与唯一性检查，复现证据待补 |

上表按“推荐器”计有 9 个（SASRec、HSTU、TIGER、LCRec、OneRec、COBRA、RPG、NoteLLM、GRAM）；RQ-VAE 与 RQ-KMeans 是 tokenizer / quantizer 组件。代码目录因此会看到 11 个模型文件，但不能把它们都当成同一层级的 recommender 排名。

2026-10-22 验收的主矩阵是 **9 条训练/评测管线 × 3 数据集 × 2 协议 × 3 seed = 162 格**：
它把 OneRec SFT 和 GRPO 分开计数，未包含 GRAM 或 OneRec CLM。所有主行均为
item-level；代码版本与配方并非全表一致。OneRec/LCRec 评测 prompt 模板尚未冻结，
特别是 GRPO 回退 SFT 后的微小测试分差不能解释为 RL 收益。完整边界见
[BENCHMARKS.md](https://github.com/phonism/genrec/blob/main/BENCHMARKS.md)。

## 入口矩阵

| 名称 | 模型 | 数据适配器 | 训练器 | 默认配置 |
|---|---|---|---|---|
| SASRec | `genrec/models/sasrec.py` | `amazon_sasrec.py` | `sasrec_trainer.py` | `config/sasrec/amazon.gin` |
| HSTU | `genrec/models/hstu.py` | `amazon_hstu.py`, `ml1m_hstu.py` | `hstu_trainer.py` | `config/hstu/amazon.gin` |
| RQ-VAE | `genrec/models/rqvae.py` | `amazon.py` | `rqvae_trainer.py` | 各下游目录的 `rqvae.gin` |
| RQ-KMeans | `genrec/models/rqkmeans.py` | `amazon.py` | `rqkmeans_trainer.py` | `config/onerec/amazon/rqkmeans.gin` |
| TIGER | `genrec/models/tiger.py` | `amazon.py` | `tiger_trainer.py` | `config/tiger/amazon/tiger.gin` |
| LCRec | `genrec/models/lcrec.py` | `amazon_lcrec.py` | `lcrec_trainer.py` | `config/lcrec/amazon/lcrec.gin` |
| OneRec | `genrec/models/onerec.py` | `amazon_onerec.py`, `amazon_onerec_clm.py` | `onerec_{sft,clm,grpo}_trainer.py` | `config/onerec/amazon/*.gin` |
| COBRA | `genrec/models/cobra.py` | `amazon_cobra.py` | `cobra_trainer.py` | `config/cobra/amazon/cobra.gin` |
| RPG | `genrec/models/rpg.py` | `amazon_rpg.py` | `rpg_trainer.py` | `config/rpg/amazon.gin` |
| NoteLLM | `genrec/models/notellm.py` | `amazon_notellm.py` | `notellm_trainer.py` | `config/notellm/amazon.gin` |
| GRAM | `genrec/models/gram.py` | `amazon_gram.py` | `gram_trainer.py` | `config/gram/amazon/gram.gin` |

## 什么结果可以进入主表

一条结果至少需要同时满足：

1. 明确数据版本、split、过滤方式、`max_seq_len` 和随机种子；
2. 明确 `leave_one_out` 或 `global_temporal`，不同协议不得聚合；
3. 生成式 SID 已用**仅训练交互**构建的映射解析为 item，并对最终 item 排名去重；
4. 指标是 full catalog 或明确标注候选集，不能把 sampled/in-batch 指标伪装为 full catalog；
5. `results.json` 包含源码 provenance、模型、seed、协议、metric level；
6. 多 seed 汇总使用相同配置与协议，报告均值及标准差。

机器可检查的结果约束见 `scripts/validate_matrix_results.py`；当前可追溯矩阵见仓库根目录的 `BENCHMARKS.md`。

## 未实现模型放在哪里

未实现工作统一收录在[领域模型目录](../landscape/model-catalog.md)，并标注论文、官方代码、技术路线和建议优先级。只有在模型、数据、训练、评测和最小测试都落地后，条目才从“未实现”迁入本页。
