# 生成式推荐模型目录

本页是截至 **2026-08-31** 的代表作目录，不声称穷尽所有论文。收录和排除规则、状态与 P0–P3 定义见[总览](index.md)。“官方代码”只有论文作者或所属机构明确发布的仓库；无法确认时写“未发现”，不列第三方复现。

先做名称对齐：`SASRec` 只作判别式基线；`rqvae`、`rqkmeans` 是 SID 组件；`tiger`、`lcrec`、`rpg`、`hstu` 有完整训练/评测路径；`cobra`、`gram`、`notellm` 有实验实现但还需补论文级闭环。仓库 `onerec` 提供 RQ-KMeans + Qwen 的 SFT、GRPO、CLM 路线，是对公开 OneRec 生态的研究实现，不能等号连接到 OneRec V1、V2、MiniOneRec 或 OpenOneRec 任意一篇论文。

## 1. 语义 ID 与生成式检索主线

| 模型 | 年份/出处 | 核心表示 | 生成机制 | 论文 / 官方代码 | 本仓库状态 | 建议 |
|---|---|---|---|---|---|---|
| TIGER | 2023, NeurIPS | Sentence-T5 内容向量经 RQ-VAE 得到层次 SID | T5 式 encoder-decoder 自回归生成下一物品 SID，beam search | [论文](https://arxiv.org/abs/2305.05065)；论文作者未发布官方代码 | ✅ 已实现；纯 PyTorch T5 与 RQ-VAE 路径齐全，仍需按物品级协议重评 | P0：先完成诚实重评 |
| LC-Rec | 2024, ICDE | 语义量化码 + 冲突消解 token，并用协同任务对齐 LLM | 多任务指令微调后自回归生成完整 SID | [论文](https://arxiv.org/abs/2311.09049) / [代码](https://github.com/RUCAIBox/LC-Rec) | ✅ 已实现；旧 SID-tuple 数字不可作最终结论 | P0：与 TIGER 同协议重评 |
| LIGER | 2024, arXiv | RQ-VAE SID 与 dense item embedding 双表示 | 生成候选，再与 dense retrieval 候选联合重排，专门补冷物品 | [论文](https://arxiv.org/abs/2411.18814) / [代码](https://github.com/facebookresearch/liger) | ⬜ 未实现 | P1：最有价值的 sparse–dense 对照 |
| GRAM | 2025, ACL | semantic-to-lexical 层次 ID；多粒度物品文本 | 多编码器分别编码粒度不同的 prompt，在解码端 late fusion 后生成 | [论文](https://aclanthology.org/2025.acl-long.1596/) / [代码](https://github.com/skleee/GRAM) | 🧪 模型、训练器、配置已存在；专用预处理数据与论文级复现未闭环 | P1：先补数据契约和 parity test |
| ActionPiece | 2025, ICML Spotlight | 把 action 的特征集合按上下文做 BPE 式合并，形成可变粒度 token | 对集合排列不敏感的 action-piece 序列生成 | [论文](https://arxiv.org/abs/2502.13581) / [代码](https://github.com/google-deepmind/action_piece) | ⬜ 未实现 | P2：适合研究 token 粒度，而非首个新基线 |
| COBRA | 2025, NeurIPS | 多个 sparse code + 一个连续 dense 尾向量 | 先生成 SID，再条件生成 dense 向量；BeamFusion 融合 beam 与近邻分数 | [论文](https://arxiv.org/abs/2503.02453)；未发现官方代码 | 🧪 已实现并使用物品级评测，尚未完成论文数字复现 | P1：完成消融与复现审计 |
| RPG | 2025, KDD | OPQ 子空间 code 组成论文所称的 unordered long SID；codebook 位置固定，长度按数据集选择（Sports/Toys 16、Beauty 32、CDs 64） | GPT-2 编码历史，各 codebook 的独立 ResBlock head 并行预测 token；可选图约束传播解码 | [论文](https://arxiv.org/abs/2506.05781) / [代码](https://github.com/facebookresearch/RPG_KDD2025) | ✅ 已实现；含图传播评测路径 | P1：补多数据集/多 seed |
| OneRec V1 | 2025, arXiv/工业报告 | item token 与 session/page 序列 | encoder-decoder 生成整页推荐；迭代对齐与 DPO/奖励模型优化 | [论文](https://arxiv.org/abs/2502.18965)；未发现完整官方训练代码 | ⬜ 未按论文架构实现；本仓库同名代码不等价 | P3：数据和工业模块不可得 |
| OneRec V2 | 2025, arXiv/工业报告 | item token；lazy decoder-only 表示 | 高效 decoder-only 页面生成，duration-aware RL 与自适应 clipping | [论文](https://arxiv.org/abs/2508.20900)；未发现完整官方代码 | ⬜ 未实现 | P3：作为系统设计参考 |
| MTGR | 2025, CIKM（工业） | 保留 DLRM 交叉特征的生成式 item token | HSTU 式多位置生成，用户序列压缩、GLN 与动态 mask | [论文](https://arxiv.org/abs/2505.18654) / [正式版](https://doi.org/10.1145/3746252.3761565)；未发现官方代码 | ⬜ 未实现 | P3：工业特征接口差异很大 |
| PLUM | 2025, arXiv/YouTube | LLM 词表与 SID 共存 | 持续预训练再任务微调，用统一生成器完成大规模检索 | [论文](https://arxiv.org/abs/2510.07784)；未发现官方代码 | ⬜ 未实现 | P2：适合作为规模化预训练路线 |
| MiniOneRec | 2025, 技术报告 | RQ-VAE SID + Qwen 0.5B–7B | SFT 强制全流程 SID 对齐，随后约束生成与混合奖励 RL | [论文](https://arxiv.org/abs/2510.24431) / [代码](https://github.com/AkaliKong/MiniOneRec) | 🧪 本仓库 OneRec 与其高度相关，但 tokenizer、任务和 RL 细节并非逐项 parity | P1：做配置级兼容层，而非再写一套模型 |
| OpenOneRec | 2025 报告，2026 开源 | Qwen3 1.7B/8B + itemic token | 预训练、SFT、on-policy distillation、RL 和指令评测全链路 | [论文](https://arxiv.org/abs/2512.24762) / [代码](https://github.com/Kuaishou-OneRec/OpenOneRec) | 🧪 本仓库有较轻量 SFT/GRPO/CLM 路线，但不等价 | P1：优先接数据/评测协议，再考虑权重兼容 |
| RecGPT（顺序基础模型） | 2025, EMNLP | 纯文本物品表示经 FSQ 得到跨域 token | decoder-only，自回归历史建模；item 内双向、item 间因果注意力；catalog-aware beam | [论文](https://aclanthology.org/2025.emnlp-main.513/) / [代码](https://github.com/HKUDS/RecGPT) | ⬜ 未实现 | P1：跨域/冷启动的重要开放基线 |
| MaskGR | 2025, arXiv | 多层 SID 序列 | 离散 masked diffusion，迭代并行去掩码，替代严格左到右生成 | [论文](https://arxiv.org/abs/2511.23021) / [代码](https://github.com/snap-research/MaskGR) | ⬜ 未实现 | P1：与现有 SID 数据最大程度复用 |
| SpecGR | 2026, AAAI | 已见/新物品候选与 SID/内容表示 | 可扩展 drafter 提案，生成式 verifier 接受/拒绝，并引导重提案 | [论文](https://ojs.aaai.org/index.php/AAAI/article/view/38486) / [代码](https://github.com/Jamesding000/SpecGR) | ⬜ 未实现 | P2：适合专门的 inductive 评测 |
| Align³GR | 2026, AAAI | item token、行为与偏好三层对齐 | 生成 SID，并用 progressive DPO 逐步对齐 | [论文](https://ojs.aaai.org/index.php/AAAI/article/view/38651)；未发现官方代码 | ⬜ 未实现 | P2：与 RL 路线有交叉 |
| GenCDR | 2026, AAAI | 通用 encoder + 域专属 adapter 动态路由得到跨域 SID | 跨域自回归模块融合通用/域专属兴趣，并以 domain-aware PrefixTree 解码 | [论文](https://ojs.aaai.org/index.php/AAAI/article/view/38508) / [代码](https://github.com/hupeiyu21/GenCDR) | ⬜ 未实现 | P2：需先增加跨域数据协议 |
| MACRec | 2026, AAAI | 多视角跨模态量化 SID | 在 ID 学习时 cross-modal quantization 降冲突，并在生成器中做隐式/显式跨模态对齐 | [论文](https://ojs.aaai.org/index.php/AAAI/article/view/38664) / [代码](https://github.com/zhangfw123/MACRec) | ⬜ 未实现 | P2：多模态 tokenizer 的代表对照 |
| MusicRec | 2026, AAAI | 共享/专属模态融合 + 协同信号的 SID | 多模态与行为共同构造 identifier，再由 LLM 自回归生成 | [论文](https://ojs.aaai.org/index.php/AAAI/article/view/38685)；未发现官方代码 | ⬜ 未实现 | P3：先建立多模态数据管线 |

## 2. LLM 文本、原生词表与生成式排序/列表生成

| 模型 | 年份/出处 | 核心表示 | 生成机制 | 论文 / 官方代码 | 本仓库状态 | 建议 |
|---|---|---|---|---|---|---|
| P5 | 2022, RecSys | 用户、物品 ID 与自然语言统一成文本 prompt | T5 text-to-text，多任务生成 item、评分、解释等 | [论文](https://arxiv.org/abs/2203.13366) / [代码](https://github.com/jeykigung/P5) | ⬜ 未实现 | P2：经典统一范式，算力可控 |
| P5 Item Indexing | 2023, arXiv | 顺序、协同、语义和混合 item identifier | 在 P5 中把下一物品当 token 串生成 | [论文](https://arxiv.org/abs/2305.06569) / [代码](https://github.com/Wenyueh/LLM-RecSys-ID) | ⬜ 未实现 | P1：低成本 tokenizer 基线 |
| GenRec（LLM） | 2024, ECIR | 交互历史和物品标题/描述文本 | 微调 LLaMA 直接生成目标物品文本 | [论文](https://arxiv.org/abs/2307.00457) / [正式版](https://doi.org/10.1007/978-3-031-56063-7_42) / [代码](https://github.com/rutgerswiselab/GenRec) | ⬜ 未实现 | P2：用于量化文本幻觉/grounding |
| BIGRec | 2023, arXiv | LLM 生成的语义 item 文本/向量 | 先生成目标语义，再用向量相似度 grounding 到真实目录 | [论文](https://arxiv.org/abs/2308.08434) / [代码](https://github.com/SAI990323/Grounding4Rec) | ⬜ 未实现 | P1：文本生成与目录检索的清晰桥梁 |
| IDGenRec | 2024, SIGIR | 由 LLM 学出的可读 textual ID | 联合学习 ID 生成与推荐，支持跨域零样本 | [论文](https://arxiv.org/abs/2403.19021) / [代码](https://github.com/agiresearch/IDGenRec) | ⬜ 未实现 | P2：适合跨域研究 |
| RecGPT（文本模型） | 2024, ACL | 领域文本预训练的 7B LLM | 生成式预训练 + instruction tuning，输出文本/评分/推荐 | [论文](https://aclanthology.org/2024.acl-short.29/) / [代码](https://github.com/VinAIResearch/RecGPT) | ⬜ 未实现 | P3：与 2025 同名 RecGPT 必须分开 |
| NoteLLM | 2024, WWW | 特殊 token 压缩出的 note embedding；标题、类别、hashtag 文本 | LLM 生成辅助文本，同时用对比学习做 I2I 向量检索 | [论文](https://arxiv.org/abs/2403.01744) / [代码](https://github.com/Applied-Machine-Learning-Lab/NoteLLM) | 🧪 已实现对称 GCL+CSFT LoRA Amazon adaptation、同物品伪负例屏蔽与 exact full-catalog I2I item-level 评测；论文 hard-negative 闭环未对齐 | P2：补 hard-negative 与官方 parity |
| NoteLLM-2 | 2025, KDD ADS | 文本 LLM + vision encoder 的多模态 note embedding | multimodal in-context prompt 平衡模态，或把视觉信息 late-fuse 到最终表示；仍以 I2I 检索推荐 | [论文](https://arxiv.org/abs/2405.16789) / [代码](https://github.com/Applied-Machine-Learning-Lab/NoteLLM) | ⬜ 未实现；与 NoteLLM 共用官方仓库 | P3：属生成增强表示边界项，需多模态数据 |
| LAMIA（早期 v1 名为 STORE） | 2024, arXiv（v3 于 2025 改名） | 多个相互独立、语义并行的 aspect embedding 组成 item palette | 域内文本重构调优语义 encoder，再对多 aspect palette 做 semantic tokenization | [当前论文](https://arxiv.org/abs/2409.07276) / [STORE v1](https://arxiv.org/abs/2409.07276v1)；未发现官方代码 | ⬜ 未实现 | P2：多 aspect tokenizer 的代表 |
| NAR4Rec | 2024, KDD | 候选 item 的目标排列 | 非自回归生成多个重排序列；matching model、序列级 unlikelihood 与 contrastive decoding | [论文](https://arxiv.org/abs/2402.06871)；未发现官方代码 | ⬜ 未实现 | P2：实时生成式重排的重要对照 |
| GenRank（大规模生成排序） | 2025, arXiv/工业 | item 作为上下文与位置信号，action 作为生成单元 | action-oriented 序列组织，对候选 item 预测用户 action；candidate mask 防候选泄漏，线性 position/time embedding + ALiBi | [论文](https://arxiv.org/abs/2505.04180)；未发现官方代码 | ⬜ 未实现 | P2：需新建候选 action ranking 与延迟协议 |
| GReF | 2025, CIKM | 候选集合到有序列表的多 token 表示 | ordered multi-token prediction + Rerank-DPO 生成重排结果 | [论文](https://arxiv.org/abs/2510.25220)；未发现官方代码 | ⬜ 未实现 | P2：与检索模型分层评测 |

> **同名提醒：** “RecGPT”至少指 ACL 2024 的文本推荐 LLM、EMNLP 2025 的 FSQ 顺序基础模型，以及其他 prompt/工业报告。本文始终带副标题或年份；实验记录也必须这样命名。“GenRec”同样既是本仓库名，也被多篇互不等价的论文使用。

## 3. 生成式序列建模与工业基础模型

| 模型 | 年份/出处 | 核心表示 | 生成机制 | 论文 / 官方代码 | 本仓库状态 | 建议 |
|---|---|---|---|---|---|---|
| HSTU | 2024, ICML | 原子 item/action ID + 时间与相对注意力偏置 | 对序列多位置做生成式 transduction；最终仍在 item 空间打分 | [论文](https://arxiv.org/abs/2402.17152) / [代码](https://github.com/meta-recsys/generative-recommenders) | ✅ 已实现；含 SiLU attention、Update gate、temporal RAB | P0：稳定的 item-level 强基线 |
| GPR | 2025, arXiv/腾讯广告 | 广告与自然内容统一 SID | hierarchical hybrid decoder，多 token 预测、value 微调与 HEPO | [论文](https://arxiv.org/abs/2511.10138)；未发现官方代码 | ⬜ 未实现 | P3：工业多目标数据不可得 |
| GenRec（京东） | 2026, SIGIR（工业系统） | page-wise 物品 token；asymmetric token merger | 页面 next-token prediction，GRPO-SR 混合奖励优化整页 | [论文](https://arxiv.org/abs/2604.14878)；未发现官方代码 | ⬜ 未实现 | P3：跟踪系统设计，不做数字复现承诺 |

HSTU 被收录是因为论文明确把大规模推荐表述为生成式序列 transduction；它不生成多 token SID，因此其存储、解码和碰撞性质与 TIGER/OneRec 不同。SASRec 只作为判别式 softmax 基线，不列为生成模型。

## 4. 扩散、去噪与流匹配推荐

| 模型 | 年份/出处 | 核心表示 | 生成机制 | 论文 / 官方代码 | 本仓库状态 | 建议 |
|---|---|---|---|---|---|---|
| DiffRec | 2023, SIGIR | 全目录用户交互向量；L-DiffRec 用低维 latent | 高斯前向加噪、MLP 反向去噪；T-DiffRec 加时间权重 | [论文](https://arxiv.org/abs/2304.04971) / [代码](https://github.com/YiyanXu/DiffRec) | ⬜ 未实现 | P1：扩散 CF 的标准起点 |
| DiffuRec | 2023/2024, TOIS | 目标 item embedding 的条件分布 | 对目标向量加噪，按历史条件反演，再 rounding 到物品 | [论文](https://arxiv.org/abs/2304.00686) / [代码](https://github.com/WHUIR/DiffuRec) | ⬜ 未实现 | P1：顺序连续扩散基线 |
| DreamRec | 2023, NeurIPS | “理想目标物品”连续向量 | Transformer 历史 guidance 下从噪声生成目标向量，再近邻落地 | [论文](https://arxiv.org/abs/2310.20453) / [代码](https://github.com/YangZhengyi98/DreamRec) | ⬜ 未实现 | P1：与 SID 生成的输出空间对照 |
| DimeRec | 2025, WSDM | 多兴趣向量 | stationary 多兴趣 guidance + diffusion 聚合生成目标表示 | [论文](https://arxiv.org/abs/2408.12153)；未发现官方代码 | ⬜ 未实现 | P2：关注多兴趣和工业延迟 |
| DDSR | 2024, NeurIPS | RQ-VAE 语义标签上的离散 fuzzy state | 离散状态空间扩散，逐步锐化目标物品分布 | [论文](https://arxiv.org/abs/2410.23994)；未发现官方代码 | ⬜ 未实现 | P2：离散扩散与 SID 的桥梁 |
| PreferDiff | 2025, ICLR | item preference 表示 | 条件扩散并加入个性化 ranking loss | [论文](https://arxiv.org/abs/2410.13117) / [代码](https://github.com/lswhim/PreferDiff) | ⬜ 未实现 | P1：代码开放、Beauty/Sports/Toys 可对齐 |
| DiQDiff | 2025, WWW | 语义向量量化 guidance | SVQ 量化语义引导扩散，并用 Contrastive Discrepancy Maximization 增强目标区分度 | [论文](https://arxiv.org/abs/2501.17670) / [正式版](https://doi.org/10.1145/3696410.3714955) / [代码](https://github.com/maowenyu-11/DiQDiff) | ⬜ 未实现 | P2：与本仓库量化器有复用点 |
| FlowCF | 2025, KDD | 隐式反馈的离散交互状态 | 离散 flow matching 从行为先验流向用户偏好分布 | [论文](https://arxiv.org/abs/2502.07303) / [代码](https://github.com/chengkai-liu/FlowCF) | ⬜ 未实现 | P2：用少步推理对照 diffusion |
| DiffDiv | 2025, SIGIR | 目标 item embedding / 多样性 guidance | 多样性引导的条件扩散顺序推荐 | [论文](https://doi.org/10.1145/3726302.3730109)；未发现官方代码 | ⬜ 未实现 | P3：先建立准确率–多样性联合指标 |
| MDGR | 2026, arXiv | 多 codebook SID | 时间/样本 mask 训练，两阶段并行去掩码生成 | [论文](https://arxiv.org/abs/2601.19501)；论文称录用后开源，截至快照未确认 | ⬜ 未实现 | P2：MaskGR 之后再做 |

MaskGR 同时属于 SID 和扩散路线，已放在第 1 节，不重复计为第二个模型。扩散模型之间输出的是交互向量、目标 embedding 或 SID，评测协议见[扩散与流模型小节](evaluation.md#7)。

## 5. 强化学习、偏好优化与显式推理

| 模型 | 年份/出处 | 核心表示 | 生成机制 | 论文 / 官方代码 | 本仓库状态 | 建议 |
|---|---|---|---|---|---|---|
| ReRe | 2025, arXiv | 推荐候选/SID 与偏好 reward | recommendation-oriented reinforced preference optimization | [论文](https://arxiv.org/abs/2510.12211) / [代码](https://github.com/sober-clever/ReRe) | 🧪 本仓库 GRPO 路线相关但并非 ReRe parity | P1：统一 reward/rollout 接口 |
| Rec-R1 | 2025, TMLR | LLM 文本推理与推荐动作 | 用冻结黑盒推荐器反馈构成闭环 RL，优化搜索/顺序推荐 | [论文](https://arxiv.org/abs/2503.24289) / [代码](https://github.com/linjc16/Rec-R1) | ⬜ 未实现 | P2：需要外部 recommender 环境 |
| LatentR³ | 2026, ICLR（预印本 2025） | 紧凑 latent reasoning token | SFT 后用修改版 GRPO 优化推理和推荐结果 | [论文](https://arxiv.org/abs/2505.19092) / [ICLR 页面](https://iclr.cc/virtual/2026/poster/10008315) / [代码](https://github.com/xuwenxinedu/R3) | ⬜ 未实现 | P2：需先定义可验证 latent reward |
| R²Rec / Reason-to-Recommend | 2025, arXiv | 交互图中的 item-over-time 推理链 | 构造推理轨迹，SFT + RL 生成理由与推荐 | [论文](https://arxiv.org/abs/2506.05069) / [作者提供的匿名代码快照](https://anonymous.4open.science/r/R2Rec-7C5D)；尚未确认稳定公开 GitHub | ⬜ 未实现 | P2：图数据和解释评测成本较高 |
| OneRec-Think | 2026, ACL | itemic token + 文本推理轨迹 | item-text 对齐、reasoning scaffolding、推荐专用 RL；Think-Ahead 部署 | [论文](https://aclanthology.org/2026.acl-long.123/) / [代码](https://github.com/wangshy31/OneRec-Think) | ⬜ 未实现 | P2：优先复现可观测 reward，不追工业模块 |
| PROMISE | 2026, arXiv | SID 每一步的过程价值 | process reward model 评估部分 SID，PRM-guided beam 做测试时扩展 | [论文](https://arxiv.org/abs/2601.04674)；未发现官方代码 | ⬜ 未实现 | P1：可直接研究现有 trie/beam 的 test-time scaling |
| DASO | 2026, arXiv 预印本 | SID 前缀难度与 prefix-level reward | 按前缀深度分配 GRPO rollout，prefix-guided completion + SFT anchor | [论文](https://arxiv.org/abs/2608.20611)；未发现官方代码 | ⬜ 未实现 | P2：非常新，先独立验证再并入主线 |

这组工作不能只报告最终 Recall。至少还要报告有效 rollout 比例、全零 reward 组、平均/分位生成长度、KL、reward 分量、搜索预算和 SFT 起点。仓库现有经验表明，稀疏 exact reward 下的全零组是数据事实，并不等于奖励函数实现错误。

## 6. Tokenizer、量化与唯一寻址组件

以下条目本身不一定是推荐器，但直接决定 SID 模型是否健康、可寻址和可泛化。

| 组件/方法 | 年份/出处 | 核心表示 | 生成/量化机制 | 论文 / 官方代码 | 本仓库状态 | 建议 |
|---|---|---|---|---|---|---|
| VQ-VAE | 2017, NeurIPS | 单层离散 latent code | encoder 最近邻量化 + straight-through + decoder 重构 | [论文](https://arxiv.org/abs/1711.00937) / [示例代码](https://github.com/google-deepmind/sonnet) | ⬜ 未单列实现 | P3：概念祖先，RQ 路线已覆盖 |
| RQ-VAE | 2022, CVPR | 多层残差 code | 每层量化上一层残差，联合重构与 commitment loss | [论文](https://arxiv.org/abs/2203.01941) / [代码](https://github.com/kakaobrain/rq-vae-transformer) | 🧩 已实现并供 TIGER/LC-Rec/COBRA 使用 | P0：统一健康度报告 |
| RQ-KMeans / residual K-Means | 2025, GRID/OneRec 实践 | 多层残差聚类码 | 逐层 K-Means 量化残差，无神经 decoder | [GRID](https://arxiv.org/abs/2507.22224) / [GRID 代码](https://github.com/snap-research/GRID) | 🧩 已实现；含 dead-cluster 重置，供 OneRec 使用 | P0：作为 RQ-VAE 的简单强基线 |
| FSQ | 2024, ICLR | 各维有限标量级别的笛卡尔积 code | 无码本查找/commitment loss 的有限标量量化 | [论文](https://arxiv.org/abs/2309.15505) / [代码](https://github.com/google-deepmind/fsq) | ⬜ 未实现 | P1：RecGPT tokenizer 所需且稳定 |
| CoST | 2024, RecSys | 关系保持的语义 token | contrastive quantization 保留物品邻接关系 | [论文](https://arxiv.org/abs/2404.14774)；未发现官方代码 | ⬜ 未实现 | P2：用于 tokenizer 消融 |
| LETTER | 2024, CIKM | 语义、协同与均衡的层次 SID | RQ-VAE + 协同对比对齐 + code diversity regularization | [论文](https://arxiv.org/abs/2405.07314) / [代码](https://github.com/honghuibao2000/LETTER) | ⬜ 未实现 | P1：直接对应码本坍缩研究 |
| Breaking the Hourglass | 2024, EMNLP Industry | codebook 使用分布 | 分析/缓解生成式推荐中的 codebook utilization imbalance | [论文](https://aclanthology.org/2024.emnlp-industry.50/)；未发现官方代码 | ⬜ 未实现 | P1：健康度指标和失败机理参考 |
| ETEGRec | 2025, SIGIR | 可被推荐目标反向塑造的 SID | tokenizer 与推荐器交替端到端训练，语义/偏好双对齐 | [论文](https://arxiv.org/abs/2409.05546) / [代码](https://github.com/BishopLiu/ETEGRec) | ⬜ 未实现 | P1：从两阶段走向联训的首选 |
| UNGER（早期版本 PRORec） | 2025, TOIS | 语义与协同知识自适应融合后的统一 code | 联合 cross-modality alignment 与 next-item prediction 学习融合表示，再用 intra-modality KD 补偿量化信息损失 | [当前论文](https://arxiv.org/abs/2502.06269) / [PRORec v1](https://arxiv.org/abs/2502.06269v1)；未发现官方代码 | ⬜ 未实现 | P2：LETTER/ETEGRec 后比较 |
| UTGRec | 2025, arXiv | 多模态、树结构通用 token | 在多域多模态表示上学习可迁移层次 codebook | [论文](https://arxiv.org/abs/2504.04405) / [代码](https://github.com/zhengbw0324/UTGRec) | ⬜ 未实现 | P2：需图像与跨域数据管线 |
| Purely Semantic Indexing | 2025, arXiv | 无随机歧义后缀的唯一语义 ID | ECM/RRS 在保语义的同时重排冲突、保证唯一 | [论文](https://arxiv.org/abs/2509.16446)；未发现官方代码 | ⬜ 未实现 | P1：碰撞修复的直接对照 |
| Pctx | 2025, arXiv | 同一 item 的上下文个性化多 SID | 按上下文选择语义 facet，解码后对同一物品概率聚合 | [论文](https://arxiv.org/abs/2510.21276) / [代码](https://github.com/YoungZ365/Pctx) | ⬜ 未实现 | P2：先解决多 ID 评测语义 |
| UniSID | 2026, arXiv | 从原始广告内容学习的统一 SID | 多粒度对比 + summary reconstruction，端到端构造 token | [论文](https://arxiv.org/abs/2602.10445)；未发现官方代码 | ⬜ 未实现 | P3：工业多模态数据依赖重 |
| Variable-Length Semantic IDs | 2026, arXiv | 自适应长度 SID | discrete VAE + Gumbel-Softmax 学习何时停止编码 | [论文](https://arxiv.org/abs/2602.16375) / [代码](https://github.com/KhrylchenkoKirill/varlen_semantic_ids) | ⬜ 未实现 | P2：适合研究热门度–码长关系 |
| QuaSID | 2026, arXiv | 工业规模唯一 SID | qualification-aware collision repulsion，量化时主动排斥冲突 | [论文](https://arxiv.org/abs/2603.00632)；未发现官方代码 | ⬜ 未实现 | P2：先在小目录验证可扩展性 |
| VaLiDRec | 2026, arXiv | LLM 原生词表中的可变长 token 串 | 剪枝/碰撞细化生成唯一 ID，并以 token-set 目标训练 | [论文](https://arxiv.org/abs/2607.25209)；未发现官方代码 | ⬜ 未实现 | P2：新工作，先验证目录约束 |

## 7. 目录之外但需要持续观察的相邻路线

- **纯解释生成**：若推荐结果仍完全来自另一个判别式排序器，只把 LLM 用来写理由，不纳入模型主表。
- **纯对话推荐**：只有自然语言对话、没有可复现的 catalog grounding 和 item-level 结果，不纳入。
- **生成式数据增强**：GAN/VAE/LLM 只生成训练样本，而最终推荐器仍是普通判别式模型，不纳入；它是独立研究主题。
- **生成式重排/排序**：NAR4Rec/GReF 直接产生或并行补全有序列表；GenRank 生成候选 action 概率而不是 item 列表。两者都必须和全目录检索分层报告。
- **新近预印本**：2026 年条目只是截至快照日的研究信号，不代表已通过同行评议或适合立即工程化。
