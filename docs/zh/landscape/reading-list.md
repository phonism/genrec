# 阅读路线与原始资料

本页按“要回答什么问题”组织，不按引用量或发布时间简单堆论文。快照日期为 **2026-08-31**；模型的完整状态表见[模型目录](model-catalog.md)。

## 先读这五篇

1. [TIGER: Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)：理解内容 embedding → RQ-VAE SID → seq2seq → beam 的基本闭环。
2. [GRID: A Practitioner's Handbook](https://arxiv.org/abs/2507.22224) + [代码](https://github.com/snap-research/GRID)：理解为什么 tokenizer、数据窗口、解码与去重必须一起消融。
3. [Faithful Evaluation of Semantic-ID Tokenizers](https://arxiv.org/abs/2605.25330)：理解 SID hit 为什么可能夸大 item hit，以及碰撞纠正/零碰撞分配。
4. [HSTU: Actions Speak Louder than Words](https://arxiv.org/abs/2402.17152) + [代码](https://github.com/meta-recsys/generative-recommenders)：理解另一种“生成式序列 transduction”，避免把生成式推荐窄化为 LLM + SID。
5. [DiffRec](https://arxiv.org/abs/2304.04971) + [代码](https://github.com/YiyanXu/DiffRec)：理解对交互向量做去噪与生成 SID 是不同问题。

读完这五篇，再看某个新方法时应先问：输出空间是什么？目录怎样绑定？碰撞怎样处理？与 baseline 是否共享 split、内容和候选预算？

## 路线 A：语义 ID 生成式检索

### 奠基与对齐

- [TIGER, NeurIPS 2023](https://arxiv.org/abs/2305.05065)：层次 SID 自回归检索。
- [LC-Rec, ICDE 2024](https://arxiv.org/abs/2311.09049) / [官方代码](https://github.com/RUCAIBox/LC-Rec)：把协同语义对齐到 LLM，并直接生成目录 item。
- [LIGER, 2024](https://arxiv.org/abs/2411.18814) / [官方代码](https://github.com/facebookresearch/liger)：在同一输入条件下比较 dense 与 generative，再做混合检索。
- [GRAM, ACL 2025](https://aclanthology.org/2025.acl-long.1596/) / [官方代码](https://github.com/skleee/GRAM)：语义到 lexical ID 与多粒度 late fusion。

### 改变表示或分解

- [ActionPiece, ICML 2025](https://arxiv.org/abs/2502.13581) / [官方代码](https://github.com/google-deepmind/action_piece)：上下文 action tokenization。
- [COBRA, NeurIPS 2025](https://arxiv.org/abs/2503.02453)：离散码后接 dense 尾向量。
- [RPG, KDD 2025](https://arxiv.org/abs/2506.05781) / [官方代码](https://github.com/facebookresearch/RPG_KDD2025)：OPQ 子空间 code 组成论文所称的 unordered long SID；codebook 位置固定、独立 head 并行预测，长度按数据集选择。
- [RecGPT: A Foundation Model for Sequential Recommendation, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.513/) / [官方代码](https://github.com/HKUDS/RecGPT)：FSQ、跨域预训练和 catalog-aware beam。
- [MaskGR, 2025](https://arxiv.org/abs/2511.23021) / [官方代码](https://github.com/snap-research/MaskGR)：用 masked diffusion 替代左到右 SID 生成。
- [SpecGR, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38486) / [官方代码](https://github.com/Jamesding000/SpecGR)：drafter–verifier 处理 inductive retrieval。

### 工业扩展

- [OneRec V1](https://arxiv.org/abs/2502.18965) → [OneRec V2](https://arxiv.org/abs/2508.20900)：从 encoder-decoder 页面生成到 lazy decoder-only 与 RL。
- [MiniOneRec](https://arxiv.org/abs/2510.24431) / [代码](https://github.com/AkaliKong/MiniOneRec)：公开的 RQ-VAE + Qwen + SFT/RL 流程。
- [OpenOneRec](https://arxiv.org/abs/2512.24762) / [代码](https://github.com/Kuaishou-OneRec/OpenOneRec)：预训练、SFT、on-policy distillation、RL、RecIF-Bench。
- [PLUM](https://arxiv.org/abs/2510.07784)、[MTGR, CIKM 2025](https://arxiv.org/abs/2505.18654)、[GPR](https://arxiv.org/abs/2511.10138)、[京东 GenRec, SIGIR 2026](https://arxiv.org/abs/2604.14878)：用于理解规模化/多目标系统，不应在无数据和代码时承诺逐数字复现。

## 路线 B：Item tokenizer 与量化

建议按“通用量化 → 推荐约束 → 唯一寻址”阅读：

1. [VQ-VAE, 2017](https://arxiv.org/abs/1711.00937)；
2. [RQ-VAE, 2022](https://arxiv.org/abs/2203.01941) / [代码](https://github.com/kakaobrain/rq-vae-transformer)；
3. [FSQ, ICLR 2024](https://arxiv.org/abs/2309.15505) / [代码](https://github.com/google-deepmind/fsq)；
4. [CoST, RecSys 2024](https://arxiv.org/abs/2404.14774)：关系保持的 contrastive tokenization；
5. [LETTER, CIKM 2024](https://arxiv.org/abs/2405.07314) / [代码](https://github.com/honghuibao2000/LETTER)：语义、协同、diversity 三种正则；
6. [Breaking the Hourglass, EMNLP Industry 2024](https://aclanthology.org/2024.emnlp-industry.50/)：中间 code 过度集中；
7. [ETEGRec, SIGIR 2025](https://arxiv.org/abs/2409.05546) / [代码](https://github.com/BishopLiu/ETEGRec)：tokenizer 与推荐器交替联训；
8. [Purely Semantic Indexing](https://arxiv.org/abs/2509.16446)：不依赖随机歧义后缀的唯一化；
9. [Pctx](https://arxiv.org/abs/2510.21276) / [代码](https://github.com/YoungZ365/Pctx)：一个 item 多个上下文 SID；
10. [Variable-Length Semantic IDs](https://arxiv.org/abs/2602.16375) / [代码](https://github.com/KhrylchenkoKirill/varlen_semantic_ids)、[QuaSID](https://arxiv.org/abs/2603.00632)、[VaLiDRec](https://arxiv.org/abs/2607.25209)：可变长、冲突排斥和原生 LLM 词表的新方向。

实现研究前，还应读 [GRID](https://arxiv.org/abs/2507.22224)：其开放框架同时提供 RK-Means、R-VQ/RQ-VAE 和统一解码，是验证“复杂 tokenizer 是否真的胜过简单聚类”的重要参照。

## 路线 C：文本生成、LLM、生成式排序与基础模型

- [P5, RecSys 2022](https://arxiv.org/abs/2203.13366) / [代码](https://github.com/jeykigung/P5)：prompt 化的 text-to-text 多任务推荐。
- [How to Index Item IDs, 2023](https://arxiv.org/abs/2305.06569) / [代码](https://github.com/Wenyueh/LLM-RecSys-ID)：顺序/协同/语义/混合 ID 的直接比较。
- [GenRec: LLM for Generative Recommendation, ECIR 2024](https://doi.org/10.1007/978-3-031-56063-7_42) / [代码](https://github.com/rutgerswiselab/GenRec)：直接生成目标物品文本。
- [BIGRec, 2023](https://arxiv.org/abs/2308.08434) / [代码](https://github.com/SAI990323/Grounding4Rec)：生成语义后 grounding 到真实目录。
- [IDGenRec, SIGIR 2024](https://arxiv.org/abs/2403.19021) / [代码](https://github.com/agiresearch/IDGenRec)：生成 textual identifiers 并做跨域零样本。
- [RecGPT: Generative Pre-training for Text-based Recommendation, ACL 2024](https://aclanthology.org/2024.acl-short.29/) / [代码](https://github.com/VinAIResearch/RecGPT)：领域文本 LLM；不要与 EMNLP 2025 的 FSQ RecGPT 混淆。
- [NoteLLM, WWW 2024](https://arxiv.org/abs/2403.01744) / [NoteLLM/NoteLLM-2 代码](https://github.com/Applied-Machine-Learning-Lab/NoteLLM)：生成 hashtag/category/摘要帮助学习可检索 embedding，属于边界路线；[NoteLLM-2](https://arxiv.org/abs/2405.16789)进一步加入视觉 encoder。
- [LAMIA（2024 投稿，2025 v3 改名）](https://arxiv.org/abs/2409.07276)（[v1 名为 STORE](https://arxiv.org/abs/2409.07276v1)）：当前版本以多 aspect item palette 做 semantic tokenization；v1 的 text↔token 联合生成仅作历史版本阅读。
- [NAR4Rec, KDD 2024](https://arxiv.org/abs/2402.06871) 与 [GReF, CIKM 2025](https://arxiv.org/abs/2510.25220)：前两者研究有序列表生成；[GenRank, 2025](https://arxiv.org/abs/2505.04180)则是 action-oriented candidate ranking，生成候选 action 概率而不直接生成 item 排列。

## 路线 D：扩散与流

按输出对象阅读，避免把所有 diffusion 数字混在一起：

- **交互向量**：[DiffRec, SIGIR 2023](https://arxiv.org/abs/2304.04971) / [代码](https://github.com/YiyanXu/DiffRec)。
- **连续目标物品**：[DiffuRec, TOIS](https://arxiv.org/abs/2304.00686) / [代码](https://github.com/WHUIR/DiffuRec)；[DreamRec, NeurIPS 2023](https://arxiv.org/abs/2310.20453) / [代码](https://github.com/YangZhengyi98/DreamRec)。
- **多兴趣/偏好**：[DimeRec, WSDM 2025](https://arxiv.org/abs/2408.12153)；[PreferDiff, ICLR 2025](https://arxiv.org/abs/2410.13117) / [代码](https://github.com/lswhim/PreferDiff)。
- **离散状态/SID**：[DDSR](https://arxiv.org/abs/2410.23994)、[MaskGR](https://arxiv.org/abs/2511.23021)、[MDGR](https://arxiv.org/abs/2601.19501)。
- **流匹配**：[FlowCF, KDD 2025](https://arxiv.org/abs/2502.07303) / [代码](https://github.com/chengkai-liu/FlowCF)。

阅读时始终标注：它生成的是 score vector、embedding 还是 SID；最终 item 是全目录排序、最近邻还是约束解码得到。

## 路线 E：RL、推理与测试时扩展

- [ReRe](https://arxiv.org/abs/2510.12211) / [代码](https://github.com/sober-clever/ReRe) 与 [MiniOneRec](https://arxiv.org/abs/2510.24431) / [代码](https://github.com/AkaliKong/MiniOneRec)：推荐专用偏好优化和公开后训练流程。
- [Rec-R1, TMLR 2025](https://arxiv.org/abs/2503.24289) / [代码](https://github.com/linjc16/Rec-R1)：黑盒 recommender 反馈闭环 RL。
- [LatentR³, ICLR 2026](https://arxiv.org/abs/2505.19092) / [ICLR 页面](https://iclr.cc/virtual/2026/poster/10008315) / [代码](https://github.com/xuwenxinedu/R3)：latent reasoning token + GRPO。
- [Reason-to-Recommend](https://arxiv.org/abs/2506.05069) / [作者提供的匿名代码快照](https://anonymous.4open.science/r/R2Rec-7C5D)：从交互图构造推理轨迹；尚未确认稳定公开 GitHub。
- [OneRec-Think, ACL 2026](https://aclanthology.org/2026.acl-long.123/) / [代码](https://github.com/wangshy31/OneRec-Think)：itemic alignment、reasoning scaffolding 和推荐 reward。
- [PROMISE](https://arxiv.org/abs/2601.04674)：SID 步级过程奖励与 PRM-guided beam。
- [DASO](https://arxiv.org/abs/2608.20611)：按 SID 前缀难度分配 rollout；截至快照极新，应先独立核验。

这条路线的重点不是“用了 GRPO”，而是正 reward 是否可采到、reward 是否泄漏 target、约束生成是否改变探索空间，以及相同推理预算下是否仍有增益。

## 路线 F：评测与反例

- [Faithful Evaluation of Semantic-ID Tokenizers, CIKM 2026](https://arxiv.org/abs/2605.25330)：碰撞纠正 item metrics 与 zero-collision reassignment。
- [Understanding Semantic IDs, 2026](https://arxiv.org/abs/2607.24995)：表示邻域损失、前缀过滤和 Item-Supported Decoding。
- [Can Generative Recommendation Reach Cold Items?, 2026](https://arxiv.org/abs/2607.21101)：absolute-time 下按 token/prefix 可达性分析冷物品。
- [Understanding Generative Recommendation from a Model-scaling View, KDD 2026（预印本始于 2025）](https://arxiv.org/abs/2509.25522)：分别检查 encoder、tokenizer 与 generator 扩展，避免“更大必然更好”。
- [GRID, 2025](https://arxiv.org/abs/2507.22224)：滑窗、去重、码长/宽度和架构消融。

这些工作不只是“相关研究”。它们直接决定旧排行榜是否可信，应在新模型实现前阅读。

## 来源与维护规则

资料可信度按以下顺序：

1. 会议/期刊正式页、ACL Anthology、OpenReview；
2. arXiv 原文及其版本历史；
3. 作者/机构官方 GitHub、项目页、模型卡；
4. 综述、教程和 awesome 列表仅用于发现候选；
5. 博客、媒体摘要和第三方复现不能单独证明机制或官方结果。

更新本页时应记录核验日期；把“arXiv 预印本”与“已录用”分开；论文声称“代码即将发布”不等于已有官方代码。若链接失效，优先保留 DOI/Anthology/arXiv 永久入口。
