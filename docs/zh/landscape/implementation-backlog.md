# 实现与研究待办

这份待办面向 GenRec 的定位：公平比较、可复现和研究沙箱，而不是追逐最多模型数量。优先级定义见[总览](index.md)，验收统一遵守[评测协议](evaluation.md)。

## 决策原则

一个新实现进入主矩阵前，依次回答：

1. 它能否复用现有数据、SID 或评测，从而形成**单变量对照**？
2. 它回答的是新科学问题，还是只是换 backbone？
3. 官方代码/数据是否足以做 parity，而不是凭摘要猜实现？
4. item-level、碰撞和计算预算能否诚实对齐？
5. 预期维护成本是否与证据价值相称？

因此当前最值得做的不是再接一个超大 LLM，而是先把 collision-corrected evaluation、简单 tokenizer 基线和不同生成分解补齐。

## P0：先修评测和现有实现

### P0.1 统一物品级生成评测

**范围**：TIGER、LC-Rec、OneRec、GRAM 以及以后所有 SID 模型。

**工作项**：

- 统一 `generated tokens → valid SID → train-only SID resolver → unique item list → metric`；
- 正式指标只输出 `item_recall/item_ndcg`，旧 tuple 指标改名 `sid_hit`；
- resolver 和 trie 保存输入 catalog hash、训练交互 hash；
- 输出 invalid、duplicate、valid unique yield、collision bucket 等诊断；
- 对 COBRA/RPG 复用相同结果 schema，即使它们已是 item-level。

**验收**：构造一个含一对多 SID 的微型目录，SID hit 与 item hit 必须出现预期差异；测试标签改变不能影响 resolver；beam 内同一 item 多次出现只占一个 rank。

### P0.2 重评已有模型

**范围**：Amazon 2014 Beauty/Sports/Toys，至少 3 seeds。

**顺序**：SASRec/HSTU → TIGER/LC-Rec → COBRA/RPG → OneRec SFT/GRPO → GRAM。

**验收**：

- 数据 manifest、tokenizer hash、checkpoint 选择和 decoding config 完整；
- 每个模型逐 seed JSON 与均值/标准差可重新聚合；
- SID 历史数字单列为 legacy，不与新 item-level 主表混用；
- 失败/OOM run 有记录，不静默删除；
- 生成模型同时报告训练和解码成本。

### P0.3 码本健康度门禁

**范围**：RQ-VAE、RQ-KMeans、OPQ 和以后所有 tokenizer。

**最低报告**：每层 usage/entropy/perplexity/dead codes/max frequency，完整 SID unique rate、碰撞桶 p95/max、重构误差、逐前缀分布、item-neighborhood preservation。

**门禁建议**：若存在大碰撞桶、单 code 吞掉异常比例物品或有效路径显著少于目录，应阻止下游训练并要求显式 `--allow-unhealthy-codebook` 才能继续。阈值由数据规模标定，不能写死一个跨数据集百分比。

### P0.4 数据与结果 manifest

为 Amazon 2014/2023 明确 source version、category、core、split、metadata 可见范围、ID map、raw/processed hash。结果表拒绝只有 `amazon/beauty` 而无版本/协议的 run。

## P1：形成有解释力的最小扩展

| 项目 | 研究问题 | 可复用资产 | 最小实现 | 验收标准 | 规模 |
|---|---|---|---|---|---|
| P1.1 GRID 对齐的 tokenizer 基线 | 神经 RQ-VAE 是否胜过简单残差聚类？码长/宽度如何影响碰撞？ | 现有 RQVAE/RQKMeans、Amazon adapter、trie | 统一 tokenizer protocol，加入 R-VQ/标准 RK-Means 配置和相同 decoder | 相同 embedding、SID 容量、generator、seed；报告健康度和 item-level 指标 | M |
| P1.2 MaskGR | 严格自回归是否是 SID 性能/延迟瓶颈？ | 现有 SID dataset、token vocab、item evaluator | time/sample mask、离散去掩码 schedule、catalog constraint | 与 TIGER 使用同一 SID；匹配参数量并画 accuracy–steps–latency 曲线 | M–L |
| P1.3 LIGER | sparse SID 与 dense retrieval 各自在哪些 seen/cold 桶有效？ | TIGER encoder/decoder、item embedding、full-rank evaluator | dense head、生成候选融合与统一 reranker | 分别给 gen/dense/hybrid；固定候选数并做 absolute-time cold buckets | M |
| P1.4 LETTER | 当前 K=256 改进来自简单容量匹配还是协同/均衡目标？ | RQVAE trainer、codebook health | collaborative contrastive、diversity regularizer，可单独开关 | 每个 loss 独立消融；同 SID 容量和下游生成器；无 SID-level 主指标 | M |
| P1.5 扩散双基线 | 生成交互向量和生成目标 embedding 与 SID 生成有何差别？ | Amazon/MovieLens adapter、SASRec evaluator | DiffRec + DreamRec 或 DiffuRec，先忠实移植官方代码逻辑 | full-catalog item-level；固定 split/历史；报告 steps、NFE、近邻 grounding | M–L |
| P1.6 GRAM 闭环 | multi-granular late fusion 在统一数据上是否仍有增益？ | 已有模型/训练器/config | 官方数据转换器、shape/weight parity、统一 evaluator | 官方小样本 smoke；至少一个公开 split 接近论文趋势；item-level 主表 | M |
| P1.7 RecGPT-FSQ | 内容-only tokenizer 能否改善跨域与冷物品？ | 文本 encoder、trie、Amazon2023 | FSQ、hybrid attention、catalog-aware beam 的最小兼容实现 | 一个 seen 域 + 一个 zero-shot 域；token/prefix 冷启动分桶 | L |
| P1.8 PROMISE 式过程评估 | beam 失败主要发生在哪个 SID 层？测试时计算是否有效？ | OneRec/TIGER beam、现有 GRPO reward | prefix value/PRM 接口，先用可验证 oracle 诊断再训练 PRM | 同 expansion 预算优于普通 beam；报告逐层 target survival 和校准 | M |

`M/L` 只表示相对工程与实验规模，不是工期承诺。

### 推荐先后顺序

```text
P0 evaluator + manifests
        ├── P1.1 tokenizer controls ── P1.4 LETTER
        ├── P1.2 MaskGR ───────────── P1.8 process reward
        ├── P1.3 LIGER
        ├── P1.5 diffusion controls
        └── P1.6 GRAM closure ─────── P1.7 cross-domain RecGPT
```

P1.2 MaskGR 是最有信息量的单个新推荐器：它可沿用现有 SID、数据和 item resolver，只改变生成分解。P1.1 则是整个“码本坍缩”研究故事的必要控制组。

## P2：扩展任务边界

### P2.1 ETEGRec 端到端 tokenizer–recommender

在 LETTER 的固定两阶段基线上加入交替训练。必须冻结一个相同初始化的 tokenizer control，记录每轮 SID 漂移、历史 checkpoint 兼容性和碰撞变化；否则无法判断收益来自联合优化还是重新分配 ID。

### P2.2 文本生成与 grounding

选择 P5 + BIGRec，而不是一次接入所有 LLM 文本方法：

- P5 提供小模型、多任务 text-to-text 基线；
- BIGRec 提供“生成语义 → catalog grounding”基线；
- 共用 alias normalization、ungrounded/hallucination 和 oracle grounding 评测。

只有当两者协议稳定后，再考虑 IDGenRec/LAMIA（早期 v1 名为 STORE）。

### P2.3 OpenOneRec/MiniOneRec 兼容层

目标不是复制整个外部仓库，而是支持：

- 导入/导出 SID manifest；
- 复用其数据 task schema 和 constrained decoding；
- 映射 SFT/RL checkpoint metadata；
- 在本仓库 item-level evaluator 中评测外部结果。

本仓库 OneRec 的名字和文档必须持续声明“inspired/experimental”，直到 tokenizer、task mix、reward、decode 和指标逐项 parity。

### P2.4 RL 研究矩阵

固定同一 SFT checkpoint，只改变：

| 轴 | 值 |
|---|---|
| reward | exact、rank、hierarchical、prefix/process |
| rollout | group 8/16/32，是否 `add_gt` |
| decoding | free、trie-constrained、prefix-guided |
| anchor | 无、KL reference、SFT loss 混合 |
| budget | 固定 rollout token / GPU-hour |

主假设应围绕 reward sparsity、top-1 集中和 test-time search，而不是承诺 R@10 翻倍。

### P2.5 冷启动/跨域协议

接 Steam 或 Yelp，加 Amazon2023 absolute-time。先实现 LIGER/RecGPT 两类内容可泛化模型，再接 GenCDR/Pctx。验收必须区分 unseen item、unseen atomic token 和 unsupported prefix。

### P2.6 列表/页面生成与候选排序

NAR4Rec/GReF/OneRec page generation 需要新 task interface：一个样本含候选集/页面约束和多个有效目标。必须独立于 next-item trainer，指标包含顺序、重复、多样性、约束和延迟。不要把 listwise 方法强塞进单目标 Recall 接口。GenRank 则是候选 action 预测，需单独的 candidate-ranking interface，不能归入直接列表生成。

## P3：跟踪但暂不承诺复现

| 路线 | 原因 | 当前动作 |
|---|---|---|
| OneRec V1/V2、GPR、京东 GenRec、MTGR | 依赖私有规模、特征和线上 reward，完整代码/数据不可得 | 维护架构卡与可验证公开子模块，不承诺论文数字 |
| OneRec-Think 全链路 | 工业 reasoning 数据与部署结构不可得 | 先做可观测的 reasoning/reward ablation |
| 多模态 UTGRec/UniSID | 图像/视频数据与 encoder 成本高 | Amazon2023 小域建立 multimodal manifest 后再评估 |
| 可变长/上下文多 SID | evaluator 语义尚未稳定 | 先定义一个 item 多 ID 的概率聚合和唯一寻址测试 |
| 2026 极新预印本 | 状态和代码快速变化 | 每季度核验，不以摘要直接排入 benchmark |

## 每个新模型的 Definition of Done

“文件存在、能启动训练”不算完成。至少应满足：

1. **来源**：论文版本、官方代码 commit、实现差异清单；
2. **数据**：manifest/hash、无 test 泄漏、一个小样本 deterministic test；
3. **模型**：shape、mask、loss 和 decoding 单测；参数量与论文对照；
4. **评测**：item-level 主指标；碰撞/invalid/duplicate；全目录或明确候选集；
5. **复现**：至少一个论文数据集的趋势或数字核验；差异诚实记录；
6. **效率**：训练/评测时间、峰值显存、推理预算；
7. **稳定性**：至少 3 seeds；最佳 checkpoint 只由 validation 决定；
8. **文档**：命令、依赖、checkpoint 前置关系、已知限制和模型卡；
9. **回归**：CPU smoke 或小 GPU test，不破坏现有模型；
10. **结果**：机器可读逐 seed artifact，可由统一脚本聚合。

## 暂不做的反模式

- 在 P0 未完成前继续用 legacy SID-tuple 数字调参；
- 用 K=8192 在一万级目录上追求“更大容量”，却不看浅残差层坍缩；
- 为每篇论文复制一套 Amazon preprocessing 和 metric；
- 把官方仓库能跑等同于与本仓库协议可比；
- 把外部 LLM API 当前返回当稳定 benchmark；
- 只报最优 seed、最优 test epoch 或最有利的 K；
- 未实现 catalog grounding 就声称文本生成模型完成推荐；
- 将 tokenizer、推荐器、beam 和 resolver 的联合变化归因给单一模型模块。
