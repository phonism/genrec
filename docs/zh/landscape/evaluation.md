# 生成式推荐评测协议

生成式推荐最容易出现的错误，不是 Recall 公式写错，而是**模型输出、目录物品和指标对象没有对齐**。本页给出本仓库建议采用的最低协议。它以 item-level 推荐为主，兼顾 SID、文本、连续扩散、列表生成和 RL。

## 1. 先冻结任务定义

每个结果表必须同时给出以下字段：

| 字段 | 必填内容 |
|---|---|
| task | next-item、full-catalog retrieval、candidate reranking、page/slate generation 或 I2I |
| catalog | 训练时目录、评测时目录及物品数；是否允许未来新物品 |
| split | leave-one-out、per-user temporal、absolute-time temporal 或显式 cold-item split |
| candidates | 全目录、采样负例或给定候选集；采样器和负例数 |
| input | 最大历史长度、文本字段、图像/类别、时间信息 |
| target | 单一下一物品、多个可接受物品或有序列表 |
| grounding | SID/text/vector 如何变成真实 item ID |
| decoding | greedy/beam/top-p、beam 宽度、长度惩罚、trie、采样步数 |
| checkpoint | 只由 validation 指标选择，不得观察 test 选 epoch |
| seeds | 数据、tokenizer、模型和采样各自的种子 |

**全目录 Recall@K**与“1 个正例 + 99 个采样负例”的 Recall@K 不是同一指标；next-item 与候选重排也不是同一任务。禁止把它们放进同一列比较。

## 2. 划分协议

### Leave-one-out

每个用户最后一次交互作 test、倒数第二次作 validation，其余作 train。它适合与 TIGER/LC-Rec 等经典公开设置比较，但目录和内容通常是全局可见的，不能证明真正的时间冷启动。

### Per-user temporal

每个用户内部按时间排序再切分。它避免用户序列反序，但不同用户的“未来”仍可进入另一用户的训练集。应称为时间有序或 leakage-resistant，而不要称作严格 inductive。

### Absolute-time temporal

用全局时间点切 train/validation/test，tokenizer、热门度映射和目录统计也只能看过去。这是评估新物品可达性的首选。最新诊断表明，SID 模型能偶尔组合出由已见 token/前缀支持的未来物品，但对全新原子 token 和无支持路径仍困难，不能把“token 可组合”直接宣传成开放目录能力（[Can Generative Recommendation Reach Cold Items?](https://arxiv.org/abs/2607.21101)）。

### 推荐的双轨报告

1. **Transductive LOO**：与历史论文可比；
2. **Absolute-time inductive**：把 test target 分为 seen item、unseen item with seen tokens、unseen atomic token、unsupported prefix 四组。

两轨应共享清洗规则，但不能共享使用未来交互训练出的 tokenizer、SID→item 解析器或热门度。

## 3. 基础 item-level 指标

若每个样本只有一个目标 \(i^*\)，推荐列表为去重后的 \(R_K\)：

\[
\mathrm{Recall@K}=\mathbb{1}[i^*\in R_K],\qquad
\mathrm{NDCG@K}=\frac{\mathbb{1}[i^*\in R_K]}{\log_2(\operatorname{rank}(i^*)+1)}.
\]

数据集结果是对样本取均值。HitRate@K 在单目标设置与 Recall@K 相同，但文档应统一用一个名字。还应至少报告：

- Recall/NDCG @ 1、5、10、20；
- catalog coverage@K、long-tail coverage 和 popularity 分桶 Recall；
- 有效 item 数、重复率、无法落地率；
- 均值 ± 标准差（至少 3 个模型种子），最好再给 bootstrap 95% CI；
- 训练时间、峰值显存、item/token 参数量和每样本解码延迟。

多目标或 page/slate 任务应另报 Precision、MAP、MRR、ILD/多样性、重复率和业务约束满足率，不要套用单目标公式。

## 4. SID 碰撞：必须做物品级纠正

设完整 SID 为 \(s\)，碰撞桶为

\[
C(s)=\{i:Q(i)=s\}.
\]

只有当 \(|C(s)|=1\) 时，“生成目标 SID”才等价于“推荐目标物品”。[CIKM 2026 的 Faithful Evaluation](https://arxiv.org/abs/2605.25330)在其评测 tokenizer/数据上观察到碰撞率最高 30.52%，SID-level Hit@10 相对 item-level 最多膨胀 103.36%，甚至能翻转 tokenizer 排名。因此：

> **禁止把 SID tuple 命中作为正式推荐指标。** 它最多只能作为 tokenizer/生成器诊断项，必须明确标成 `sid_hit`。

### 本仓库建议的解析流程

对每个用户的 beam 输出依次执行：

1. 用 PrefixTrie/grammar 过滤不在**评测时可用目录**中的完整 SID；
2. 用只依赖训练数据构造的确定性 \(g(s)\) 将 SID 解析为一个 item；
3. 按 beam 顺序对 item ID 去重；
4. 去掉历史中不可重复推荐的 item（若任务这样定义）；
5. 截断到 K，再计算 item-level Recall/NDCG。

当前仓库对碰撞 SID 采用“训练交互中最热门 item”作为确定性解析器，并在物品级去重。这比任意把整个碰撞桶都算命中诚实得多，但它会偏向桶内热门物品，所以还应报告 oracle upper bound 和 zero-collision 版本。测试目标或测试热门度绝不能参与 \(g\)。

### 建议并行报告三种视图

| 视图 | 用途 | 能否做主结果 |
|---|---|---|
| collision-corrected item-level | 用训练期解析器得到真实推荐列表 | 是 |
| zero-collision assignment | 加唯一后缀或最小成本重分配后重训/重评 | 是，需说明 ID 改变 |
| SID-level tuple hit | 判断生成器是否走到同一语义桶 | 否，只是诊断 |

碰撞健康度至少包含：

- `unique_sid / num_items`；碰撞 SID 占比与受影响 item 占比；
- 碰撞桶 size 的 mean、p95、max；
- 每层 code usage、perplexity/entropy、dead-code 数、最大 token 频率；
- 完整路径和逐前缀的频率分布；
- invalid sequence rate、duplicate item rate、每个 beam 的 valid unique item yield；
- 使用热门解析、随机解析、dense 解析、zero-collision 后的指标差异。

仓库历史上的 K=8192 浅残差层坍缩正说明：高 code usage 也不够，必须同时看最大频率、逐层残差和完整路径分布。[Breaking the Hourglass](https://aclanthology.org/2024.emnlp-industry.50/)也把中间码过度集中视为生成式检索上限问题。

## 5. 解码公平性

### 约束不是“免费优化”

PrefixTrie 能把概率质量限制在有效目录，但比较时必须披露：

- trie 是用 train catalog 还是包含 test 新物品的 catalog 构建；
- beam width、每步可选 token 数和最终有效候选数；
- 是否把 target 强制加入候选/rollout；
- 碰撞发生在叶子前还是靠额外唯一 token 消解；
- KV cache、batch size 与最大生成长度。

给一种方法 trie、另一种方法自由生成，会同时改变准确率和计算预算。公平设置应同时给：相同 beam 宽度，以及相同最终 scored-item 数或近似 FLOPs/延迟预算。

### 分解生成瓶颈

[Understanding Semantic IDs](https://arxiv.org/abs/2607.24995)指出 SID 会保留粗粒度组织、损失部分 encoder 的细邻域结构，而且 autoregressive 前缀过滤会提前淘汰原本合理的 item。建议记录：

1. encoder nearest-neighbor quality；
2. tokenizer 重构误差和邻域保持率；
3. target prefix 在每一层是否仍在 beam 中；
4. 完整 SID 生成正确率；
5. 解析后 item rank。

这样才能区分“表示丢失”“搜索剪枝”和“碰撞解析”三类失败。

## 6. 文本与 LLM 生成模型

直接生成标题或描述时，应先冻结 catalog grounding 规则：Unicode/大小写规范化、标点、别名、版本/规格、同名 item 以及向量阈值。正式报告至少包括：

- exact/normalized string match；
- grounded item-level Recall/NDCG；
- `ungrounded_rate`、一对多 alias 率和 hallucination rate；
- grounding retriever 自身的 oracle recall；
- prompt、模型版本、量化和最大输出 token；
- 若使用外部 API，记录调用日期和采样参数。

BIGRec 一类“先生成语义再近邻检索”的方法，应把生成器和 grounding 检索器分开消融。NoteLLM 的 I2I embedding retrieval 不能拿辅助 hashtag/category 生成准确率代替推荐指标。

## 7. 扩散与流模型

扩散家族必须按输出空间分组：

| 输出 | 例子 | 主评测注意点 |
|---|---|---|
| 交互向量 | DiffRec | 对所有 item 的输出分数做全目录排序；隐藏已交互 item 的规则一致 |
| 连续目标 embedding | DiffuRec、DreamRec、PreferDiff | 冻结 item encoder/index；报告 ANN/精确近邻、top-M grounding oracle |
| 离散 item/SID | DDSR、MaskGR、MDGR | 仍需 item-level 碰撞纠正与有效路径统计 |
| 离散流 | FlowCF | 报步数、solver、先验与每步函数评估次数 |

必须报告采样/反演步数、噪声 schedule、guidance scale、每个用户的样本数及聚合方式。若从多个随机样本取“最好命中”的 oracle 结果，必须与实际可用的打分/选择器分开，不能当主结果。比较速度应同时给 wall-clock 和 network function evaluations。

## 8. RL、GRPO 与推理增强

RL 结果至少记录：

- SFT 起点的同协议 test/validation 数字；
- group size、每批 rollout 数、`add_gt`/强制正例开关；
- `has_exact_match` 比例、全零 reward 组比例、正 reward 数/组；
- exact、rank、hierarchical、格式和长度等奖励分量；
- reward 均值/方差/分位数及 advantage clipping；
- KL/reference coefficient、学习率、累计生成 token；
- R@1 与 R@10 的分别变化，而非只报最有利的一个 K。

过程奖励或 in-text reasoning 还应报告“移除可见推理文本”“打乱推理”“相同 test-time token 预算”等对照，以判断收益来自真实中间决策还是更多计算。PROMISE 一类 PRM-guided beam 要在相同 expansion 数下与普通 beam 比较。

## 9. 冷启动、规模与覆盖

至少按下列桶报告：

- item interaction count：0、1–5、6–20、head；
- item 是否在 tokenizer 训练中出现；
- 完整 SID、各 token 和各前缀是否在生成器训练中出现；
- 内容字段是否可用、是否在 encoder 预训练语料中可能出现；
- catalog size、SID 层数/宽度、模型参数和训练 token。

“新 item 能被编码”不等于“生成器能走到该路径”；“模型更大”也不保证 SID 瓶颈消失。规模实验必须分别扩 encoder、tokenizer、generator 和数据量，而不是只改 LLM 参数。

## 10. 可复现性清单

- 数据下载 URL、版本日期、文件 hash、许可和过滤规则；
- user/item 映射是否只由 train 构造；
- 文本 encoder 名称、revision、pooling、归一化和维度；
- tokenizer seed、层数、K、训练集范围、冲突处理、checkpoint hash；
- 模型 config、参数量、优化器、scheduler、有效 batch、精度；
- 最佳 checkpoint 选择规则、early stopping patience；
- 解码器所有参数和 catalog/trie hash；
- 三个以上 seed 的逐 seed JSON，不只保留均值；
- 失败 run 和 OOM 不应从统计中静默删除；
- 日志中明确区分 `sid_*`、`item_*`、`candidate_*`、`oracle_*`。

建议的最小结果键：

```text
dataset/split/protocol/seed/model/tokenizer
item_recall@{1,5,10,20}
item_ndcg@{5,10,20}
sid_hit@{5,10}                 # diagnostic only
collision_rate/max_bucket
invalid_rate/duplicate_rate/valid_unique_yield
decode_ms_per_user/train_gpu_hours/peak_memory_gb
```

在 P0 重评完成前，本仓库历史 TIGER、LC-Rec、OneRec 的 SID-tuple 数字只能作为旧记录；COBRA、SASRec、HSTU、RPG 已采用 item-level 输出，但仍要统一 split、seed 和计算预算后才能形成正式排行榜。
