# 术语与边界

生成式推荐领域经常把“生成”“检索”“排序”“语义 ID”混用。本页给出 GenRec 文档中的固定含义。

## 任务术语

| 术语 | 本项目中的含义 | 不应混同 |
|---|---|---|
| **序列推荐** | 根据按时间排序的交互历史预测下一 item | 任意静态 user–item 匹配 |
| **生成式推荐（广义）** | 以生成模型、生成目标或生成过程完成推荐的一类方法 | 只因为 backbone 是 Transformer 就称“生成式” |
| **生成式检索（狭义）** | 自回归或非自回归地产生可映射到目录 item 的标识符/token 序列 | 生成自然语言理由但仍由传统召回给候选 |
| **判别式全排序** | 对目录中的 item 直接产生分数并取 Top-K | sampled-negative 指标 |
| **next-item prediction** | 每个测试样本只有一个目标 item 的下一项预测 | list continuation、多目标 slate generation |
| **foundation / LLM recommender** | 以大规模预训练模型跨任务或跨域建模推荐 | 参数量大但只在单数据集训练的模型 |

## 表示与量化

| 术语 | 定义 |
|---|---|
| **item ID / PID** | 目录中唯一对应真实物品的标识符；0 通常保留为 padding |
| **Semantic ID（SID）** | 由一个或多个离散 code 组成的 item 表示，例如 `(c_1,c_2,c_3)` |
| **codebook** | 把连续向量映射为离散索引的原型集合 |
| **RQ（Residual Quantization）** | 第 1 层量化原向量，后续层依次量化前一层残差 |
| **RQ-VAE** | 用神经 encoder/decoder 与残差向量量化联合学习 SID 的组件 |
| **RQ-KMeans** | 对输入及逐层残差执行 K-Means 的非神经量化器 |
| **OPQ** | Optimized Product Quantization；旋转后把向量切成子空间并分别量化 |
| **collision** | 两个或更多 item 具有同一完整 SID |
| **dead code / dead cluster** | 某层 codebook 中没有 item 被分配到的 code |
| **codebook usage** | 至少被使用一次的 code 数占码本容量的比例；高 usage 不保证完整 SID 唯一 |

## 解码与评测

| 术语 | 定义 |
|---|---|
| **constrained decoding** | 用 Trie、合法 token 集或 catalog 规则限制每一步生成，只允许形成有效标识符 |
| **beam search** | 保留累计分数最高的若干部分序列；beam width 不是最终 `K` 的同义词 |
| **SID-level metric** | 预测 SID 与目标 SID 元组相等即命中；存在碰撞时会把同桶其他 item 误计为目标 |
| **item-level metric** | 先将候选解析到唯一真实 item，再计算 Recall/NDCG；这是 GenRec 的 headline 口径 |
| **training-only resolver** | 只用训练交互的流行度为碰撞 SID 选择代表 item，防止 valid/test 目标泄漏 |
| **item deduplication** | 多个生成候选解析到同一 item 时只保留分数最高/顺序最前的一个，并从更深 beam 补足 |
| **full catalog** | 每次预测面对完整可推荐目录；可以直接全打分，也可以用受约束生成近似 Top-K |
| **in-batch metric** | 只在当前 batch 的 item 中排序；只能诊断，通常远易于 full catalog |
| **leave-one-out（LOO）** | 每位用户最后一次交互作 test、倒数第二次作 valid，之前作 train |
| **global temporal** | 用全局时间切点分 train/valid/test，降低未来交互进入训练历史的风险 |

## 指标

本项目的 next-item 测试每个样本只有一个相关 item。于是：

\[
\operatorname{Recall@K}(u)=\mathbb{1}[r_u\le K]
\]

\[
\operatorname{NDCG@K}(u)=
\begin{cases}
1/\log_2(r_u+1), & r_u\le K \\
0, & r_u>K
\end{cases}
\]

其中 \(r_u\) 是目标 item 在去重后候选列表中的 1-based rank。最终指标对测试样本取平均。

!!! note "Recall 与 Hit Rate"
    在“每个用户只有一个测试目标”的协议里，Recall@K 与 HR@K 数值相同；文档仍统一写 Recall@K，以免切换到多目标任务时概念失真。

## “实现”与“复现”

- **实现**：存在明确的模型、数据、训练与评测代码路径。
- **对齐**：关键表示、目标函数、解码和数据处理与论文设定有证据对应。
- **复现**：在明确硬件、数据和评测协议下得到与原文可解释地一致的结果。
- **复现失败**：完成了可比实验但差距显著；应保留诊断证据，而不是删除条目。
- **实验性**：能运行但证据不足；不能进入 headline 比较。

