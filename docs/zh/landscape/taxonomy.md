# 生成式推荐分类框架

“生成式推荐”不是单一模型类别。判断两个方法能否公平比较，至少要同时看它生成什么、如何分解概率、怎样绑定目录物品，以及训练信号来自哪里。

## 一条统一流水线

对最常见的语义 ID（SID）路线，可写成：

\[
x_i \xrightarrow{E} z_i
\xrightarrow{Q} s_i=(c_{i,1},\ldots,c_{i,L})
\xrightarrow{G_\theta} p(s_{t+1}\mid s_{\le t})
\xrightarrow{D} \hat i .
\]

- \(E\)：把标题、描述、图像或协同信号编码成连续表示；
- \(Q\)：RQ-VAE、残差 K-Means、OPQ、FSQ、BPE/树量化等离散化；
- \(G_\theta\)：自回归、并行多头、掩码扩散或其他生成器；
- \(D\)：trie/grammar 约束、beam search、碰撞解析、密集重排或图传播，把输出绑定回真实 item ID。

这四段任一变化都可能改变结果。只说“模型都是 T5”或“都用了相同 SID 长度”，仍不足以构成公平比较。[GRID 手册](https://arxiv.org/abs/2507.22224)把 tokenizer、模型、训练和解码作为一个整体实验系统，正适合作为本仓库的协议参照。

## 维度一：生成到哪个空间

| 输出空间 | 典型工作 | 优点 | 主要风险 |
|---|---|---|---|
| 原子 item ID | HSTU、传统 next-item LM | 无碰撞，指标语义清楚 | 词表/输出层随目录增长；新物品困难 |
| 多 token SID | TIGER、LC-Rec、GRAM、OneRec、RPG | 共享 token，可压缩大目录并利用内容语义 | 碰撞、无效路径；自回归 SID 还会有前缀流行度偏置（RPG 不依赖 AR 前缀） |
| 自然语言/标题 | P5、GenRec、BIGRec | 利用 LLM 知识，可解释、可跨域 | 幻觉、别名和目录落地不唯一 |
| 连续“理想物品”向量 | DreamRec、DiffuRec、COBRA 的 dense 尾部 | 可表达不确定性或细粒度残差 | 最近邻检索仍是判别式绑定；索引成本不可忽略 |
| 交互向量/集合 | DiffRec、FlowCF | 一次建模全目录偏好分布 | 输出维度随目录增长，不是开放词表生成 |
| 有序列表/页面 | NAR4Rec、GReF、工业 OneRec/京东 GenRec | 直接优化列表结构、去重和页面目标 | 与 next-item 指标不是同一个任务 |
| 候选 action/生成式排序 | GenRank | 复用序列生成架构建模候选行为 | 输出候选 action 概率，不等于直接生成 item 列表 |
| 文本推理 + 物品 token | OneRec-Think | 可引入显式推理和推荐专用 RL | 推理真实性、延迟和奖励投机 |
| SID 过程评分/测试时搜索 | PROMISE | 定位失败层并利用额外推理预算 | PRM 校准、搜索预算和 reward 投机 |

## 维度二：怎样分解生成过程

### 自回归

按 token 逐步分解 \(p(s)=\prod_l p(c_l\mid c_{<l},h)\)。TIGER、LC-Rec、GRAM 和大多数 LLM-SID 方法属于此类。优点是可直接复用语言模型和 trie 约束；缺点是误差累积、beam 开销与长 SID 延迟。

### 并行或非自回归

RPG 将 OPQ 子空间 code 组成论文所称的 unordered long SID；各 code 仍对应固定 codebook 位置，但由独立 head 并行预测，组合分数不依赖自回归前缀，随后可做图传播。ActionPiece 改变 action token 的粒度；MaskGR/MDGR 通过迭代去掩码并行恢复 SID。这里“并行”不等于“一步完成”：掩码扩散仍可能需要多个反向步。

### 连续扩散与流

DiffuRec、DreamRec 对目标 item embedding 加噪并反演；DiffRec 对用户—物品交互向量去噪；FlowCF 用离散流匹配连接先验和数据分布。它们输出对象不同，不能只因都叫 diffusion 就共用同一评测解释。

### 混合检索

LIGER 用生成式候选覆盖和 dense retrieval 的物品分辨率互补；COBRA 先生成若干离散码，再生成连续尾向量；SpecGR 用可扩展 drafter 提案、生成式 verifier 验证。混合系统应分别报告候选召回、重排增益和总延迟。

## 维度三：ID/tokenizer 从哪里来

| 来源 | 方法 | 特性 |
|---|---|---|
| 随机/原子 ID | 原始 ID、随机字符串 | 可保证唯一，但没有可迁移语义 |
| 顺序/协同 ID | P5 indexing、协同聚类 | 编码行为邻近关系，容易受划分和热门度影响 |
| 内容语义量化 | TIGER RQ-VAE、RecGPT FSQ | 新物品可从内容获得码，但不自动保留协同偏好 |
| 语义 + 协同联合 | LETTER、ETEGRec、UNGER（早期版本 PRORec）、GRAM | 减少内容与行为目标错位，训练更复杂 |
| 上下文相关 ID | Pctx、ActionPiece | 同一物品可有多个视角/粒度；解析需概率聚合 |
| 可变长/原生词表 | Variable-Length SID、VaLiDRec | 把热门度或语义复杂度映射到码长；约束解码更复杂 |
| 量化后唯一化 | 冲突后缀、ECM/RRS、QuaSID | 保证目录可寻址；后缀是否有语义、是否泄漏要单独检查 |

Tokenizer 不是中性的预处理。码本大小、层数、训练数据、去重后缀以及冻结/联训策略都会改变推荐器面对的任务难度。

## 维度四：目录绑定方式

生成结果必须落到可消费的 item ID：

1. **唯一 SID**：生成码唯一对应物品；最清楚，但通常需要冲突处理。
2. **碰撞桶解析**：一个 SID 对应多个物品，再用训练集热门度、额外打分器或向量距离选一个。若测试标签参与解析就发生泄漏。
3. **受约束生成**：PrefixTrie/有限状态约束只允许已有目录路径；能消除无效码，不能自动消除叶子碰撞。
4. **文本 grounding**：标题/描述经过规范化、别名表或向量召回映射到目录；必须报告无法落地和一对多比例。
5. **dense/graph 后处理**：LIGER、COBRA、RPG 等把生成结果送入密集重排或图传播；后处理预算也属于模型。

## 维度五：训练与后训练

| 训练信号 | 代表 | 需要单独记录的变量 |
|---|---|---|
| next-token / teacher forcing | TIGER、LC-Rec、MiniOneRec SFT | task mix、序列截断、label smoothing、曝光偏差 |
| 多位置生成式训练 | HSTU、MTGR | causal mask、目标位置数、负采样或全 softmax |
| 去噪/score matching | DiffRec、DreamRec、MaskGR | 噪声调度、采样步数、guidance、随机种子 |
| 偏好优化/DPO | OneRec V1、Align³GR、GReF | 正负构造、reference policy、列表级目标 |
| RL/GRPO 类 | MiniOneRec、OpenOneRec、Rec-R1、DASO | group size、约束解码、全零组、奖励尺度与 KL |
| 文本推理/推荐 RL | OneRec-Think | reasoning scaffold、可见推理长度、reward |
| SID 过程监督/测试时搜索 | PROMISE | step reward、搜索宽度、expansion/时间预算 |

## 六条主谱系与两条横向能力

1. **语义 ID 生成式检索**：TIGER → LC-Rec/GRAM → OneRec/MiniOneRec/OpenOneRec → 非自回归和混合方法。
2. **LLM 文本生成推荐**：P5 → GenRec/BIGRec/IDGenRec → 文本基础模型与目录 grounding。
3. **生成式序列建模/基础模型**：HSTU、RecGPT、MTGR、PLUM 以及工业规模系统。
4. **扩散/流推荐**：交互向量、连续目标物品、离散 SID 三种不同子支。
5. **列表/页面生成与生成式排序**：直接列表生成、action-oriented candidate ranking 和 session/page 级生成；三者的输出接口不能混用。
6. **RL 与推理增强**：DPO/GRPO、过程奖励、in-text reasoning 与测试时扩展。
7. **横向：tokenizer/量化**：决定可寻址性、语义共享、码本健康和冷物品能力。
8. **横向：评测/碰撞**：决定论文数字是否真的代表 item-level 推荐质量。

完整代表作和状态见[模型目录](model-catalog.md)，协议见[评测文档](evaluation.md)。
