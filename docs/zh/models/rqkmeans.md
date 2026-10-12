# RQ-KMeans：非神经残差量化器

## 定位与状态

| 项目 | 内容 |
|---|---|
| 任务 | 将 catalog item embedding 离散化为 semantic ID |
| 类型 | 多层 residual K-Means；不是推荐模型，也不是 `nn.Module` |
| 输入 / 输出 | `[N,D]` 浮点嵌入 → `[N,C]` 整数 code |
| 当前消费者 | OneRec SFT、CLM 与 GRPO 数据适配器 |
| 默认配置 | 3 层、每层 256 个 centroid、Qwen3-Embedding-8B |
| 评估 | code usage、最大簇频率、collision；不单独报告推荐 Recall |

RQ-KMeans 是上游 tokenizer。它没有用户历史输入，也不会输出 next-item 排名，因此不能与 SASRec、TIGER 或 OneRec 放在同一推荐效果表。当前实现状态见[模型实现状态](../reference/model-status.md)。

## 来源与使用边界

残差量化是经典向量量化思路：每层量化上一层残差，多个 code 联合近似原向量。当前类是本仓库的轻量工程实现，用来支撑 [OneRec](onerec.md) 管线；它不是某篇论文官方代码的逐行移植。OpenOneRec 与 MiniOneRec 的公开工作都讨论了生成式推荐中的 SID 构造，但本仓库的 encoder、$K$、dead-cluster 修复和拟合范围应按这里的代码理解，而不能从论文默认值反推。

与 [RQVAE](rqvae.md) 最大的区别是：RQ-KMeans 直接在冻结的外部 item embedding 上聚类，没有可训练 encoder/decoder、重构 loss 或 commitment loss。速度更快、状态更简单，但表示能力完全受外部 embedding 空间与 K-Means 假设约束。

## 核心算法

给定 $N$ 个 $D$ 维嵌入 $x_i$，令第一层残差 $r_i^{(0)}=x_i$。第 $\ell$ 层求解

$$
\{\mu_k^{(\ell)}\}_{k=1}^K
=\arg\min_{\{\mu_k\}}\sum_i
\min_k\left\|r_i^{(\ell)}-\mu_k\right\|_2^2,
$$

并分配

$$
c_i^{(\ell)}=\arg\min_k
\left\|r_i^{(\ell)}-\mu_k^{(\ell)}\right\|_2^2,
\qquad
r_i^{(\ell+1)}=r_i^{(\ell)}-\mu_{c_i^{(\ell)}}^{(\ell)}.
$$

最终 SID 为 $(c_i^{(0)},\ldots,c_i^{(C-1)})$，重构近似为各层 centroid 之和。`assign()` 用

$$
\|r-\mu\|^2=\|r\|^2+\|\mu\|^2-2r^\top\mu
$$

构造 `[N,K]` 距离矩阵后取 `argmin`，再逐层更新 residual。

### backend 选择

`RqKmeans(backend="auto")` 按 $N\times K$ 选择实现：

- 小于 $5\times10^7$：scikit-learn `KMeans`，k-means++ 初始化，默认 `n_init=10`。
- 否则：`MiniBatchKMeans`，batch 默认 4096，仍使用默认 10 次初始化。

trainer 没有把 `backend` 和 `n_init` 暴露为 Gin 参数；它只传层数、$K$、随机种子、最大迭代数和 batch size，所以常规命令总是走 `auto`。

### dead-cluster 重置

每层首次拟合后，实现统计没有任何 item 分配到的 centroid。若 `dead_reinit=True`：

1. 计算每个 item 到当前已分配 centroid 的平方误差；
2. 依误差从大到小选择 residual point，覆盖 dead centroid；
3. 用修补后的 centroid 重算完整 `[N,K]` 距离并重新分配。

代码注释称其为额外 Lloyd step；精确地说，当前实现只做一次全量 **assignment**，没有在重新分配后再计算簇均值。因此这是 dead centroid reseeding + reassignment，不是完整收敛的第二轮 K-Means。

## 数据流

```mermaid
flowchart LR
    A[Amazon item metadata] --> B[deterministic item text]
    B --> C[external text encoder]
    C --> D[item embedding cache]
    D --> E[layer 0 K-Means]
    E --> F[residual]
    F --> G[layer 1 ... C-1]
    G --> H[rqkmeans.pt]
    H --> I[assign every catalog item]
    I --> J[SID-aware recommender dataset]
```

[`AmazonItemDataset`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon.py) 以 title、price、sales rank、brand 和 categories 等 metadata 组装文本，并缓存 encoder 输出。trainer 以 `train_test_split="all"` 加载完整 catalog，而不是只量化训练交互出现过的 item。这能给所有候选生成 SID，但也是 transductive catalog tokenizer：即使使用 global-temporal interaction split，未来交互中的 item metadata 仍可能参与聚类。

embedding cache 文件名包含 encoder 路径 basename，避免不同 encoder 的常见混用；但 checkpoint 本身不记录 encoder 身份，资产 provenance 仍须由配置和 run metadata 保存。

## 代码映射

| 层次 | 文件 / 符号 | 职责 |
|---|---|---|
| 核心 | [`RqKmeans`](https://github.com/phonism/genrec/blob/main/genrec/models/rqkmeans.py) | `fit`、`assign`、保存/加载、collision 统计 |
| 输出类型 | `RqKmeansOutput` | `sem_ids: np.ndarray[N,C]` |
| 数据 | [`AmazonItemDataset`](https://github.com/phonism/genrec/blob/main/genrec/data/amazon.py) | 构造/缓存完整 catalog 的文本向量 |
| trainer | [`fit_rqkmeans`](https://github.com/phonism/genrec/blob/main/genrec/trainers/rqkmeans_trainer.py) | 读取 embedding、拟合、打印健康度、保存 |
| 默认配置 | [`rqkmeans.gin`](https://github.com/phonism/genrec/blob/main/config/onerec/amazon/rqkmeans.gin) | 3×256 与 Qwen3 embedding 路径 |
| 诊断 | [`verify_codebook_health.py`](https://github.com/phonism/genrec/blob/main/scripts/verify_codebook_health.py) | 每层 usage、dead codes、最大频率、collision |
| 诊断 | [`report_sid_collisions.py`](https://github.com/phonism/genrec/blob/main/scripts/report_sid_collisions.py) | prefix collision、组大小与多来源对比 |

`save()` 只写 `n_layers`、`codebook_size` 和 centroid 数组。`load()` 包含 NumPy 2.x pickle 名称的兼容别名，然后按 CPU 加载；它不会验证输入 embedding 维度以外的 encoder 语义一致性。

## 拟合与健康检查

```bash
# 默认 Amazon Beauty：Qwen3-Embedding-8B，3×256
python genrec/trainers/rqkmeans_trainer.py \
  config/onerec/amazon/rqkmeans.gin --split beauty

# 重新从同一 encoder 的 catalog embedding 分配并检查每层健康度
python scripts/verify_codebook_health.py \
  --kind rqkmeans \
  --checkpoint out/onerec/amazon/beauty/rqkmeans/rqkmeans.pt \
  --dataset-folder dataset/amazon \
  --split beauty \
  --encoder ./models_hub/Qwen3-Embedding-8B
```

可用 Gin 覆盖直接做 $K$ 或层数消融：

```bash
python genrec/trainers/rqkmeans_trainer.py \
  config/onerec/amazon/rqkmeans.gin --split sports \
  --gin 'fit_rqkmeans.codebook_size=128' \
  --gin 'fit_rqkmeans.n_layers=4' \
  --gin 'fit_rqkmeans.save_dir_root="out/onerec/amazon/{split}/rqkmeans_k128_c4"'
```

改变 $C$ 或 $K$ 后，下游 OneRec 的 dataset、模型 code token 数和 checkpoint 目录也必须同步；只覆盖拟合器会让 token 越界或把不同 code 空间误配。

### 最低健康报告

每次新 tokenizer 至少记录：

- 每层 `used_codes / K` 与 dead-code 数；
- 每层 `max_freq` 和覆盖 50% item 所需 code 数；
- 完整 SID 的 unique 数、collision rate 与最大 collision group；
- prefix depth 1..$C$ 的 collision rate；
- 最终 residual norm，以及 encoder、catalog 版本、seed 和拟合 backend。

仓库诊断脚本把 usage < 80% 或最大单簇频率 > 10% 标成 collapse signature；这是运维告警阈值，不是统计学定义或论文结论。

项目在 Amazon Beauty 约 1.2 万 item 上曾观察到 $K=8192$ 的浅 residual 层使用率下降和 super-cluster；改用 $K=256$ 后健康度明显改善。因此当前配置选择 256。该观察不能推出“所有小 catalog 必然在 8192 崩溃”，新数据必须实测。

## 输出与依赖

默认输出目录为 `out/onerec/amazon/<split>/rqkmeans/`：

- `rqkmeans.pt`：centroid checkpoint；
- `results.json`：collision、unique SID、总 item 数、最大碰撞组与 run config；
- stdout：逐层 residual norm、usage 和最大簇频率。

依赖 NumPy、PyTorch（仅用于序列化）、scikit-learn 和外部文本 encoder 所需的 Transformers/Sentence-Transformers 数据路径。核心聚类在 CPU/scikit-learn 上执行；GPU 可用并不会自动切换到 GPU K-Means。

## 与 RQVAE 及论文式 tokenizer 的差异

| 方面 | RQ-KMeans | RQVAE |
|---|---|---|
| 量化空间 | 外部原始 item embedding | learned encoder latent |
| 优化 | 每层独立 K-Means | 重构 + quantization/commitment 的梯度训练 |
| decoder | 无 | 有，可监控重构 |
| 层间训练 | 顺序拟合 residual | 联合前向与训练 |
| 状态 | centroid 数组 | 网络、codebook 与训练状态 |
| 失败模式 | dead/super clusters、外部 embedding 偏置 | codebook collapse、重构与离散目标权衡 |

本仓库实现也没有 OPQ 旋转、product subspace、端到端 recommendation loss 或可学习 codebook。RPG 使用的是另一条 FAISS OPQ 路径，不共享该 checkpoint。

## 已知限制与常见坑

1. **`K > N` 直接失败** → scikit-learn 无法拟合比样本更多的 cluster。先按 catalog 大小选择 $K$。
2. **内存突然升高** → `assign()` 和 dead reinit 都显式构造 `[N,K]` 距离；大 catalog 即使拟合用 MiniBatch，分配仍可能成为内存瓶颈。
3. **拟合耗时远超预期** → auto 进入 full K-Means 且 `n_init=10`。trainer 目前不能从 Gin 设 `n_init`/`backend`；需要扩展 `fit_rqkmeans` 接口，而不是误以为 `batch_size` 会强制 MiniBatch。
4. **加载成功但 SID 全变** → checkpoint 不记录 encoder；相同维度的不同 encoder 也能运行。核对原始 gin、embedding cache basename 和 run provenance。
5. **dead code 重置后仍有 dead code** → 一次 reassignment 不保证每个新 centroid 最终得到 item。以健康检查结果为准，不要仅凭 `dead_reinit=True` 宣称 100% usage。
6. **collision 很低却推荐不好** → 离散唯一性不等于语义可预测性；还要检查每层熵、前缀平衡与下游 item-level 指标。
7. **修改 tokenizer 后旧 OneRec checkpoint 还能加载** → token 数相同只代表形状相容，centroid 语义已经变化。必须重新训练或建立显式映射，禁止静默复用。
8. **把 tokenizer 指标当推荐指标** → collision/usage 只描述 ID 空间，不包含用户偏好学习。

## 扩展点

- 将 `backend`、`n_init`、`dead_reinit` 暴露到 trainer/Gin，并把它们写入 checkpoint；需保持旧 checkpoint load 兼容。
- 把距离计算改成 chunked 或 GPU 实现，测试结果需与 NumPy 全量 `argmin` 一致，尤其处理距离并列时。
- 在 dead reinit 后增加真正的 centroid update/多轮 Lloyd 迭代，并用固定 seed 比较 inertia、usage 和 collision。
- 支持 train-visible-only tokenizer 时，必须定义 unseen item 的 assignment、catalog cutoff 和 metadata 可见性。
- 为 checkpoint 增加 encoder identifier、embedding dimension、catalog fingerprint 和归一化设置；加载时 fail fast。
- 新增模型单测应覆盖保存/加载一致性、两层 residual 手算样例、$K>N$ 错误、dead cluster 与 collision 统计。

## 参考与相关文档

- [OneRec 管线](onerec.md)
- [RQVAE](rqvae.md)
- [模型总览与选型](overview.md)
- [MiniOneRec 论文](https://arxiv.org/abs/2510.24431)
- [OpenOneRec 官方仓库](https://github.com/Kuaishou-OneRec/OpenOneRec)
