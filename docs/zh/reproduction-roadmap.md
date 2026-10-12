# GenRec 复现行动计划（2026-06）

!!! warning "带日期的历史研究记录"

    本页保留 2026-06 实验决策、阶段性数字与当时的实现判断，便于追溯研究过程；它不是当前代码手册，其中个别“已有/待实现”、脚本名和解码能力可能已变化。当前模型状态以[实现与复现状态](reference/model-status.md)为准，下一步优先级以[实现待办](landscape/implementation-backlog.md)为准，训练命令以[命令索引](reference/command-index.md)为准。未经对应 checkpoint、日志和数据指纹复核，不要把本页数字直接作为对外 headline。

> 本文是**可执行的复现路线图**，不是领域综述（综述见 [`latest-advances.md`](latest-advances.md)）。
> 每个候选都给出：做什么 / 为什么 / 学术规模可复现性 / 与现有代码接口点 / 验收标准 / 工作量。
> 数字凡未亲自核实的标注「原文称」或「待核实」。
>
> **口径更新（2026-08）：**本文 2026-06 的 `raw/A1` 数字是历史 SID-tuple
> 口径，只用于展示碰撞膨胀；当前 TIGER/LCRec/OneRec trainer 的主指标、
> 早停与结果文件均已切换为 training-only most-popular SID→item、排序去重后的
> item-level Recall/NDCG。README 主表在 checkpoint 按新代码重评前不展示旧分数。

---

## 0. 定位与既有结论（决策前提）

GenRec 的 moat = **fair-comparison + 可复现 + 清晰叙事（RecBole 学派），不追工业 SOTA**。
本计划据此硬筛，优先「能补一条干净对照轴 / 强化诚实对比叙事 / 学术规模单机 4–8×A100 可跑」的工作。

**本 session 已确立、可直接复用的结论与资产：**

1. **RL（GRPO）对推荐是边际收益，非显著。** Beauty / Qwen3-1.7B 上跑满最扎实配置（#51：g16 + rank reward + add_gt，2 epoch）：

   | Beauty Test | R@1 | R@5 | R@10 | N@10 |
   |---|---|---|---|---|
   | SFT baseline (v2b) | 0.0186 | 0.0596 | 0.0912 | 0.0493 |
   | #51 GRPO (best=epoch0) | 0.0190 | 0.0594 | 0.0919 | 0.0499 |
   | Δ | +2.2% | −0.3% | **+0.8%** | +1.2% |

   epoch0 即见顶，epoch1 略退。与 OpenOneRec Table 12（RL vs SFT 仅 +0.8~14% rec、−1~2% understanding，已亲自读 PDF 核实）同档。

2. **「显著收益」很大程度是分母造的。** OpenOneRec 全 foundation model 在 Beauty 的 R@10 = **0.0924**（Table 7，已核实）≈ 我们 1.7B 纯 SFT 的 **0.0925**。他们「+26.8% 平均提升」是 over 弱 baseline（其 SASRec Beauty R@10=0.0639，我们的 SASRec=0.0851）。**同等模型质量，分母不同 → 「显著」vs「边际」。** 真正的发动机是大规模推荐域预训练（96M 交互、32–130B token）+ scaling，非 GRPO。

3. **两篇旗舰复现都回避隔离的 SFT→RL 数字**（MiniOneRec 只有柱状消融；OpenOneRec 把功劳算进全 pipeline）。**公开隔离增益矩阵正是社区缺的诚实数字 → GenRec 的机会。**

4. **多卡 deadlock 根因已修并二次验证**（`has_cached_test` 跨 rank 一致，非 best epoch 正确 skip test，无 hang）。已落 4 个 trainer（sft/grpo/clm/lcrec）。

5. **资产**：本 session 已测 TIGER 3-code 碰撞率 4.47% / 4-code 0.72%（honest-protocol 对照表已完成）；v2b SFT 抗过拟合配方（Finding 2）；K=256 + dead-cluster reinit 抗坍塌（Finding 1）。

---

## 1. 总体策略：建一个正交对照矩阵（而非再加一个模型）

```
tokenizer 轴 :  RQ-VAE  /  RQ-KMeans  /  LETTER  /  ActionPiece
解码    轴 :  自回归(TIGER/OneRec)  /  并行(RPG)  /  diffusion(DiffGRM)
训练    轴 :  SFT-only  /  +GRPO  /  +S-DPO  /  +Rollout-Beam reward
评测    轴 :  LOO(现有)  /  去泄漏 temporal split  +  碰撞率/幻觉率
```

GenRec 的差异化产出 = 把这四轴交叉测出**隔离、诚实、可复现**的对照，而不是某一格的 SOTA。

---

## 2. 分层计划

图例：现状 ✅已有 / 🟡部分 / ❌无；工作量 S(<3天) M(1–2周) L(2–4周)。

### 🟢 Tier 0 — 零/低训练成本，最强 moat，部分已有数据打底

#### T0-1. SID 生成方法学验证（Practitioner's Handbook / GRID）
- **做什么**：在现有模型上系统复现 GRID 的核心消融结论：① enc-dec > decoder-only；② Residual-KMeans ≈/优于 RQ-VAE；③ trie 约束非必需（无约束 beam 不掉点且更便宜）；④ user-token 有害。
- **为什么**：纯方法学、不加新模型，直接验证 Finding 1（K-Means 更稳），是「公平对照」立身之本。
- **接口点**：`genrec/models/{rqvae,rqkmeans,tiger}.py` + `tiger_trainer` 的 trie-beam（已有 ✅🟡）。
- **代码**：[GRID](https://github.com/snap-research/GRID)（有官方代码）｜[arXiv 2507.22224](https://arxiv.org/abs/2507.22224)（CIKM'25）
- **验收**：一张「设计选择 × R@10/N@10」消融表，复现/反驳上述 4 条结论各一行。
- **工作量**：M

#### T0-2. 去泄漏评测协议（第二评测轴）— 🟡 代码完成，矩阵待重跑
- **做什么**：在现有 leave-one-out 之外，**并行报告 global temporal split**；量化 temporal LOO 的泄漏水分。
- **为什么**：RecSys'25 两篇证明 temporal LOO 把 seq-rec sampled nDCG **夸大 21.7–73.4%（原文称）**、并会翻转模型排名。把「别人不敢测的泄漏」测给同行 = moat 最有冲击力的延伸。
- **接口点**：`genrec/data/evaluation_protocol.py` 提供统一 `global_temporal` 切分，Amazon 侧 SASRec/HSTU/TIGER/LCRec/OneRec/COBRA/RPG 数据集均已接入；默认仍为 LOO 以保持旧实验可复现。全模型 temporal 对照矩阵尚待重跑。
- **代码**：[Time to Split, arXiv 2507.16289](https://arxiv.org/abs/2507.16289)（有代码）；[Don't Get Ahead of Yourself, RecSys'25](https://dl.acm.org/doi/10.1145/3705328.3759329)
- **验收**：所有现有模型在 LOO 与 temporal-split 两套协议下的对照表 + 排名是否翻转的结论。
- **工作量**：M

#### T0-3. SID 碰撞率 / unique-SID 报告项（生成式专属诚实性指标）— ✅ 完成（真实数据已跑）
- **工具**：`scripts/report_sid_collisions.py`（纯 NumPy、零侵入、读 [N,L] SID 数组或 token 串，含 per-prefix 逐层碰撞 / 容量利用率 / 组大小直方图 / 多源 Markdown 表 / JSON）+ `scripts/dump_sem_ids.py`（从 parquet emb + tokenizer dump SID，无需 encoder）；单测 `tests/test_sid_collisions.py`（14 项全过）。
- **真实结果（2026-06，beauty/sports/toys/home，全库真实 item）**：

  RQ-KMeans k256（OneRec tokenizer，3 层）跨类目碰撞：

  | split | n_items | 碰撞率 | 碰撞 item 占比 | 最大碰撞组 |
  |---|---|---|---|---|
  | beauty | 12101 | 17.25% | 28.7% | 15 |
  | sports | 18357 | 26.82% | 41.9% | 19 |
  | toys | 11924 | 16.96% | 27.8% | 15 |
  | home | 28237 | 27.32% | 41.9% | 41 |

  同 beauty、同 k256 容量的 tokenizer-family 对照：

  | tokenizer | 碰撞率 | 碰撞 item 占比 |
  |---|---|---|
  | RQ-KMeans 3L (OneRec) | **17.25%** | 28.7% |
  | RQ-VAE 3L (TIGER, Sinkhorn) | **3.69%** | 6.6% |
  | RQ-VAE 4L (TIGER) | **0.23%** | 0.45% |

- **结论**：① 同容量下 RQ-KMeans 碰撞是神经 RQ-VAE 的 ~4.7×；② 2026-06 的旧 OneRec eval 按 `target_sem_ids` 的 SID-tuple 精确匹配、未做 unique-item 区分，因此报告 Recall 存在结构性膨胀；**该实现现已修复**，SID 命中只保留为诊断项。③ per-layer 码利用率健康（Finding 1）≠ 整 SID 唯一：RQ-KMeans 每层 100% 利用但整 SID 仍碰 17%。
- **诚实 Recall：4 数据集实测折算（2026-06，复刻 OpenOneRec pid-mode）**：用真实 test 目标 + item 流行度，对每 split 算两种诚实口径——`1/M`（结构精确，组内均匀，= OpenOneRec random-pid 期望）与 `most_popular`（OpenOneRec pid-mode 默认：SID→桶内最热 item，命中需 target 即最热）。`scripts/honest_recall_pid.py` 直接挂真实 `AmazonOneRecSFTDataset` 读 `sem_ids_list`+`sequences`，零对齐假设。

  | split | raw R@10(报告) | retention(1/M) | honest 1/M | honest most_pop | SASRec | HSTU | raw 结论 | **honest 结论** |
  |---|---|---|---|---|---|---|---|---|
  | beauty | 0.0925 | 0.808 | 0.0748 | 0.0788 | 0.0851 | **0.0859** | 生成 +7.7% | **输 8–13%** |
  | toys   | 0.0946 | 0.811 | 0.0767 | 0.0807 | **0.0922** | 0.0914 | 生成 +2.6% | **输 12–17%** |
  | sports | 0.0596 | 0.723 | 0.0431 | 0.0473 | **0.0495** | 0.0439 | 生成 +20% | **输 4–13%** |
  | home   | 0.0348 | 0.696 | 0.0242 | 0.0275 | **0.0277** | 0.0208 | 生成 +26% | **≈ 打平 SASRec** |

  - **杀手级推论（升级版）**：碰撞膨胀**翻转了排名**——raw 下生成式在全部 4 个数据集都赢判别式（+2.6%~+26%）；honest 下 beauty/toys/sports 全输、home 打平。**「生成式打败判别式」本身就是 SID 碰撞造的 artifact。** 判别式预测唯一 item id、零碰撞膨胀，故其数诚实；拉到同一诚实标准后生成式优势消失。
  - **capacity 缩放规律**：膨胀随 catalog 增大加重——~12k items(beauty/toys) 碰撞~17%/膨胀~19%；18–28k items(sports/home) 碰撞~27%/膨胀~28–30%。固定 K=256³ 容量被大 catalog 挤塌，直接咬合 Finding 1（高 K codebook collapse）——同一枚硬币两面。
  - **外部铁证（OpenOneRec 官方 eval 自证）**：clone `Kuaishou-OneRec/OpenOneRec`，其 benchmark（`benchmarks/.../recommendation/{evaluator,utils,utils_by_pid}.py`）默认 `evaluation_mode:"both"`，**同时算 sid 模式（碰撞膨胀）与 pid 模式（SID→`sid2pid` 的 most_popular 真实 item）**。即 SOTA lab 在代码里自承「SID-tuple 命中 ≠ item 命中」并内置消歧，只是论文 headline 报 sid。我们 `honest_recall_pid.py` 即复刻其 pid-mode（其 pid all-or-nothing 给最热，比我们 1/M 更狠）。
  - **假设/边界**：1/M、most_popular 均为结构折算（假设命中率与 M 无关）；为点估计。精确化见下「严格化协议」（模型侧 generate-once 直接测量 + CI）。

##### T0-3 严格化协议（2026-06 议定 · Tier 0：零新训练 · 全 4 数据集 · 无多 seed）

目标：把上表的「解析折算」升级为「直接测量 + 95% CI + 配对显著性」，使结论无法被攻击。

- **统一 harness（generate-once → eval-many）**：每个 (模型,split) 只跑一次生成，dump 每 test user 的 top-N(=128) 排序 SID + target；离线评估器对同一份预测套所有口径。`scripts/dump_predictions.py` + `scripts/sid_eval_suite.py`（本地，不提交）。
- **评估口径（同一份预测）**：A1 SID-tuple（上界/oracle）｜B1 most_popular pid（OpenOneRec）｜B2 random=1/M 期望｜B3 1/M 部分计分｜DEP item-dedup@top-N（可部署）｜LB 下界。报 bracket 并证收敛。
- **模型范围（本轮议定）**：OneRec-SFT（RQ-KMeans，现成 4 split）+ 现成 RQ-VAE-4L（beauty 免费第二 tokenizer 点）+ 判别式锚点 SASRec/HSTU（构造即诚实）。**不引入新 SID 模型、不做多 seed**。
- **统计**：每个 Recall 带 bootstrap 95% CI（重采样 user，B=10k）；ΔRecall(生成−判别) 配对 bootstrap p 值，逐 split×口径。
- **rigor 细节**：beam N≥200 防截断（堵「beam 太窄」反驳）；核实 SASRec/HSTU 与 OneRec 用同一 LOO 测试集（同 N）；max_seq_len=50 / 同 K / target=seq[-1] 对齐。
- **已知缺口**：home 无 SASRec/HSTU ckpt（需 ~1–2h 重训或暂用 README 点估计无 CI）。
- **未来层（暂不做）**：tokenizer 重训扫（Sinkhorn / disambig on Qwen8B）、TIGER 跨模型、3 seed → 见 Tier 1/2。

##### T0-3 实测结果（generate-once → eval-many，2026-06-05 · 已完成）

**Harness**：`scripts/dump_predictions.py`（OneRec v2b checkpoint，单进程 HF beam=10，无 NCCL → 不会触发多卡 watchdog 崩溃；A1 与 trainer 的 `TopKAccumulator` 逐位 exact-match 完全一致）→ `scripts/sid_eval_suite.py`（逐用户 A1命中 × 因子；bootstrap 95% CI，B=10k；配对 bootstrap p vs 判别式）。

**判别式 baseline 已重训到收敛**：仓库里旧 `best_model.pt` 是**欠训练**的（SASRec beauty 仅 10-epoch / ~2min，test 0.053），README 的 0.0851 跟它对不上。重训（文档 gin 配置）后 SASRec test R@10 = **0.0864 / 0.0497 / 0.0925 / 0.0275**（beauty/sports/toys/home）≈ README 0.0851/0.0495/0.0922/0.0277 → **README baseline 是对的，旧 checkpoint 被 debug 短跑覆盖**。HSTU 重训收敛 0.0701/0.0371/0.0696（略低于其 README 值，但 SASRec 是更强 baseline 且完美复现，结论靠 SASRec 即成立）。

| split | raw A1 (95%CI) | honest most_pop | honest 1/M | 收敛 SASRec (95%CI) | HSTU | Δ(mp−SAS) 配对 p |
|---|---|---|---|---|---|---|
| beauty | 0.0954 [.092,.099] | **0.0688** [.066,.072] | 0.0612 | 0.0864 [.083,.090] | 0.0701 | −0.0176 *** |
| toys   | 0.0955 [.091,.100] | **0.0602** [.057,.064] | 0.0540 | 0.0925 [.089,.097] | 0.0696 | −0.0323 *** |
| sports | 0.0612 [.059,.064] | **0.0362** [.034,.038] | 0.0271 | 0.0497 [.047,.052] | 0.0371 | −0.0135 *** |
| home   | 0.0356 [.034,.037] | **0.0210** [.020,.022] | 0.0153 | 0.0275 [.026,.029] | (训练中) | −0.0065 *** |

> *** = 配对 bootstrap p < 0.001（实为 p≈0，B=10k 无一次反向）。raw 实测 ≈ 报告值（0.0954/0.0955/0.0612/0.0356 vs 0.0925/0.0946/0.0596/0.0348）→ harness 忠实。

**五条铁结论**：
1. **honest 在全部 4 split 显著低于收敛 SASRec**（配对 p<0.001，CI 不重叠：honest mp 上界全部 < SASRec 下界）。
2. **排名翻转钉死**：raw 下 OneRec ≥ SASRec（+3%~+23%），honest 下 OneRec **−20%~−35%**。"生成式打败判别式" = SID 碰撞 artifact。
3. **实测膨胀 28–43%，比解析估计（15–19%）更狠**——因模型**优先命中密集碰撞 SID**（命中样本 M 偏大，1/M 折算更重）。即解析 retention 低估了真实膨胀。
4. **baseline 重训复现 README** → 杜绝"baseline 欠训练所以你赢"的反驳；且揭露仓库旧 checkpoint 欠训练这一数据卫生问题。
5. honest OneRec 甚至 ≈ 或低于**更弱**的 HSTU → 生成式在学术规模、诚实评测下不优于判别式。

工具与产物（本地，不提交）：`dump_predictions.py` / `sid_eval_suite.py` / `dump_disc_predictions.py` / `honest_recall_pid.py`；预测与 per-user dump 在 `out/_pred_dumps/`；重训收敛 baseline 在 `out/{sasrec,hstu}/amazon/{split}_conv/`。

##### T0-3 tokenizer 全扫（2026-06-05 启动 · 8 卡并行 · 2026-06-09 全部完成）

设计：固定 Qwen3-8B embedding（dim=4096）+ v2b SFT recipe（lr 8e-5 / wd 0.05 / attn_dropout 0.1 / 无 LS / patience 2 / 8 epochs），**只换量化器**：RQ-KMeans K=256×3L（原 OneRec）vs 新训 RQ-VAE 3L vs RQ-VAE 4L（每 split 各训 2000 epoch tokenizer + 完整 SFT，单卡单进程）。generate-once → eval-many，beam=10，判别锚点 = 收敛 SASRec。

**R@10（test，bootstrap 95% CI 略，p 均 ≈0 ***）：**

| split (SASRec) | tokenizer | 碰撞率 | A1 raw | B1 mp | B3 1/M | LB |
|---|---|---|---|---|---|---|
| beauty (0.0864) | RQ-KMeans 3L | .1725 | 0.0954 | 0.0688 | 0.0612 | 0.0437 |
| | RQ-VAE 3L | .2005 | 0.0953 | 0.0615 | 0.0509 | 0.0321 |
| | RQ-VAE 4L | .0645 | 0.0765 | 0.0661 | **0.0634** | 0.0550 |
| toys (0.0925) | RQ-KMeans 3L | — | 0.0955 | 0.0602 | 0.0540 | — |
| | RQ-VAE 3L | .1983 | 0.0985 | 0.0576 | 0.0515 | 0.0333 |
| | RQ-VAE 4L | .0741 | 0.0807 | **0.0649** | **0.0623** | 0.0515 |
| sports (0.0497) | RQ-KMeans 3L | — | 0.0612 | 0.0362 | 0.0271 | — |
| | RQ-VAE 3L | .3581 | 0.0744 | 0.0344 | 0.0214 | 0.0090 |
| | RQ-VAE 4L | .1007 | 0.0475 | **0.0377** | **0.0333** | 0.0261 |
| home (0.0275) | RQ-KMeans 3L | .273 | 0.0356 | 0.0210 | 0.0153 | 0.0078 |
| | RQ-VAE 3L | .369 | 0.0407 | 0.0186 | 0.0122 | 0.0055 |
| | RQ-VAE 4L | .087 | 0.0232 | **0.0194** | **0.0178** | 0.0145 |

> home RQ-KMeans 行已用完整 dump（`home_b10.json` 14MB）复跑确认（task #60，带 95% CI + 配对 p=0 ***）；早先 `eval_home.log` 的 JSONDecodeError 是在 dump 写入中途读取的误报，数值本身无误。

**最终结论（全 4 split × 3 tokenizer = 12 个生成式格子全部完成）：**
1. **raw 与 honest 反向移动 = 碰撞 artifact 的同族对照实锤**：同模型同 recipe 只换码本，3L→4L 碰撞大幅下降，A1 raw **降**（beauty −13% / toys −18% / sports −36% / home −43%），honest B3 **升**（beauty +25% / toys +21% / sports +56% / home +46%）。4/4 split 方向完全一致——raw 排名几乎完全由碰撞率主导，而非模型质量。
2. **碰撞越重膨胀越大**：sports RQ-VAE-3L（碰撞 .358）A1/B3 = 3.5×、home RQ-VAE-3L（.369）= 3.3×；低碰撞 4L 普遍降到 1.2~1.4×。
3. **没有任何 tokenizer 在任何 split 让 honest 生成式 ≥ 收敛 SASRec**：12 个格子的最佳 honest B1 全部低于对应 SASRec，差距 −0.0081（home 4L）~ −0.0348（toys RQ-KMeans），**全部配对 p≈0 ***，CI 不重叠**。这是本扫描要回答的核心问题——答案是**否**。
4. tokenizer 间无一致赢家（beauty: RQ-KMeans B1 最优 0.0688；toys/sports/home: RQ-VAE-4L 最优），但「降碰撞 → 提 honest」方向在 **4/4 split 一致**。
5. LB（无碰撞下界）4L 全面大幅高于 3L（beauty .032→.055 / sports .009→.026 / home .0055→.0145）→ 低碰撞 tokenizer 的可信收益真实存在（最高近 3×），只是绝对值仍不足以追上判别式。

运行与产物：SFT ckpt `out/onerec/amazon/{split}/sft_rqvae_q8b_{3,4}L/`；评估日志 `out/_pred_dumps/eval_{split}_rqvae{3,4}L.log` + home RQ-KMeans `eval_home_rqkmeans.log`；自动评估链 = gpu-remote task #59（按格等待→dump→eval，2026-06-09 打印 `ALL RQVAE EVALS DONE` 收官）+ task #60（home RQ-KMeans 补测）。
- **做什么**：离线免训练算每个 tokenizer 的「碰撞率 / unique-SID 率」，作为标准报告项，揭示 SID 碰撞对 Recall 的系统性膨胀。
- **为什么**：生成式专属的指标膨胀来源（多 item 落同一 SID → beam 命中被高估），全领域几乎没人报。**半成品**：本 session 已有 TIGER 3-code 4.47% / 4-code 0.72% 数据。
- **接口点**：`scripts/verify_codebook_health.py`（已有 ✅，扩成通用报告项）。
- **代码/依据**：[QuaSID, arXiv 2603.00632](https://arxiv.org/abs/2603.00632)（免训练 SID 质量指标）｜[FORGE 2509.20904](https://arxiv.org/html/2509.20904v1)
- **验收**：所有 SID 模型（TIGER/LCRec/COBRA/OneRec/RPG）的碰撞率列入主结果表；附「碰撞率 vs R@10」相关性一图。
- **工作量**：S

> **Tier 0 产出**：一篇「Honest Benchmarking of Generative Recommenders」——去泄漏协议 + 碰撞率 + GRID 方法学验证，几乎不训练新模型，全用现有 9 模型 + 本 session 已有数据。**最防御性、最快收割。**

---

### 🟡 Tier 1 — 干净的新对照轴，学术规模可复现

#### T1-1. RPG 长 SID（n=32 → n=64）
- **做什么**：把已有 RPG 从 n=32 扩到论文的 n=64，复现并行解码的核心 claim。
- **为什么**：一半工作仓库已有；复现「平均 N@10 +12.6%、推理 +15×、显存 −25×」（原文称）三条硬 claim。
- **接口点**：`genrec/models/rpg.py` + `rpg_trainer`（✅ 已有 n=32）。
- **代码**：[RPG, KDD'25, arXiv 2506.05781](https://arxiv.org/abs/2506.05781)
- **验收**：n∈{32,64} 的 R@10/N@10/推理时延对照；复现长 SID 增益曲线。
- **工作量**：S–M

#### T1-2. LETTER 可学习 tokenizer（语义 + 协同 + 多样性正则）
- **做什么**：在现有 RQ-VAE 上加协同对齐 loss（注入 CF 信号）+ 多样性正则（抗坍塌）。
- **为什么**：补「语义+协同混合 ID」空白；diversity loss 与我们的 dead-cluster reinit 做抗坍塌对照 → 正中 Phase 3「Codebook Collapse」论文。可复用 SASRec/HSTU 产 CF 信号。
- **接口点**：`genrec/models/rqvae.py`（🟡 只有语义重建）+ 任一 CF 模型 embedding。
- **代码**：[LETTER, arXiv 2405.07314](https://arxiv.org/abs/2405.07314)（WWW'24 / ICLR'25）
- **验收**：RQVAE vs LETTER 在 TIGER/COBRA backbone 上的 R@10 + 碰撞率 + code 利用率对照。
- **工作量**：M

#### T1-3. OneRec-Think 的 Rollout-Beam reward（打 reward 稀疏）
- **做什么**：把 GRPO 的 reward 从 exact 1-hit 换成 beam 内「多有效匹配」reward（K=32 beam 内最大 token 匹配）。
- **为什么**：**与我们 Qwen3-1.7B + K=256 + Beauty 设定撞车度最高，几乎零适配**；beam 内匹配比 exact 稠密得多，直接打 Finding 3 的 reward 稀疏（87% dead step），有望真正撬动 R@1 天花板。
- **接口点**：`genrec/trainers/onerec_grpo_trainer.py` 的 `compute_*_reward`（✅ 已有 GRPO 基础设施，只换 reward 函数）。
- **代码**：[OneRec-Think, arXiv 2510.11639](https://arxiv.org/abs/2510.11639)
- **验收**：同一 SFT ckpt 上 {exact / rank / Rollout-Beam} reward 的隔离增益对照；看 R@1 是否突破 ~0.019。
- **工作量**：S–M

#### T1-4. S-DPO 离线偏好对齐（补缺失象限）
- **做什么**：实现 softmax-DPO（1 正多负），作为 GRPO 之外的离线对齐路线。
- **为什么**：补 GenRec 缺的「离线对齐」象限，与「GRPO 边际」正交。**有隔离增益 HR@1 +13~47%（原文称，但在 LastFM/Goodreads/ML，非 Amazon）→ 在 Beauty 上能否复现是开放问题，正负结果都可发。**
- **接口点**：新增 `genrec/trainers/onerec_dpo_trainer.py`（参照 sft/grpo trainer 结构）。
- **代码**：[S-DPO, NeurIPS'24, arXiv 2406.09215](https://arxiv.org/abs/2406.09215)（有代码）
- **验收**：`SFT-only / +GRPO / +S-DPO` 三方隔离增益表（Beauty）。
- **工作量**：M

> **Tier 1 产出**：填满训练轴（`SFT / +GRPO / +S-DPO / +Rollout-Beam`）的**隔离增益矩阵** + tokenizer 轴的 LETTER 对照 + 解码轴的 RPG 长 SID。这正是 MiniOneRec/OneRec-Think 都回避的诚实数字。

---

### 🔵 Tier 2 — 更大投入或更偏研究

| 项目 | 价值 | 代码 | 工作量 | 备注 |
|---|---|---|---|---|
| **ActionPiece**（ICML'25 spotlight） | 上下文感知 BPE tokenizer，原文称 +6~13% NDCG@10，与现有 per-item 固定码全正交 | [DeepMind 官方](https://github.com/google-deepmind/action_piece) | M–L | 接口差异大，范式新 |
| **ETEGRec**（SIGIR'25） | 端到端可学习 tokenizer，代表 differentiable SID 主线（一次覆盖 UniSID/Differentiable-SID） | [BishopLiu/ETEGRec](https://github.com/BishopLiu/ETEGRec) | L | 两阶段 → 端到端联合 |
| **DiffGRM**（WWW'26） | masked diffusion 非自回归解码，原文称 Beauty N@10 +8.2% | [arXiv 2510.21805](https://arxiv.org/abs/2510.21805) | L | 凑齐「AR/并行/diffusion」三范式 |
| **ReaRec**（SIGIR'25） | 隐式 latent test-time scaling，**可挂在 SASRec/HSTU 上**，原文称稀疏/长尾 +30~50% | [arXiv 2503.22675](https://arxiv.org/abs/2503.22675) | M | 差异化最强：判别式 backbone 也能 inference-time reasoning |
| **Rotation Trick / FSQ** | RQVAE 量化器两根「无坍塌」对照棒，各几十行 | [Rotation 2410.06424](https://arxiv.org/abs/2410.06424) / [FSQ 2309.15505](https://arxiv.org/abs/2309.15505) | S | 并入量化器横评消融 |
| **On-policy Distillation**（OpenOneRec） | 唯一系统解决「RL 倒扣通用能力」的训练阶段（reverse-KL teacher） | [Thinking Machines blog](https://thinkingmachines.ai/blog/on-policy-distillation/) | M–L | 仅当做「通用能力不退化」叙事时需要 |

---

### ⚫ 不复现（工业 / 异范式 / 偏离 moat）
MTGR、RecGPT(-V2)、OneRec-IPA（依赖工业 reward model）、RecZero（评分预测任务）、AgentRecBench、PixelRec（2 亿交互太重）、HSTU Context Parallelism（工业长序列）。

---

## 3. 建议执行顺序（贴现有 Phase 2/3 roadmap）

**第一步（2–3 周，纯收割，最防御性）** → Tier 0 三件套
产出 GenRec「诚实评测层」（去泄漏协议 + 碰撞率 + GRID 方法学验证）。几乎不训练新模型，全用现有 9 模型 + 本 session 已有的 TIGER 碰撞数 / v2b SFT / RL 边际结论。可直接成文「Honest Benchmarking of Generative Recommenders」。

**第二步（并行两条）**
- tokenizer 轴：**T1-2 LETTER** → 接 Phase 3「Codebook Collapse」论文；
- 训练轴：**T1-3 Rollout-Beam reward** + **T1-4 S-DPO** → 凑齐 `SFT / +GRPO / +S-DPO / +Rollout-Beam` 隔离增益矩阵。

**第三步（扩范式）**
**T1-1 RPG n=64** + **T2 DiffGRM** → 「AR / 并行 / diffusion」三解码范式公平对照。

---

## 4. 里程碑与验收

| 里程碑 | 内容 | 验收信号 |
|---|---|---|
| M1（诚实评测层） | T0-1/2/3 | 三套对照表 + 碰撞率-Recall 相关图；现有 9 模型双协议数字 |
| M2（训练轴矩阵） | T1-3/4 | `SFT/+GRPO/+S-DPO/+Rollout-Beam` 隔离增益表（Beauty，可选 Sports/Toys） |
| M3（tokenizer 轴） | T1-2 + T2 Rotation/FSQ | RQVAE/RQKMeans/LETTER/FSQ 量化器横评 → Codebook Collapse 论文初稿 |
| M4（解码轴） | T1-1 + DiffGRM | AR/并行/diffusion 三范式 R@10/N@10/时延对照 |

---

## 5. 待决 / 风险

- **远程算力**：每个 1.7B SFT/GRPO run ~5–9h（4×A800）。Tier 0 基本零训练，Tier 1/2 需排队。
- **数字可信度**：Tier 1/2 多数外部增益为「原文称」，进入排期前需精读原文表格确认 backbone/split/算力。
- **工作流约束**：本地改代码 → `upload.sh` 反向 mirror 到远程，**不直接改远程**；commit 直接上 main，无 claude 痕迹。

---

*维护：随复现进展更新各 T 项状态（TODO / 进行中 / 完成 + 结果链接）。*
