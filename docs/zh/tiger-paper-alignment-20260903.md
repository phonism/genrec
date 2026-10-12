# TIGER 论文对齐恢复记录（2026-09-03）

## 目标与评估纪律

- 目标：尽量复现 TIGER 在 Amazon 2014 Beauty 上的论文结果。
- 推荐协议：leave-one-out；最后一次交互为 test，倒数第二次为 valid。
- 搜索阶段只使用 item-level valid 指标；配置冻结后才允许每个 seed 读取一次 test。
- 所有 Semantic ID 都使用 3 个 RQ code 加唯一 collision suffix，共 4 个 code；
  不再用可能被碰撞高估的 3-code tuple 指标选模型。
- 本轮远端实验仅使用物理 GPU 4、5、6、7。

论文 Beauty headline：R@5=0.0454、N@5=0.0321、R@10=0.0648、
N@10=0.0384。附录三次运行均值为 R@5=0.0441、N@5=0.0309、
R@10=0.0642、N@10=0.0374。

## 已对齐的实现细节

- item 文本：title、price、brand、category；Sentence-T5-XL，768 维。
- RQ-VAE：`768 -> 512 -> 256 -> 128 -> 32`、ReLU、bias、3×256、
  beta=0.25、batch=1024、Adagrad lr=0.4。
- SID：为所有 item 追加第 4 个唯一后缀，未碰撞 item 的后缀为 0。
- T5：4 层 encoder + 4 层 decoder、6 heads、`d_kv=64`、
  `d_model=128`、FFN=1024、ReLU、dropout=0.1。
- 序列：最多 20 个历史 item；可选 2000 桶稳定 user hash token。
- 优化：T5X 风格 Adafactor；前 10k step lr=0.01，之后 inverse-sqrt；
  z-loss=1e-4。
- 训练中记录 item-level、SID-level 诊断、invalid SID rate，并支持完整恢复。

## RQ-VAE 诊断

PyTorch Adagrad 默认 `initial_accumulator_value=0`、`eps=1e-10` 时，
论文的 lr=0.4 会在首批更新后塌缩。改用 Google/TF/Optax 语义的
`initial_accumulator_value=0.1`、`eps=1e-7` 后，100 轮即恢复健康。

| epoch | layer usage | raw collision | 结论 |
|---:|---|---:|---|
| 99 | 97.3% / 97.7% / 96.1% | 0.78% | 健康 |
| 149 | 97.3% / 97.7% / 96.1% | **0.72%** | 当前最佳 |
| 199 | 96.9% / 98.0% / 96.1% | 0.93% | 健康 |
| 224 | 94.9% / 98.0% / 96.5% | 1.78% | 健康但变差 |
| 249 | 71.9% / 89.1% / 92.2% | 39.86% | 突发塌缩 |
| 999 | 29.7% / 91.8% / 91.8% | 5.72% | 首层仍塌缩 |

因此不能把“训练满 20k”误解为“使用最后一个 checkpoint”。训练器现按
论文的逐层 usage >=80% 作为硬门槛，再在合格 checkpoint 中优先选择较低
碰撞率，写入 `checkpoint_best_health.pt`。paper config 每 50 epoch 验收一次。

进一步核对论文损失公式发现，论文 decoder 直接重构输入 embedding，并未声明
对 decoder 输出做 L2 归一化。仓库旧实现会强制归一化；现已增加向后兼容开关，
paper config 关闭该归一化。500 epoch 复现实验表明该变体不再出现约 250 epoch
的突发塌缩；当前健康最优点为 epoch 425，layer usage 为
94.1% / 94.5% / 94.9%，raw collision=0.760%。健康度略逊于归一化版本的
epoch 149（raw collision=0.719%），但训练稳定性更好。两者不能仅凭碰撞率
排序，最终仍以 item-level validation 的下游序列模型结果裁决。

raw-decoder tokenizer 的下游 10k validation 结果为：epoch425 checkpoint
R@10=0.06936，epoch50 checkpoint R@10=0.06837。前者高于归一化 decoder
严格论文 tokenizer 的约 0.0686，但仍低于当前旧 RQ 候选的 0.07901。
epoch50 分支已在该节点停止。epoch425 分支在 15k/20k 分别达到
R@10=0.07776/0.08013；20k 只比归一化严格论文 tokenizer 的约 0.0797
高 0.0004，仍比旧 RQ 候选的 0.09082 低 0.0107。结合归一化分支到
30k 也只达到 0.08317，raw 分支已在 20k 筛除，不再消耗 test 预算。

## 序列模型 validation 结果（seed 42，旧 RQ-VAE）

均为 4-code 唯一 SID、item-level valid；test 未读取。

| optimizer / user | step | R@5 | R@10 | N@10 | invalid SID |
|---|---:|---:|---:|---:|---:|
| Adam lr=1e-3 / user | 10k | 0.0280 | 0.0474 | 0.0236 | 0.006% |
| T5X Adafactor / user | 5k | 0.0377 | 0.0626 | 0.0315 | 0.071% |
| T5X Adafactor / user | 10k | 0.0476 | 0.0749 | 0.0392 | 0.062% |
| T5X Adafactor / no-user | 5k | 0.0420 | 0.0672 | 0.0343 | 0.076% |
| T5X Adafactor / no-user | 10k | **0.0513** | **0.0790** | **0.0419** | 0.066% |
| T5X Adafactor / no-user | 15k | 0.0597 | 0.0877 | 0.0483 | 0.074% |
| T5X Adafactor / no-user | 20k | **0.0606** | **0.0908** | **0.0510** | 0.082% |
| T5X Adafactor / no-user | 25k | 0.0613 | 0.0899 | 0.0503 | 0.081% |
| T5X Adafactor / no-user | 30k | 0.0596 | 0.0899 | 0.0503 | 0.159% |
| T5X Adafactor / no-user | 35k | 0.0591 | 0.0870 | 0.0495 | 0.171% |

当前证据表明 Adafactor 是论文 lr=0.01 的关键缺失项；no-user 在 seed 42
的 valid 上暂时更优，这也与论文附录中 no-user 的 R@10 几乎不下降相容。

将 20k 最优 checkpoint 的 beam 从 20 增至 100，valid R@10 只从
0.090820 变为 0.090864（+0.000045），而 invalid SID rate 从 0.082%
升到 0.593%。因此 beam=20 已足够覆盖 top-10，暂不把大 beam 纳入最终配置。

独立 evaluator 已在第一轮 seed42 候选 checkpoint 上做 valid 重放，逐项复现
训练期结果：R@5=0.060591、R@10=0.090820、N@10=0.050959，invalid SID
rate=0.0816%。这确认最后的 one-shot test 会走同一套 item-level resolver、
去重和 4-code SID 协议，而不是回退到旧的 SID-tuple 指标。

使用健康 paper RQ-VAE epoch149 时，no-user 10k 的 valid 为
R@5=0.0438、R@10=0.0686、N@10=0.0357；带 user 为
R@5=0.0415、R@10=0.0642、N@10=0.0334。二者都达到或接近论文 headline，
但仍显著落后于旧 RQ + no-user + Adafactor。当前验证证据不支持仅按更低的
SID 碰撞率选择 tokenizer，最终仍应由 item-level validation 决定。
paper RQ 的 no-user 分支继续到 15k 后达到 R@5=0.0511、R@10=0.0762、
N@10=0.0423；20k 进一步达到 R@5=0.0542、R@10=0.0797、N@10=0.0443，
25k 达到 R@5=0.0567、R@10=0.0820、N@10=0.0466；30k 为
R@5=0.0571、R@10=0.0832、N@10=0.0474，仍在改善，继续观察到 40k。
35k 回落到 R@10=0.0804；当前 strict 分支最优仍是 30k。
40k 继续回落到 R@10=0.0782，因此 strict 分支冻结为 step30130。

## 最近任务

| task | GPU | 内容 | 数据范围 |
|---:|---:|---|---|
| 320（完成） | 4 | 健康 paper RQ epoch149 + Adafactor，无 user，25k -> 40k | valid-only |
| 323（完成） | CPU | paper-alignment 回归测试，32/32 通过 | 无数据评估 |
| 324（完成） | 4 | 未归一化 decoder 的 paper RQ-VAE，500 epoch | tokenizer health |
| 325（已停止） | 5 | raw-decoder RQ epoch425 + Adafactor，无 user；20k 筛除 | valid-only |
| 326（已停止） | 6 | raw-decoder RQ epoch50 + Adafactor，无 user；10k 筛除 | valid-only |
| 327（完成） | 7 | 冻结候选 standalone evaluator 重放 | valid-only |
| 328（失败） | 7 | optimized include 的 `{split}` 未在子配置展开；未开始训练 | 无评估 |
| 329（完成） | 7 | 旧 RQ 候选 seed42 从头精确训练 20k，best=step15390 | valid-only |
| 331（完成） | 4 | 冻结 seed43 checkpoint 独立评估 | one-shot test |
| 332（完成） | 6 | 冻结 seed44 checkpoint 独立评估 | one-shot test |
| 335（完成） | 5 | 冻结 seed42 checkpoint 独立评估 | one-shot test |
| 336（完成） | CPU | 最终 paper-alignment 回归测试，32/32 通过 | 无数据评估 |

Adam lr=1e-3 对照已完成 50k，最佳 valid R@10=0.0612（step40267），
明显落后于 Adafactor 在 20k 的 0.0908。

旧 RQ + no-user 的 seed42 长跑在 25k/30k/35k 连续三次未超过 20k，
已在 35k 后停止；保留 step20260 最优 checkpoint。

当前 validation winner 已固化为
`config/tiger/amazon/tiger_paper_recipe_optimized.gin`：论文序列模型与优化器
配方、20k steps、no-user、4-code 唯一 SID，加仓库旧 RQ-VAE tokenizer。
严格论文 tokenizer 仍由 `tiger_paper_aligned.gin` 单独保留。

5k valid R@10 在 seed42/43/44 分别为 0.0672/0.0660/0.0686，
初期曲线稳定（均值 0.0676）。
10k valid R@10 分别为 0.0790/0.0776/0.0791（均值 0.0786，范围 0.0015）。
15k valid R@10 分别为 0.0877/0.0883/0.0876（均值 0.0879，范围 0.0008）。
第一轮候选筛选的 20k valid R@10 在 seed42/43/44 分别为
0.09082/0.09113/0.09122；其中 seed42 来自 10k checkpoint 续跑并在
step20260 验收。为统一口径，seed42 后续从头精确跑到 20k，按预先规定的
预算内最佳 valid 规则在 step15390 选中 R@10=0.08814。最终三个冻结
checkpoint 的 valid R@10 均值为 0.09016 ± 0.00101 SE。

旧 RQ 的带 user 对照在 seed42 的 15k/20k valid R@10=0.0777/0.0802，
显著低于 no-user 同节点的 0.0877/0.0908；结合论文 no-user 附录，
最终候选保留 no-user。

## 冻结配置后的 test（item-level LOO）

配置冻结后才启动独立 test evaluator；看到 test 后不再更改 tokenizer、
优化器、训练步数、user-token 或 beam。

| seed | R@5 | R@10 | N@5 | N@10 |
|---:|---:|---:|---:|---:|
| 42 | 0.03904 | 0.06354 | 0.02586 | 0.03373 |
| 43 | 0.04306 | 0.06600 | 0.02903 | 0.03645 |
| 44 | 0.04181 | 0.06694 | 0.02792 | 0.03600 |
| **均值 ± SE** | **0.04130 ± 0.00119** | **0.06550 ± 0.00101** | **0.02760 ± 0.00093** | **0.03539 ± 0.00084** |
| TIGER 附录均值 ± SE | 0.0441 ± 0.00069 | 0.0642 ± 0.00092 | 0.0309 ± 0.00062 | 0.0374 ± 0.00061 |

三 seed 的 raw RQ collision 均为 4.02%，追加第 4 个 collision suffix 后
final SID collision 为 0；平均 invalid SID rate 为 0.104%。相较修复前的
honest item-level 三 seed R@10=0.0568，本轮提升到 0.0655（约 +15.3%）。

最终配置、三 seed test、远端同步和回归测试均已完成。optimized config 已在
顶层重新绑定 `AmazonSeqDataset.split`，配置 smoke check 解析为 `beauty`；
最终回归为 32 passed。
