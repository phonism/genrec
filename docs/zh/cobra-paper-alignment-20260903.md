# COBRA Beauty 论文对齐复核（2026-09-03）

## 结论

COBRA 的新训练配方在三种子 validation 上稳定，但没有把同等收益带到 test。
严格 item-level、Beauty leave-one-out、三种子冻结测试的最终结果为：

| 指标 | seed 42 | seed 43 | seed 44 | mean ± population std |
|---|---:|---:|---:|---:|
| R@1 | 0.00863 | 0.00738 | 0.00872 | 0.00824 ± 0.00061 |
| R@5 | 0.02974 | 0.02674 | 0.02889 | 0.02846 ± 0.00126 |
| R@10 | 0.04682 | 0.04418 | 0.04691 | **0.04597 ± 0.00127** |
| N@5 | 0.01915 | 0.01705 | 0.01874 | 0.01831 ± 0.00091 |
| N@10 | 0.02464 | 0.02267 | 0.02449 | **0.02393 ± 0.00090** |

论文 Beauty 为 R@5=0.0537、R@10=0.0725、N@10=0.0456。当前 R@10
仍低 36.6%。相对此前 optimized-light 的 R@10=0.0454，本轮净提升约 1.3%，
不能宣称已复现论文结果。

## 冻结过程

所有架构、checkpoint 和 BeamFusion 参数都先由 validation 决定，随后只进行一次
test 评测。冻结配方如下：

- epoch 49；一层 LightT5 encoder、mean pooling、两层 decoder；
- `title_brand` 文本，dense InfoNCE 使用 all-batch negatives；
- memory queue 关闭；beam 64，`tau=1`，`psi=8`；
- `max_seq_len=50`，item-level 评测，训练交互内 most-popular SID→item
  消歧并对输出 item 去重。

三种子 validation R@10 为 0.06538 / 0.06444 / 0.06350，汇总为
0.06444 ± 0.00077；R@5=0.04018 ± 0.00073，N@10=0.03381 ± 0.00033。
结果跨 seed 稳定，因此 test 的下降不能归因于单 seed 偶然抬高。

memory queue 是论文之外的扩展。queue=1024、完整属性文本和 CLS pooling 的
validation 消融都没有胜过上述无 queue 配方，因此未进入冻结测试。

## Artifact 与任务

- validation 选择：
  `out/recovery/cobra/beauty/validation_search/paper_alignment/frozen_selection.json`
- frozen test：
  `out/recovery/cobra/beauty/test_frozen/paper_alignment/seed{42,43,44}/mean_all_epoch49_tau1_psi8.json`
- validation tasks：342、343（seed 42 artifact 来自此前已完成评测）
- frozen test tasks：344、345、346
- source bundle：
  `genrec-code-20260903T070137Z-e52cc02-dirty-95508.tar.gz`
- bundle SHA256：
  `2f68e0eef7dde005e28d6bf26125fa9d969152504991eaa6ece60b76a366b486`

## 后续优化边界

当前主要问题是 validation→test 泛化落差，而不是 GPU 避让、SID tuple 高估、
checkpoint 未加载或 BeamFusion 单点参数。后续若继续 COBRA，应优先检查训练轮次、
用户末项分布漂移和正则化；不得继续用 test 选择 epoch、queue 或融合权重。
