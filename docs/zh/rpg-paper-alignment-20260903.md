# RPG Beauty 论文对齐复核（2026-09-03）

## 结论

RPG 的 direct decoding 是当前 Beauty leave-one-out 的冻结选择。三种子严格
one-shot test 为：

| 指标 | seed 42 | seed 43 | seed 44 | mean ± population std |
|---|---:|---:|---:|---:|
| R@1 | 0.02178 | 0.01847 | 0.01896 | 0.01973 ± 0.00146 |
| R@5 | 0.05098 | 0.05098 | 0.04977 | 0.05057 ± 0.00057 |
| R@10 | 0.07298 | 0.07414 | 0.07249 | **0.07320 ± 0.00069** |
| N@5 | 0.03651 | 0.03489 | 0.03478 | 0.03539 ± 0.00079 |
| N@10 | 0.04358 | 0.04235 | 0.04212 | **0.04268 ± 0.00064** |

当前 tokenizer 使用 Sentence-T5-XL + PCA-512，并非论文 headline 所用的
`text-embedding-3-large`。论文 headline 为 R@5=0.0550、R@10=0.0809、
N@10=0.0464；当前分别低约 8.0%、9.5%、8.0%。论文 encoder/ID 消融中
Sentence-T5-base long-ID 的 N@10=0.0429，可作为相近量级参考，但不是完全相同
的 encoder 配方。

## Graph validation

旧矩阵记录的 graph 参数是 beam=20、edges=200、steps=3。论文 Appendix B
Table 6 对 Beauty 给出 beam=10、edges=100、steps=3，因此使用现有
validation-best checkpoint 做了独立三种子验证：

| 解码 | validation R@10 | validation N@10 |
|---|---:|---:|
| direct | **0.09391** | **0.05679** |
| graph，论文 10/100/3 | 0.08583 | 0.05226 |

图传播在三个 seed 上都下降，因此冻结 direct decoding。作为诊断，冻结 test
中论文 graph 参数也仅有 R@10=0.06737 ± 0.00050、N@10=0.03956 ± 0.00072，
不作为 headline。

## Artifact 与任务

- graph validation：
  `out/recovery/rpg/beauty/validation_graph/paper_table6/seed{42,43,44}.json`
- frozen test：
  `out/recovery/rpg/beauty/test_frozen/direct/seed{42,43,44}.json`
- validation tasks：347、348、349
- frozen test tasks：350、351、352
- checkpoints：
  `out/matrix/leave_one_out/rpg/beauty/seed{42,43,44}/best_model.pt`
- evaluation source bundle：
  `genrec-code-20260903T070137Z-e52cc02-dirty-95508.tar.gz`
- bundle SHA256：
  `2f68e0eef7dde005e28d6bf26125fa9d969152504991eaa6ece60b76a366b486`

独立 evaluator 重现了旧 `results.json` 的 direct test 数字，说明单卡旧训练最终
确实加载了 validation-best checkpoint。trainer 已另外修正为所有 rank 都加载
best checkpoint，并移除逐 epoch test 评测，避免后续分布式运行出现 rank 状态不一致
或人为 test 选择偏差。
