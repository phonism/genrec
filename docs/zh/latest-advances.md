# 研究进展入口

!!! note "页面迁移"

    本页原先是一份 2025–2026 论文快照，模型数量、实现状态和部分 headline 数字已经落后于代码。自 2026-08 起，持续维护的内容迁移到[GenRec 研究全景](landscape/index.md)；本页保留为稳定入口，不再复制一份容易漂移的论文清单。

## 当前研究地图

| 方向 | 核心问题 | 从这里继续 |
|---|---|---|
| 语义 ID 与 tokenizer | 如何获得有语义、均衡、低碰撞且唯一可寻址的 item code | [分类框架](landscape/taxonomy.md)、[tokenizer 条目](landscape/model-catalog.md#6-tokenizer) |
| 自回归生成式检索 | 如何在目录约束下生成下一物品或推荐列表 | [模型目录](landscape/model-catalog.md#1-id) |
| sparse–dense 混合 | 如何兼顾生成能力、冷启动、目录更新和精确 item grounding | [模型目录](landscape/model-catalog.md)、[实现待办](landscape/implementation-backlog.md) |
| 并行、masked 与 diffusion | 如何摆脱严格左到右生成的延迟与误差传播 | [扩散/流模型](landscape/model-catalog.md#4) |
| LLM 与原生文本 | 何时生成文本，何时生成 item token，何时只做向量检索 | [LLM 路线](landscape/model-catalog.md#2-llm) |
| 偏好优化与推理 | RL/DPO/过程奖励是否在相同 SFT 起点上带来可隔离收益 | [RL 与推理](landscape/model-catalog.md#5)、[评测协议](landscape/evaluation.md) |
| 数据与公平比较 | 划分、候选集、SID 碰撞和 item 解析如何影响结论 | [数据集](landscape/datasets.md)、[评测协议](landscape/evaluation.md) |

研究全景以 **2026-08-31** 为快照日期，模型目录共收录 68 个当前实现、未实现或相邻方法/组件，并为每项标注论文/官方代码、本仓库状态和建议优先级。它是代表性地图，不声称穷尽所有论文。

## 与当前仓库的关系

- [模型总览](models/overview.md)只描述当前仓库实际能运行的模型与组件。
- [实现与复现状态](reference/model-status.md)区分“有代码”“可端到端运行”“有 item-level 证据”和“复现论文结果”。
- [模型目录](landscape/model-catalog.md)同时收录已实现与未实现方法；其中“未实现”不表示质量较低。
- [实现待办](landscape/implementation-backlog.md)按本项目的公平比较价值和工程复用度排序，而不是按论文热度排序。
- [复现行动计划](reproduction-roadmap.md)是带日期的研究记录，包含历史实验结论；当前执行状态仍以 artifact、运行手册与 `BENCHMARKS.md` 为准。

## 阅读与引用原则

模型机制和实验结论应回到论文、会议页面或作者官方仓库核对；综述和聚合列表只用来发现线索。预印本的名称、版本、录用状态和代码地址可能变化，开始复现前应重新确认。完整的一手资料路线见[阅读清单](landscape/reading-list.md)。

对外引用仓库结果时，不要从旧快照复制数字。请先确认：

1. 数据版本和划分协议相同；
2. 候选集合与解码预算相同；
3. SID 已解析到唯一 item，并在排序后去重；
4. checkpoint、seed、代码版本和数据指纹可追踪；
5. 指标来自当前 item-level `results.json`，而不是历史 SID-tuple 日志。

当前已验证结果见 [BENCHMARKS.md](https://github.com/phonism/genrec/blob/main/BENCHMARKS.md)。
