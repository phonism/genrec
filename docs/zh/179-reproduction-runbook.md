# 179 复现运行手册

更新时间：2026-08-31。本文用于多个 Codex 共享 179 时协调实验；动态状态始终以
`gpu-remote-179.task_list`、`gpu_info` 和任务日志为准。

## 已知环境基线与动态状态

历史任务号、当前空闲 GPU、恢复任务和占位进程不能由静态文档担保。每次提交前都重新查询 `task_list(running)` 与 `gpu_info`，并以 `scripts/run-locked-179.sh` 的锁后复检为最终判据；不要因为本页曾记录“空闲”或“仍在运行”而跳过现场检查。

主机的稳定约定是：代码位于 `/root/workspace/code/genrec`，Python 环境位于该目录的 `venv`，模型必须从 AFS 备份恢复到本地 `/root/workspace/models_hub/` 后使用。Amazon 原始数据、模型分片和 checkpoint 仍需在每次 run 前核验存在性、大小与 checksum。已知低显存 `hold.py` 进程可由包装器识别并自动避让；其他具名或超过阈值的 CUDA 进程一律视为占用，不能擅自终止。

## 开跑门槛

以下条件必须全部满足：

1. 所需模型在本地 `models_hub/`，配置和全部权重分片可读取，禁止直读 AFS。
2. 数据集与目标 checkpoint 已恢复，并记录来源、大小和校验和。
3. 用 `./upload.sh` 上传源码，按输出命令应用；远端核心文件与本地 SHA256 一致。
4. `task_list(running)` 无重复实验，目标卡在 `gpu_info` 中无未知 CUDA 进程。
5. 任务同时通过 `scripts/run-locked-179.sh` 获取 cell、GPU 和源码共享锁。
6. 单 batch smoke test 通过后才放大到完整训练或评测。

数据准备等 CPU-only 任务也必须占用唯一 cell：

```bash
scripts/run-locked-179.sh --gpus none \
  --cell data.amazon2014.raw.prepare -- \
  venv/bin/python scripts/prepare_amazon2014.py --splits beauty sports toys
```

## 任务命名与提交

- MCP task 名：`genrec:<model>:<split>:<protocol>:s<seed>:<purpose>`。
- cell ID：`<model>.<split>.<protocol>.s<seed>.<purpose>`。
- 一个 task 只负责一个矩阵 cell；不要把多个不透明实验塞进同一长任务。
- 输出固定为 `out/matrix/{protocol}/{model}/{split}/seed{seed}`。

示例（仅在 GPU 2 确认空闲后）：

```bash
scripts/run-locked-179.sh \
  --gpus 2 \
  --cell sasrec.beauty.global_temporal.s42.train \
  -- venv/bin/python -m genrec.trainers.sasrec_trainer \
  config/sasrec/amazon.gin --split beauty \
  --gin 'AmazonSASRecDataset.evaluation_protocol="global_temporal"' \
  --gin 'train.seed=42' \
  --gin 'train.save_dir_root="out/matrix/global_temporal/sasrec/beauty/seed42"'
```

MCP `task_submit.gpu` 必须与 `--gpus` 使用相同物理卡号。包装器在拿到 OS 锁后
再次检查 CUDA 进程，因此两个 Codex 同时观察到空卡时也只有一个能启动。

## 复现顺序

### P0：旧 checkpoint 诚实重评（零新训练）

先盘点并恢复 OneRec v2b/v2/GRPO、TIGER、LCRec、COBRA checkpoint。以现有
leave-one-out 设置直接重评：headline 只取 item-level Recall/NDCG，SID 指标仅作
碰撞诊断。先 Beauty 单 cell 验证新旧差值，再扩到 Sports/Toys/Home。该阶段用于
填回 README 被暂时撤下的生成式结果，不得把历史 SID 分数混入新表。

### P1：评测协议基线

在 Beauty/Sports/Toys 上先跑 SASRec、HSTU：

- protocol：`leave_one_out`、`global_temporal`；
- seed：42、43、44；
- 共 2×3×2×3 = 36 cells。

这一步验证数据切分、结果隔离和聚合链路。若同 seed 的样本数或 target 集不一致，
先停下修协议，不启动昂贵生成式模型。

状态（2026-08-30）：36/36 cells 已完成；全部为 item-level，且每个
model × split × protocol 组都有 seeds 42/43/44。三 seed 汇总已写入
`BENCHMARKS.md`。

### P2：非 LLM 生成式矩阵

按 TIGER → COBRA → RPG 顺序跑同样的 3 split×2 protocol×3 seed。下游三个 seed
固定使用同一个已验证 tokenizer artifact，以隔离 recommender 训练方差；tokenizer
seed、碰撞率、利用率单独记录。

### P3：LLM 与 RL

先 LCRec、OneRec-SFT 各跑 seed 42，确认 item-level 指标、显存和耗时，再扩到三
seed。GRPO 最后作为 SFT checkpoint 上的增量消融；保持既定 reward 语义，不把
SID reward 当评测指标。NoteLLM 已有 full-catalog I2I evaluator，但当前 trainer
主要接通 GCL 路径、checkpoint 也不是自包含 artifact；完成数据/目标对齐与复现审计
前仍只列实验结果，不与 next-item 主表直接横比。

## 验收与聚合

- `results.json.extra.evaluation_level` 必须为 `item`。GRAM 当前输出尚缺该字段，修复结果契约前不得送入严格矩阵聚合。
- `config.evaluation_protocol` 必须存在；LOO 与 temporal、SID 与 item 不得混组。
- 保存代码 SHA、数据版本、tokenizer/checkpoint 路径、seed、碰撞统计和任务 ID。
- `results.json` 必须同时有 git SHA 和精确的 upload bundle SHA256；dirty
  worktree 下不能只用 git SHA 作复现标识。
- 用 `scripts/aggregate_results.py` 生成矩阵表；缺失 cell 显式列出，不以旧结果补齐。
