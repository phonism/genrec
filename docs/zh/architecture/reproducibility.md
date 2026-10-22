# 可复现与实验运行工作流

本页给出从代码到可进入 benchmark 的 artifact 流程。179 上的动态任务状态、卡占用和阶段优先级以 [`179-reproduction-runbook.md`](../179-reproduction-runbook.md) 为准；这里维护稳定的工程合同。

## 什么叫“可复现 cell”

一个结果 cell 的身份至少为：

```text
(model, dataset split, evaluation protocol, seed)
```

但重跑所需 identity 还包括：

```text
source bundle SHA256 + Git SHA
raw data manifest / preprocessing
base model & encoder revision/checksum
tokenizer artifact + C/K/geometry + item order
resolved Gin config
checkpoint selection rule
hardware/software environment
```

只记录 Git SHA 不够：研究 worktree 可能有未提交修改。只记录 checkpoint 路径也不够：路径内容可能被覆盖。

## 运行前冻结协议

在启动 GPU 前确定：

1. 模型及具体 config 文件；
2. split；
3. `leave_one_out` 或 `global_temporal`；
4. seed 集合；
5. `max_seq_len`、candidate universe、headline metrics；
6. tokenizer 是否跨下游 seed 固定；
7. valid selection 和 test 刷新规则；
8. 输出目录，必须隔离 protocol/model/split/seed。

推荐输出：

```text
out/matrix/<protocol>/<model>/<split>/seed<seed>/
```

不要在同一目录重跑不同 protocol、C/K 或 encoder。`--force` 只代表允许覆盖执行，不会替你保存旧 artifact。

## 环境与依赖

安装边界见[安装文档](../installation.md)。正式运行前记录：

```bash
python -VV
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -c 'import transformers, accelerate, gin; print(transformers.__version__, accelerate.__version__)'
nvidia-smi
git rev-parse HEAD
git status --short
```

推荐从 `pyproject.toml` 安装对应 extras：RPG 需要 FAISS；LoRA 需要 PEFT；Parquet 需要可用 pandas engine。正式任务使用本地模型镜像，并为模型目录做 manifest/checksum；不要让 Hugging Face “最新 revision”静默改变输入。

179 的 Python 环境是 `/root/workspace/code/genrec/venv`。模型必须从 AFS backup 恢复到 `/root/workspace/models_hub/` 后再训练，禁止直接从 AFS 路径流式读取。

## 数据验收

至少记录 review/meta 文件：相对路径、字节数、SHA-256。然后验证：

- gzip 可完整解压；
- unique item/user/interaction 数；
- mapping 与 embedding/SID row 数一致；
- train/valid/test 样本数；
- protocol cutoff 和过滤前/后用户数；
- target 不进入自己的 history，global temporal history 不跨 cutoff。

`global_temporal` 仍使用全 catalog metadata/SID，是 transductive catalog 设置。报告中应写“future-interaction leakage-resistant”，不要称为冷物品归纳评测。

Amazon 2023 先确认兼容 gzip 实际位于通用 Dataset 读取的 `raw/<split>`；不要在没有目录迁移说明时直接实例化读取 `processed/<split>` 的早期专用类。

## tokenizer / 表征 stage

### TIGER / LCRec / COBRA 的 RQ-VAE

三个 family 的几何不兼容，分别训练：

```bash
python genrec/trainers/rqvae_trainer.py config/tiger/amazon/rqvae.gin --split beauty
python genrec/trainers/rqvae_trainer.py config/lcrec/amazon/rqvae.gin --split beauty
python genrec/trainers/rqvae_trainer.py config/cobra/amazon/rqvae.gin --split beauty
```

训练后用同一个 encoder 和 catalog 做全量健康检查：

```bash
python scripts/verify_codebook_health.py \
  --kind rqvae \
  --checkpoint out/tiger/amazon/beauty/rqvae/checkpoint_epoch_4999.pt \
  --dataset-folder dataset/amazon \
  --split beauty \
  --encoder /absolute/path/sentence-t5-xl
```

记录各 layer usage/dead/max-frequency、完整 tuple collision 和最大碰撞组。下游三个 seeds 推荐固定使用同一份已验证 tokenizer，以测 recommender seed 方差；tokenizer seed 方差另做实验。

### OneRec RQ-KMeans

```bash
python genrec/trainers/rqkmeans_trainer.py \
  config/onerec/amazon/rqkmeans.gin --split beauty

python scripts/verify_codebook_health.py \
  --kind rqkmeans \
  --checkpoint out/onerec/amazon/beauty/rqkmeans/rqkmeans.pt \
  --dataset-folder dataset/amazon \
  --split beauty \
  --encoder /absolute/path/Qwen3-Embedding-8B
```

小 catalog 使用当前 `K=256,C=3` 主配置。健康检查中的 collapse signature 不是硬性论文结论，但 `usage<80%` 或单 code `max_freq>10%` 必须停下解释，不能直接进入 RL。

进一步碰撞报告：

```bash
python scripts/report_sid_collisions.py \
  --sem-ids /path/to/sem_ids.npy \
  --codebook-size 256
```

### RPG

RPG Dataset 首次构造会生成文本 embedding 和 OPQ SID。OPQ 只用 protocol training prefix 中出现的 items 拟合，cache 名已包含 protocol/PCA/factory。LOO 与 global temporal 不能共用 SID cache。

## smoke test

在完整训练前按风险递增：

1. `pytest -q`；
2. Dataset 构造 + 第一条样本/一个 collated batch；
3. 单 forward/backward，检查 loss 有限；
4. 一小批 eval，检查 prediction shape、item范围和去重；
5. 一个 epoch/极少样本的独立 debug 输出目录；
6. checkpoint round-trip；
7. `results.json` schema/metric range。

示例只用于 smoke，不能与正式结果共目录：

```bash
python genrec/trainers/sasrec_trainer.py config/sasrec/amazon.gin \
  --split beauty \
  --gin 'train.epochs=1' \
  --gin 'train.wandb_logging=False' \
  --gin 'train.save_dir_root="out/smoke/sasrec/beauty"'
```

大模型可用 `max_train_samples`/`max_eval_samples` 的 trainer 才能做 sample cap；不要给不支持该参数的 trainer 传 Gin binding。

## 单次正式训练

### baseline

```bash
python -m genrec.trainers.sasrec_trainer config/sasrec/amazon.gin --split beauty \
  --gin 'train.seed=42'

python -m genrec.trainers.hstu_trainer config/hstu/amazon.gin --split beauty \
  --gin 'train.seed=42'
```

### 多阶段生成模型

```text
TIGER:  matching RQ-VAE → tiger trainer
LCRec:  matching RQ-VAE → lcrec trainer
COBRA:  matching RQ-VAE → cobra trainer
OneRec: RQ-KMeans → SFT 或 CLM
         SFT checkpoint_best → GRPO
RPG:    Dataset 内 OPQ cache → rpg trainer
```

OneRec SFT 多 GPU 示例：

```bash
accelerate launch --config_file config/accelerate_4gpu.yaml \
  -m genrec.trainers.onerec_sft_trainer \
  config/onerec/amazon/sft.gin --split beauty
```

[`config/accelerate_4gpu.yaml`](https://github.com/phonism/genrec/blob/main/config/accelerate_4gpu.yaml) 当前写死物理 `gpu_ids: 3,4,5,6`。在共享主机必须让它与外层分配一致，或使用显式 `CUDA_VISIBLE_DEVICES`/匹配配置；不能因文件名叫 4gpu 就默认可用任意四张卡。

仓库中的某些历史 shell 编排脚本引用已不存在的 `amazon2023/*.gin`、`accelerate_ddp.yaml` 或旧 `sft_k256_v2.gin` 名称。执行前必须检查脚本引用是否在当前 tree 存在；正式命令以当前 [`config/`](https://github.com/phonism/genrec/tree/main/config) 和 matrix registry 为准。

## 矩阵规划与运行

先 dry-run，逐条审阅 config、split、protocol、seed、save dir：

```bash
python scripts/run_matrix.py \
  --models sasrec hstu \
  --splits beauty sports toys \
  --seeds 42 43 44 \
  --evaluation-protocol leave_one_out \
  --dry-run
```

实际运行去掉 `--dry-run`。runner：

- 串行启动子进程；
- 已有 `results.json` 时 skip，除非 `--force`；
- 可 `--continue-on-error`；
- 可用 `--log-dir` 重定向 cell stdout/stderr；
- 给 seed/save dir/protocol 注入 Gin override；
- 不创建 RQ-VAE/RQ-KMeans/OPQ prerequisite；
- 只对 OneRec GRPO 显式检查匹配 SFT `checkpoint_best`。

RPG 根据 split 路由 Beauty/Sports/Toys 专用 config。若为模型提供 `--config model=path`，要重新审计跨阶段 artifact。

在 179 上“一个 MCP task 一个 cell”，不要把整个矩阵包装成一个不透明长任务。可用 dry-run 得到命令，再分别提交并用相同目录/命名规则。

## 179 并发安全流程

### 同步源码

本地：

```bash
./upload.sh
```

它创建 source-only tar，排除 `.git`、数据、模型、out、secret 和大权重，计算 SHA-256，通过带 token/header 的上传服务提交。上传本身不改远端代码；随后在 179 执行脚本打印的：

```bash
bash /root/workspace/uploads/genrec-apply-....sh \
  /root/workspace/uploads/genrec-code-....tar.gz <sha256>
```

apply helper 会：

- 只接受 upload root 内 archive，并核对 SHA-256；
- 获取独占 source sync lock；
- 拒绝 stale bundle；
- 检查 path traversal 和 GenRec 正在运行的进程；
- staging 后 rsync，保留远端 dataset/models/out/venv；
- 记录最后 bundle、SHA 与可解析 Git SHA。

当前 rsync 没有 `--delete`：bundle 会新增/覆盖源码，但本地已经删除的旧文件可能继续留在远端 tree。涉及重命名/删除 importable 模块时，必须额外核对远端文件清单并通过受控变更处理，不能仅凭“apply 成功”断言远端 tree 与 bundle 完全同构。不要绕过 helper 直接覆盖训练中的 repo。

### 提交前只读检查

每次任务先同时查看远端 task list 的 running 项和 GPU 状态。显存看似空闲不代表没有另一个 session 已提交待启动任务。不得取消/kill 未知任务来抢卡。

MCP task 名：

```text
genrec:<model>:<split>:<protocol>:s<seed>:<purpose>
```

wrapper cell ID：

```text
<model>.<split>.<protocol>.s<seed>.<purpose>
```

### 每个任务必须加锁

```bash
scripts/run-locked-179.sh \
  --gpus 2 \
  --cell sasrec.beauty.leave_one_out.s42.train \
  -- venv/bin/python -m genrec.trainers.sasrec_trainer \
  config/sasrec/amazon.gin --split beauty \
  --gin 'train.seed=42' \
  --gin 'train.save_dir_root="out/matrix/leave_one_out/sasrec/beauty/seed42"'
```

MCP `task_submit.gpu` 的物理卡列表必须和 `--gpus` 完全相同。wrapper：

1. 获取运行期间共享 source lock；
2. 获取唯一 cell lock；
3. 对每张物理 GPU 获取独占锁；
4. 精确设置 `CUDA_VISIBLE_DEVICES`；
5. 获锁后再次用 `nvidia-smi` 检查非协作进程；
6. 只放行 `[Not Found]` 且不超过 2 GiB 的已知 auto-yield hold signature；
7. 注入 cell/GPU/source bundle provenance 到环境。

CPU-only 数据任务也用唯一 cell：

```bash
scripts/run-locked-179.sh --gpus none \
  --cell data.amazon2014.beauty.prepare -- <command>
```

## 结果验收

2026-10-22 已验收的快照包含 162 个主矩阵 cell 与 6 个独立补充 cell。
逐 seed 指标、配方、源码基线、原始 artifact 路径和 SHA256 已提交到
[`results/matrix_20261022.csv`](https://github.com/phonism/genrec/blob/main/results/matrix_20261022.csv)，
汇总与源码补丁校验和见
[`matrix_20261022.summary.json`](https://github.com/phonism/genrec/blob/main/results/matrix_20261022.summary.json)。
162 格检查了完整的 9 管线 × 3 数据集 × 2 协议 × 3 seed 覆盖；4 码 OneRec
与 HSTU sampled-softmax 消融各自单列，不能直接对整个历史 `out/` 聚合。
结构校验通过不保证 prompt 确定性：OneRec/LCRec 当前仍随机选择评测模板，
GRPO 16/18 格选回 SFT，因此微小的 SFT→GRPO 分差不能视为隔离 RL 收益。

先严格验证：

```bash
python scripts/validate_matrix_results.py out/matrix/leave_one_out \
  --models sasrec hstu \
  --splits beauty sports toys \
  --protocols leave_one_out \
  --seeds 42 43 44
```

再聚合：

```bash
python scripts/aggregate_results.py out/matrix/leave_one_out \
  --metrics Recall@5 Recall@10 NDCG@10 \
  --format md
```

聚合器使用跨规划 seed 的总体标准差 `pstdev`。缺失/重复 cell 必须显式修复，不能用旧 SID 指标、不同 protocol 或不同 tokenizer 补位。

结果审计清单：

- `metrics` 是 item-level，Recall 单调；
- SID 模型记录 training-only resolver 与 SID diagnostic；
- `config.evaluation_protocol` 存在；
- `extra.evaluation_level="item"`；
- source bundle 名/SHA、具体 Git SHA、cell/GPU 信息存在；
- best epoch 与所报告 test/checkpoint 对应；
- log 中没有 NaN、OOM 后继续或缺失 eval shard；
- tokenizer/codebook 健康报告与下游 artifact 同源。

## provenance 回填

旧 `results.json` 只有在**确知**它来自某个 exact bundle 时才可用 [`stamp_results_provenance.py`](https://github.com/phonism/genrec/blob/main/scripts/stamp_results_provenance.py)：

```bash
python scripts/stamp_results_provenance.py out/path/results.json \
  --bundle-name genrec-code-....tar.gz \
  --sha256 <64-lowercase-hex> \
  --git-sha <7-to-40-hex> \
  --cell-id model.split.protocol.s42.eval \
  --physical-gpus 2
```

脚本原子写入并拒绝覆盖冲突值，但它无法推断历史来源。没有证据时保持 provenance 缺失并让 validator fail，比伪造 lineage 更正确。

## 已知非位级确定性

- `set_seed(..., deterministic=False)` 默认不强制 cuDNN deterministic；
- CUDA kernel、FSDP/DDP reduction 和混合精度可能有数值非确定性；
- Python hash seed 在 trainer 内设置得较晚；
- DataLoader 多 worker、运行时随机负采样会消费 RNG；
- RPG graph 路径可能受 eval 前 RNG 状态影响；
- 外部 encoder/LM 若只记目录名而不记 revision/checksum，会漂移；
- basename-only embedding/text cache 不能证明输入等价。

因此 benchmark 采用多 seed 均值/总体标准差，而不是宣称跨机器 bit-exact。若研究问题需要 bit-exact，应在进程启动前设置 hash seed、开启 deterministic algorithms、固定依赖/硬件，并单独记录由此产生的性能与算子限制。

## 失败恢复

1. 保留原 log、输出目录和 source provenance；
2. 判断 checkpoint 是否包含 optimizer/scheduler；原始 state dict 不能完整 resume；
3. 不在同一目录混入改变过的 Gin 参数；若必须恢复，记录 resume source 和 next epoch；
4. 检查上游 tokenizer/cache 没有在失败期间被重建；
5. 179 重提仍使用同一稳定 cell ID，wrapper 会防止并行重复；
6. 成功产出后再运行 validator，不以文件存在作为完成证明。
