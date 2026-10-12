# 贡献指南

GenRec 是用于生成式推荐研究、复现和公平比较的实验代码库，不是生产服务框架。一个可合并的贡献不仅要“能跑”，还要让下一位研究者能够确认：运行了哪段代码、用了什么协议、产物来自哪个源码快照、文档描述的是否正是当前实现。

## 开始前

先阅读与你的改动直接相关的代码、配置和测试：

- [仓库地图](architecture/repository-map.md)说明目录职责；
- [扩展指南](architecture/extension-guide.md)说明如何加入模型、数据集、训练器和评测；
- [配置系统](architecture/configuration.md)说明 Gin 的解析顺序和 `{split}` 替换；
- [评测系统](architecture/evaluation-system.md)说明 item-level 指标、SID 冲突解析和 `results.json`；
- [文档维护规范](reference/documentation-policy.md)定义事实来源和状态写法；
- [模型文档模板](models/template.md)定义模型页的完成标准；
- [命令索引](reference/command-index.md)只收录已经从当前代码核验的入口。

仓库没有 `genrec.base.BaseModel`，也没有要求所有实现继承的统一 `BaseTrainer`。不同论文的输入、目标和解码方式差异很大；新增实现应复用相邻模型的真实接口，而不是先假设一个不存在的基类。共享能力目前主要位于 `genrec/modules/`、`genrec/trainers/trainer_utils.py` 和 `genrec/modules/utils.py`。

## 环境

从仓库根目录安装开发与文档依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,docs]"
```

深度学习实验还需要与机器 CUDA 驱动匹配的 PyTorch。预训练编码器和语言模型的位置由 `config/base.gin` 中的常量指定；不要把个人机器的绝对路径、访问令牌或大模型权重提交进仓库。

## 推荐工作流

1. 明确改动属于实现修复、协议修复、复现结果还是新模型，并记录预期影响。
2. 找到最接近的 dataset、collator、model、trainer 和 Gin 配置，先写清输入/输出 shape 与 ID 约定。
3. 实现最小闭环；对论文设定和本仓库取舍分别注释，不用“论文一致”代替核验清单。
4. 增加与风险相称的测试，至少覆盖改动过的协议边界或张量接口。
5. 运行格式、测试和文档检查。
6. 若产生实验数字，保留 checkpoint、日志、`results.json` 和精确源码 provenance。
7. 同步对应模型页、架构页或命令索引；代码行为改变但文档未更新，视为未完成。

### 保持改动可审阅

- 不把无关格式化、数据生成和模型改动混在同一提交中。
- 不提交 `dataset/`、`models_hub/`、`out/`、W&B 缓存或 checkpoint。
- 不手工修改历史结果来“对齐”新协议；旧结果应保留原口径并显式标记。
- 删除或重命名缓存键、checkpoint 字段、Gin selector 时，说明迁移方式和不兼容范围。
- 随机种子只是复现条件之一；同时记录 split、协议、配置、依赖产物和源码身份。

## 实现约定

### 数据与 ID

模型相关数据类放在 `genrec/data/`。新增或修改数据路径时，必须写清：

- 原始文件和缓存文件的目录契约；
- 原始 item ID、内部连续 ID、padding ID 与 SID token 的转换；
- train/valid/test 的切分单位和时间边界；
- `__getitem__` 与 collator 返回的字段、dtype、shape 和 padding 规则；
- 缓存键包含哪些会改变样本语义的参数；
- 训练集统计是否被错误地从 valid/test 泄漏。

序列推荐的 headline 结果必须注明 `leave_one_out` 或 `global_temporal`。更改候选集、负采样、窗口化或 SID→item 解析会改变协议，不能只当作性能优化。

### 模型与训练器

模型类放在 `genrec/models/`，可复用组件放在 `genrec/modules/`，入口放在 `genrec/trainers/`。当前训练器并不共享统一基类，因此新增实现应明确实现自己的调用链，同时遵守这些跨模型约定：

- `forward`、loss 和生成/打分接口的 tensor shape 可从代码和文档定位；
- 设备移动由 PyTorch/Accelerate 路径统一处理，不在 dataset 中硬编码 GPU；
- 多进程下只由主进程写日志、checkpoint 和结果，并在需要时正确 gather 全局统计；
- early stopping、best checkpoint 的选择指标和最终 test 使用的权重一致；
- SID 模型生成后先按训练集统计解析到 item，再去重并计算 item-level 指标；
- 新的外部模型、tokenizer 或缓存产物在配置和文档中列为前置依赖。

训练入口使用 `genrec.modules.utils.parse_config` 解析一个 Gin 文件及重复的 `--gin` 覆盖项。若顶层函数不是 `train`，覆盖 selector 必须使用真实函数名，例如 `train_sft`、`train_grpo`、`train_clm`、`train_rpg` 或 `train_gram`。

### 配置

默认实验配置放在 `config/<model>/...`，并应能从仓库根目录解析。提交前检查：

- 配置导入了实际使用的 configurable；
- 路径与上下游产物的保存位置相符；
- 模型 geometry 与 tokenizer checkpoint 一致；
- 数据集协议显式绑定，而不是依赖 Python 默认值；
- `{split}` 只用于本仓库解析器支持的字面替换；
- 命令示例引用存在的配置，不引用私人实验文件。

## 测试与静态检查

先运行与改动最相关的测试，再运行完整测试集。当前测试是 `tests/test_*.py` 的扁平结构；不要在文档中引用不存在的通用 `tests/test_models.py`。

```bash
# 示例：评测、SID 与结果协议
pytest -q \
  tests/test_evaluation_protocol.py \
  tests/test_honest_metrics.py \
  tests/test_sid_collisions.py

# 完整测试
pytest -q
```

格式和静态检查以 `pyproject.toml` 为准：

```bash
ruff check genrec scripts tests
black --check genrec scripts tests
mypy genrec
git diff --check
```

部分测试需要 PyTorch、NumPy、Transformers、FAISS 或本地模型；无法运行时，应在 PR 中给出具体缺失依赖和未执行范围，而不是写“测试通过”。静态检查若存在仓库既有基线问题，至少保证新增/修改文件不引入新问题，并附上实际命令与输出摘要。

改动文档时还要运行：

```bash
mkdocs build --strict
rg -n "待补充|TODO|训练中|7 种模型|8 种模型" docs/zh
```

## 结果与 provenance

日志中的单个数字不能直接成为正式 benchmark。要进入结果表，实验至少需要可定位的 `results.json`，并满足[复现工作流](architecture/reproducibility.md)和以下条件：

- `model`、`split`、`seed`、`config.evaluation_protocol` 完整；
- `extra.evaluation_level` 为 `item`；
- SID 模型记录 `extra.sid_resolution = "most_popular_training_only"`；
- `Recall@5`、`Recall@10`、`NDCG@10` 是有限的 `[0, 1]` 数值；
- `git_sha`、精确源码归档名和 SHA-256 可核验；
- 多 seed 只在相同模型、split、协议和评测层级内聚合。

推荐先验证，再聚合：

```bash
python scripts/validate_matrix_results.py out/matrix \
  --models sasrec hstu \
  --splits beauty sports toys \
  --protocols leave_one_out \
  --seeds 42 43 44

python scripts/aggregate_results.py out/matrix \
  --metrics Recall@5 Recall@10 NDCG@10 \
  --format md
```

`scripts/stamp_results_provenance.py` 只用于给已知确切来源的产物补写身份；不能用它把不明来源结果包装成正式复现。`BENCHMARKS.md` 应由通过验证的同协议产物聚合或逐项追溯，不应手工混入 SID-level、sampled-negative 或旧口径数字。

## 模型文档的完成定义

新增模型、关键变体或会改变结果语义的实现，必须同步模型页。复制[模型文档模板](models/template.md)，并完成而非保留占位文字。模型页只有在以下信息均可核验时才算完成：

- 状态卡区分“完整链路 / 组件 / 实验性 / 未实现”和“已复现 / 待重评 / 未复现”；
- 论文、官方代码与本仓库事实分别注明来源；
- 输入、输出、item 表示、关键 shape 与特殊 token 明确；
- 端到端数据流和代码映射落到真实文件、类、函数与配置；
- 从仓库根目录可复制的命令、前置产物和输出路径完整；
- checkpoint 选择、候选生成、约束、去重和指标层级明确；
- 与论文的差异、已知限制和失败证据没有被省略；
- 页面中的相对链接、配置名和脚本名在当前仓库存在；
- 最近核验日期只在实现或事实重新核验后更新。

仅有论文综述而没有本仓库状态，不算实现文档；仅有训练命令而没有协议与依赖，也不算可复现说明。未实现模型同样可以有模型页，但必须明确标为“未实现”，不能提供看似可运行的本仓库命令。

## 提交说明

PR 或交付说明应让审阅者无需猜测即可复核：

- 改了什么，以及为什么会影响或不会影响既有结果；
- 实际运行的测试、格式检查和文档构建命令；
- 未运行项目及原因；
- 数据集、模型权重和上游 checkpoint 前置条件；
- 若有结果，给出协议、seed、产物路径和源码 provenance；
- 若改变公开接口、缓存或 checkpoint，给出迁移说明。

发现代码、配置、测试和文档互相矛盾时，不要静默选择最顺口的版本。按[文档维护规范](reference/documentation-policy.md)记录冲突，并让代码行为、默认配置和状态声明可以并列核验。
