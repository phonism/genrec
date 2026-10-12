# 扩展指南：增加模型、数据集和训练器

## 先确认扩展类型

当前没有插件注册器或统一基类。一个“新模型”通常要分别增加：

```text
genrec/models/<name>.py          # 数学模型和生成/打分接口
genrec/data/<dataset_name>.py    # 样本与 collator
genrec/trainers/<name>_trainer.py
config/<name>/<dataset>.gin
tests/test_<name>_*.py
```

如果只是替换 loss、encoder 或 decoding ablation，优先给现有 trainer 增加一个显式参数和独立 Gin config；不要复制整套文件后让两份逻辑漂移。

## 第一步：写清实验合同

编码前先在设计说明或 PR 中回答：

| 问题 | 必须明确的内容 |
|---|---|
| target | 下一 item、item-to-item，还是文本/lexical ID？ |
| item universe | full catalog、protocol-visible catalog，还是 sampled candidates？ |
| ID | 0-based/1-based/string；padding/ignore 是什么？ |
| history | `max_seq_len` 以 item 还是 token 计？左/右截断？ |
| representation | 原始 item ID、RQ-VAE、RQ-KMeans、OPQ、文本？ |
| decoding | 全目录打分、beam、PrefixTrie、图传播？ |
| collision | SID 如何唯一还原 item？ |
| selection | valid 哪个指标、patience、test 何时运行？ |
| artifact | checkpoint 能否恢复 optimizer；依赖哪些上游 tokenizer？ |
| protocol | LOO/global temporal 是否都支持？catalog 是否 transductive？ |

这些字段决定是否能进入公平矩阵。论文里出现一个模块不等于仓库已经具备相应数据和评估条件。

## 增加数据集适配器

### 最小要求

新类只需继承 `torch.utils.data.Dataset`；不需要虚构 `BaseDataset`。推荐：

```python
@gin.configurable
class ExampleDataset(Dataset):
    def __init__(
        self,
        root="dataset/example",
        split="beauty",
        train_test_split="train",
        max_seq_len=50,
        evaluation_protocol="leave_one_out",
        temporal_train_ratio=0.8,
        temporal_valid_ratio=0.1,
    ):
        self.evaluation_protocol = evaluation_protocol
        # 1. deterministic item mapping
        # 2. user -> sorted (timestamp, item_id)
        # 3. shared protocol
        selected, self.protocol_info = select_protocol_sequences(...)
        # 4. model-specific samples

    def __len__(self): ...
    def __getitem__(self, index): ...
```

`train_test_split` 使用 `train`/`valid`/`test`；不要把它与 catalog item embedding 的 `all`/`train`/`eval` 95/5 划分混用。

### item mapping

必须保证三个 split 和 tokenizer stage 使用完全相同的 item mapping。推荐保存可审计的 mapping artifact，至少验证：

```text
mapping size == catalog size
min/max ID 符合约定
padding 不出现在真实 item 中
embedding row / SID row / item index 一一对应
```

若沿用 first-seen 顺序，应在同一原始文件、同一过滤顺序上构造；不能在 train 和 test 分别按各自出现顺序编号。

### protocol

Amazon 风格数据优先调用 [`select_protocol_sequences`](https://github.com/phonism/genrec/blob/main/genrec/data/evaluation_protocol.py)，不要在新类里复制时间 cutoff。训练构造器要理解它返回的 placeholder 约定：既有实现通过 `seq[:-2]` 取得 protocol-visible train prefix。

若数据源无法提供 timestamp，明确只支持 LOO，并在不支持 global temporal 时抛出错误，不要悄悄退回 LOO。

### 样本与 collator

对每个 key 写 shape 注释，并区分 Dataset 单样本与 collated batch：

```python
def example_collate_fn(batch):
    return {
        "input_ids": ...,       # LongTensor[B,T], 0=padding
        "targets": ...,         # LongTensor[B]，raw catalog item
        "attention_mask": ...,  # Bool/LongTensor[B,T]
    }
```

SID 模型即使训练只需要 `[C]` semantic target，也必须在 eval 样本保留 `target_item`。否则无法做碰撞诚实的 item-level 指标。

### cache

新缓存键应尽量包含：数据版本/hash、split、encoder identity/revision、文本模板版本、维度、protocol（若拟合范围受 protocol 影响）。写缓存时复用现有模式：

```text
fcntl exclusive lock
  → 获锁后再次检查文件
  → 写 <path>.part.<pid>
  → os.replace 原子提交
```

避免把 secret、绝对临时路径或 GPU 特定对象序列化进缓存。

## 增加模型

### forward 合同

`nn.Module.forward` 应只包含可训练计算，并明确返回值：dict、NamedTuple 或 `(logits, loss)` 均可，但 trainer/test 必须固定。至少记录：

- 输入 dtype/shape；
- padding 与 attention mask；
- 输出 logits 的最后一维对应什么 vocab/item；
- loss reduction 和 ignored positions；
- 训练/评估模式差异。

例如直接 item 模型常见接口：

```python
class ExampleModel(nn.Module):
    def forward(self, input_ids, attention_mask=None, labels=None):
        # input_ids [B,T] -> hidden [B,T,D] -> logits [B,T,N+1]
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def predict(self, input_ids, attention_mask=None, top_k=10):
        # return item_ids [B,K]
        ...
```

生成式 SID 模型推荐让 inference 直接返回结构化 `LongTensor[B,K,C]`，不要让 evaluator 依赖脆弱的自由文本正则；若必须经 tokenizer，缺失/越界 code 应报诊断计数。

### checkpoint

选择一种明确格式：

- 原始 state dict：结构由 Gin 重建；
- dict checkpoint：保存 model/optimizer/scheduler/epoch 和模型几何；
- HF directory：实现 `save_pretrained`/`from_pretrained`，同时保存 tokenizer。

为可恢复训练，dict state 应在 unwrap 后保存，加载顺序为 model → optimizer → scheduler → next epoch。大模型/FSDP 参考 [`onerec_sft_trainer.py`](https://github.com/phonism/genrec/blob/main/genrec/trainers/onerec_sft_trainer.py) 的 all-rank state gathering，而不是 main-only `state_dict()`。

## 增加训练器

### 入口骨架

```python
@gin.configurable
def train_example(
    epochs=10,
    batch_size=32,
    dataset=None,
    dataset_folder="dataset/amazon",
    save_dir_root="out/example/{split}",
    seed=42,
    split_batches=True,
    amp=True,
    mixed_precision_type="bf16",
):
    run_config = dict(locals())
    logger = setup_logger(save_dir_root, "example")
    set_seed(seed)
    accelerator = setup_accelerator(
        split_batches=split_batches,
        amp=amp,
        mixed_precision_type=mixed_precision_type,
    )
    # build train/valid/test, model, optimizer, scheduler
    # prepare → train → globally reduced valid → checkpoint → test
    if accelerator.is_main_process:
        save_run_results(
            save_dir=save_dir_root,
            model="example",
            split=get_run_split(default="unknown"),
            seed=seed,
            metrics=test_metrics,
            config=run_config,
            extra={"evaluation_level": "item"},
        )
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    parse_config()
    train_example()
```

此骨架不是基类。按模型需要调整，但必须保留结果/protocol/分布式合同。

### 分布式控制流

所有 rank 必须以相同顺序进入 collective。正确 best/early-stop 模式：

1. eval 在每个 rank 本地累计 raw counts；
2. all-reduce，所有 rank 得到同一 metric；
3. 所有 rank 更新 `best`/`patience`；
4. checkpoint I/O 可 main-only，但 FSDP state gathering 可能要求 all-rank；
5. barrier；
6. 所有 rank 同时 break 或继续；
7. final test 前所有 rank 加载相同 best 权重。

不要把含 `break` 的 early stop 完全放在 `if accelerator.is_main_process` 里。

### test 使用规则

best selection 只由 validation 决定。为节约生成成本，可以只在 valid 改善时刷新 test cache，但 `results.json.metrics` 必须对应所记录的 best epoch。若 trainer 不保存 best artifact（当前 LCRec/GRPO 情况），`extra.best_epoch` 和具体 epoch checkpoint 的对应关系必须明确。

## 接入公平评估

### 直接 item 模型

将预测整理成 `[B,K,1]`，真值为 `[B,1]`：

```python
acc = TopKAccumulator([1, 5, 10])
acc.accumulate(target_items[:, None], predicted_items[:, :, None])
acc.sync_across_processes(accelerator)
metrics = acc.reduce()
```

排除 padding、query item（I2I 场景）和模型无法推荐的 sentinel。

### SID 模型

必须构建 `SemanticIdItemMapper`，并同时保留解析后 item 与解析前 SID：

```python
mapper = SemanticIdItemMapper.from_sequences(
    train_dataset.sem_ids_list,
    train_dataset.sequences,
)
pred_items = mapper.map_predictions(pred_sids, topk=10, deduplicate=True)
item_acc.accumulate(target_items[:, None], pred_items[:, :, None])
sid_acc.accumulate(target_sids, pred_sids)  # diagnostic only
```

结果 `extra` 至少包含：

```python
{
    "evaluation_level": "item",
    "sid_resolution": "most_popular_training_only",
    "sid_reference_metrics": sid_metrics,
}
```

如果研究提出新的碰撞解析策略，先增加独立策略名、无泄露测试和对照报告，不要覆盖历史策略同名结果。

## 增加 Gin 配置

1. `import` 数据与模型模块；
2. 给 trainer 顶层函数绑定所有实验关键参数；
3. 用 `@DatasetClass` 注入 class；
4. 为 Dataset 绑定 split/protocol；
5. 把 output path 写成含 `{split}` 的独立目录；
6. SID pipeline 将上游 checkpoint、encoder、C/K 和几何放在相邻段落；
7. 从仓库根执行一个极小 dry run，确认 resolved paths 和日志；
8. 不提交本机 secret 或只对自己成立的模型路径。

配置语法详见[配置系统](configuration.md)。

## 接入矩阵工具

要让 [`run_matrix.py`](https://github.com/phonism/genrec/blob/main/scripts/run_matrix.py) 识别新 recommender：

1. `REGISTRY[model_key] = (trainer_module, default_config)`；
2. 若函数不叫 `train`，加入 `TRAIN_FN`；
3. 把所有需要 protocol 覆盖的 Dataset 加入 `DATASET_CONFIGURABLES`；
4. 若 split 有专用 config，增加显式 routing，避免静默使用错误 recipe；
5. 若依赖上游 artifact，在 `Cell` 构造/`missing_dependencies()` 中验证；
6. SID mapper 模型加入 [`validate_matrix_results.py`](https://github.com/phonism/genrec/blob/main/scripts/validate_matrix_results.py) 的 `SID_RESOLVED_MODELS`；
7. 补 matrix command/protocol/dependency 单测。

量化器 stage 不宜直接塞进 recommender matrix：不同模型可能使用不兼容几何，应该先按 pipeline 显式准备和健康检查。

## 测试清单

最低覆盖：

- Dataset train/valid/test target 没有未来泄露；
- LOO 与 global temporal 的 cutoff/placeholder；
- item ID、padding、embedding row 和 SID row 对齐；
- collator 的 shape、dtype、截断边界和空/最短序列；
- model forward 有限 loss、mask 不训练 padding；
- generation 输出范围/shape，Trie 只允许 catalog path；
- SID 碰撞不会膨胀 item metric，映射后去重；
- distributed raw-count reduction（可用 fake accelerator/unit contract）；
- checkpoint round-trip，错误几何有明确失败；
- `results.json` 包含 protocol、item level、provenance 所需字段；
- matrix dry-run 命令含 seed/save dir/protocol；
- 聚合不会混合 protocol 或 evaluation level。

现有测试集中在 [`tests/`](https://github.com/phonism/genrec/tree/main/tests)：`test_evaluation_protocol.py`、`test_honest_metrics.py`、`test_notellm_full_catalog.py`、`test_rqvae_checkpoint.py`、matrix/provenance/lock tests，可作为模式参考。

## 文档更新清单

实现落地时同步：

1. 模型页：论文、公式、实现覆盖/差异；
2. 本架构章：新的文件/shape/checkpoint/评估路径；
3. 训练命令与 prerequisite；
4. benchmark protocol，不先填未经验证的数字；
5. `BENCHMARKS.md` 只接受通过严格 validator 的 artifact；
6. 如有已知限制，明确标“未实现/部分实现/实验性”，不要用论文能力替代代码事实。
