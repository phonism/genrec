# API 参考

本节是当前源码的调用索引，不是一个承诺稳定性的公共 SDK。GenRec 是研究代码库：模型、数据适配器和 trainer 多数按实验独立演进，没有统一的 `BaseModel`、`BaseDataset`、`DatasetFactory` 或 `BaseTrainer`。以下签名均以当前 `genrec/` 源文件为准。

## 从哪里开始

| 需求 | API 页 | 真实入口 |
|---|---|---|
| 查看全部数据类和样本 schema | [数据集](datasets.md) | [`genrec/data/`](https://github.com/phonism/genrec/tree/main/genrec/data) |
| 理解共享切分契约 | [数据集基础契约](base-dataset.md) | [`evaluation_protocol.py`](https://github.com/phonism/genrec/blob/main/genrec/data/evaluation_protocol.py)、[`schemas.py`](https://github.com/phonism/genrec/blob/main/genrec/data/schemas.py) |
| 配置和命令行覆盖 | [Gin 与 CLI](configs.md) | [`parse_config`](https://github.com/phonism/genrec/blob/main/genrec/modules/utils.py)、[`config/`](https://github.com/phonism/genrec/tree/main/config) |
| 模型公用层、loss、metric | [Modules](modules.md) | [`genrec/modules/`](https://github.com/phonism/genrec/tree/main/genrec/modules) |
| 数据预处理实际放在哪里 | [预处理索引](processors.md) | 数据模块内 helper、`scripts/` |
| 替代历史 DatasetFactory | [数据类选择与注入](dataset-factory.md) | Gin `dataset=@ClassName` 或直接构造 |
| 训练函数、checkpoint、results | [训练入口](trainers.md) | [`genrec/trainers/`](https://github.com/phonism/genrec/tree/main/genrec/trainers) |
| RQ-VAE 的精确返回类型 | [RQ-VAE API](rqvae.md) | [`genrec/models/rqvae.py`](https://github.com/phonism/genrec/blob/main/genrec/models/rqvae.py) |
| TIGER 的 token/beam 接口 | [TIGER API](tiger.md) | [`genrec/models/tiger.py`](https://github.com/phonism/genrec/blob/main/genrec/models/tiger.py) |

## 模型模块

仓库有 11 个 `genrec/models/*.py` 实现模块，其中 RQ-VAE 与 RQ-KMeans 是 tokenizer/quantizer，不是独立推荐器。原理、数据流和论文差异由模型页维护；API 页不再复制这些内容。

| 源模块 / 主类 | 模型说明 |
|---|---|
| [`sasrec.py`](https://github.com/phonism/genrec/blob/main/genrec/models/sasrec.py) / `SASRec` | [SASRec](../models/sasrec.md) |
| [`hstu.py`](https://github.com/phonism/genrec/blob/main/genrec/models/hstu.py) / `HSTU` | [HSTU](../models/hstu.md) |
| [`rqvae.py`](https://github.com/phonism/genrec/blob/main/genrec/models/rqvae.py) / `RqVae` | [RQVAE](../models/rqvae.md)；[API](rqvae.md) |
| [`rqkmeans.py`](https://github.com/phonism/genrec/blob/main/genrec/models/rqkmeans.py) / `RqKmeans` | [RQ-KMeans](../models/rqkmeans.md) |
| [`tiger.py`](https://github.com/phonism/genrec/blob/main/genrec/models/tiger.py) / `Tiger` | [TIGER](../models/tiger.md)；[API](tiger.md) |
| [`lcrec.py`](https://github.com/phonism/genrec/blob/main/genrec/models/lcrec.py) / `LCRec` | [LCRec](../models/lcrec.md) |
| [`cobra.py`](https://github.com/phonism/genrec/blob/main/genrec/models/cobra.py) / `Cobra` | [COBRA](../models/cobra.md) |
| [`onerec.py`](https://github.com/phonism/genrec/blob/main/genrec/models/onerec.py) / `OneRec` | [OneRec](../models/onerec.md) |
| [`rpg.py`](https://github.com/phonism/genrec/blob/main/genrec/models/rpg.py) / `RPGModel` | [RPG](../models/rpg.md) |
| [`notellm.py`](https://github.com/phonism/genrec/blob/main/genrec/models/notellm.py) / `Query2Embedding` | [NoteLLM](../models/notellm.md) |
| [`gram.py`](https://github.com/phonism/genrec/blob/main/genrec/models/gram.py) / `GRAM` | [GRAM](../models/gram.md) |

## 导入约定

[`genrec.models.__init__`](https://github.com/phonism/genrec/blob/main/genrec/models/__init__.py) 当前只重导出：

```python
from genrec.models import (
    RqVae, QuantizeForwardMode, Tiger, SASRec,
    HSTU, LCRec, Cobra, GRAM,
)
```

OneRec、RPG、NoteLLM 和 RQ-KMeans 没有从包根重导出，应从实现模块导入：

```python
from genrec.models.onerec import OneRec
from genrec.models.rpg import RPGModel
from genrec.models.notellm import Query2Embedding
from genrec.models.rqkmeans import RqKmeans
```

[`genrec.data`](https://github.com/phonism/genrec/blob/main/genrec/data/__init__.py) 只重导出 `SeqData`、`SeqBatch`、`TokenizedSeqBatch` 和 `cycle`；各 dataset 也应从具体模块导入。`genrec.trainers` 重导出少量共享工具，但不重导出每个 `train*` 函数。

## 当前稳定的跨模型契约

跨模型真正共享的接口只有少数几项：

- PyTorch dataset：实现 `__len__()` 与 `__getitem__()`，样本 schema 由对应 trainer 的 collate 函数决定。
- CLI：trainer 的 `__main__` 调用 `parse_config()`，接受一个 gin 文件、`--split` 和重复的 `--gin`。
- 评测：多数 item/SID 路径复用 `TopKAccumulator`；SID 模型可复用 `SemanticIdItemMapper`。
- 结果：多数 trainer 调用 `save_run_results(...)` 写 `results.json`，但 checkpoint 文件名和格式并不统一。

不要假定“某个 trainer 支持的参数，其他 trainer 也支持”。调用前应同时看目标 trainer 的 `@gin.configurable` 函数签名和对应 `.gin` 文件。

## 符号核验方法

本地不必导入 PyTorch 即可用静态搜索确认符号：

```bash
rg -n '^class |^def |^@gin.configurable' genrec/models genrec/data genrec/modules genrec/trainers
rg -n 'train\.|train_[a-z]+\.' config
```

需要运行时签名且环境已安装依赖时：

```python
import inspect
from genrec.models.rqvae import RqVae

print(inspect.signature(RqVae))
print(inspect.signature(RqVae.get_semantic_ids))
```

## 相关文档

- [模型总览与选型](../models/overview.md)
- [项目架构](../architecture/index.md)
- [模型实现状态](../reference/model-status.md)
