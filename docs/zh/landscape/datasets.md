# 数据集与划分建议

生成式推荐的数据不只是交互三元组。内容 encoder、SID tokenizer、目录约束和碰撞解析都可能看见额外信息，因此必须记录**哪一版数据、哪些字段、在哪个时间点可见**。

## 数据集目录

| 数据集 | 内容与规模特征 | 适合回答的问题 | 主要风险 | 本仓库状态 | 官方来源 |
|---|---|---|---|---|---|
| Amazon Reviews 2014 5-core | 旧版全量含 142.8M reviews；类别 5-core 提供 rating、review text、metadata、时间和关系图 | 与 TIGER/LC-Rec/扩散论文对齐；小目录 tokenizer 消融 | 不同论文的去重、5-core、标题清洗和 LOO 实现常不同；它不是 2023 版的子集 | ✅ 主数据集；常用 `beauty/sports/toys/clothing`，多数模型 adapter 已有 | [UCSD 官方页](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) |
| Amazon Reviews 2023 | 571.54M reviews、33 域、时间到 2023-09；更丰富 metadata、原图/链接和官方处理说明 | 大目录、跨域、冷物品、多模态、LLM 文本实验 | 0-core/5-core和标准 split 不同；metadata 缺失；下载体积大 | 🧪 数据准备脚本、SASRec/HSTU raw-adapter 配置和 OneRec adapter 已存在；`amazon2023.py` 内独立 Dataset 类与 CLI 输出目录尚未闭环 | [项目页](https://amazon-reviews-2023.github.io/main.html) / [Hugging Face](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) |
| MovieLens 1M | 约 1M ratings，稳定的用户/电影 ID、时间、genre | 小规模 item-level sanity check、HSTU/扩散快速回归 | 内容弱、目录小、正反馈定义不统一；不能代表开放目录 | 🧪 仅有 HSTU adapter，尚无统一多模型配置 | [GroupLens](https://grouplens.org/datasets/movielens/1m/) |
| Yelp Open Dataset | review、business、照片、check-in、属性；官方页面当前为约 6.99M reviews、150K businesses | 地点/文本/长尾、多模态和 DiffRec 系对照 | 教育用途条款；版本会更新，城市与过滤差异大 | ⬜ 未实现 | [Yelp 官方页](https://business.yelp.com/data/resources/open-dataset/) |
| Steam | 用户–游戏 review/playtime 和游戏 metadata | 游戏跨域、冷启动；LIGER/RecGPT 常用域 | UCSD 有多个 Steam 版本；bundle、免费游戏和 playtime 清洗差异明显 | ⬜ 未实现 | [UCSD 数据页](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data) |
| MIND | 约 160K 英文新闻、1M 用户、15M+ impression；标题、摘要、正文、类别、实体 | 动态目录、严格时间切分、文本生成与 candidate reranking | 官方任务是 impression candidate ranking，不是默认 full-catalog next-item；研究许可 | ⬜ 未实现 | [Microsoft 官方页](https://msnews.github.io/) / [论文](https://aclanthology.org/2020.acl-main.331/) |
| Goodreads / UCSD Book Graph | 2.36M books、约 229M shelf interactions，并有 review、author、series、genre | 大目录文本 SID、图关系、跨 genre | 仅学术用途、禁止再分发/商业使用；genre 子集重叠 | ⬜ 未实现 | [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html) |
| KuaiRec | 快手短视频；小矩阵接近全观测，另有普通日志、视频类别和 caption | 去偏/OPE、RL、反馈闭环、多兴趣 | fully observed 子集很小且暴露机制特殊；不能替代一般离线日志 | ⬜ 未实现 | [官方仓库](https://github.com/chongminggao/KuaiRec) / [论文](https://arxiv.org/abs/2202.10842) |
| KuaiRand | 随机曝光日志、watch ratio、多种反馈和视频特征 | 无偏评测、policy learning、RL reward | 与标准 leave-one-out Recall 不可直接比较；需处理 propensity | ⬜ 未实现 | [项目页](https://kuairand.com/) |

表中的规模只帮助选数据，不应复制成实验统计。每次预处理后都要从实际文件重新输出 users/items/interactions、时间范围、密度、文本缺失率和过滤后规模。

Amazon 2023 的“部分支持”是具体的数据契约缺口：当前 CLI 写入 `dataset/amazon2023/raw/<category>`，而同文件中的 `Amazon2023ItemDataset` / `Amazon2023SASRecDataset` 读取 `processed/<split>`。在这两条路径统一并加入端到端测试前，不能把它称为通用 loader 已闭环。

## 本仓库的推荐主矩阵

短期内最可靠的公平矩阵不是“所有模型跑所有数据”，而是分层推进：

| 层级 | 数据 | 目的 | 模型 |
|---|---|---|---|
| A：历史可比 | Amazon 2014 Beauty/Sports/Toys | 对齐公开工作、跑多 seed、完成物品级纠正 | SASRec、HSTU、TIGER、LC-Rec、COBRA、RPG、OneRec |
| B：规模与域 | Amazon 2023 的 3 个中型域 | 更大目录、更新文本、跨域/冷物品 | 先 SASRec/HSTU + 两个 SID 模型，再扩展 |
| C：协议外推 | MovieLens-1M + Yelp 或 Steam | 检查结论是否只在 Amazon 成立 | 轻模型、DiffRec/DreamRec、LIGER/RecGPT |
| D：动态/RL | MIND 或 KuaiRec/KuaiRand | 动态目录、候选重排、policy/reward | 仅当任务接口独立后加入 |

GRAM 官方实现使用专门准备的 Amazon 文件和大小写 split；它不应在没有转换清单时假装与仓库统一 Amazon adapter 相同。

## Amazon 2014 与 2023 不能混用

两版的时间范围、字段、类别名、原始 item/user universe、去重和 core 处理均不同。例如旧版 `Beauty` 5-core 与新版 `All_Beauty` 或 `Beauty_and_Personal_Care` 不是同一 split。输出目录、缓存键和 W&B run name 必须包含：

```text
source_version/category/core/filter/split_protocol
```

推荐示例：`amazon2014/beauty/5core/loo`、`amazon2023/all_beauty/5core/absolute_time`。只写 `amazon/beauty` 不足以复现实验。

## 内容字段与泄漏

SID 路线常把 item metadata 当“静态知识”，但时间实验中它也有可见性问题：

- 标题/描述是否来自交互发生时的快照，还是 2023 抓取的最终页面；
- sales rank、also-bought、also-viewed 是否包含未来行为；
- review text 是否包含 test 期反馈；
- 预训练文本 encoder 是否可能见过同一商品页面；
- 新 item 的内容在它进入目录前是否可用。

最低要求是为每个字段标注 `train-only`、`catalog-time available` 或 `global-static assumption`。后两者不是错误，但必须诚实命名，不能声称严格无泄漏。

## 建议的数据工件

每个数据版本应产生不可变 manifest：

```yaml
dataset: amazon2014
category: beauty
raw_sha256: ...
filter: 5core_source
split: leave_one_out
min_history: 3
max_seq_len: 50
num_users: ...
num_items_train: ...
num_items_eval_catalog: ...
num_interactions: ...
time_min: ...
time_max: ...
metadata_fields: [title, description, categories]
metadata_missing_rate: ...
id_map_sha256: ...
```

并保存：

1. `user_map`、`item_map` 的构造范围；
2. train/valid/test 每个样本的最后时间戳；
3. seen/unseen item 与 token/prefix 冷启动分桶；
4. metadata 文本规范化版本；
5. tokenizer 输入 item 列表和 checkpoint hash；
6. license/terms URL 与下载日期。

## 新数据接入验收

一个 adapter 只有满足以下条件才算“支持”：

- 在离线/缓存命中两种路径产生相同 manifest；
- 相同 seed 的 split 和 ID map byte-identical；
- train-only 统计不会读取 validation/test；
- 至少一个 item-level baseline 端到端跑通；
- 能输出碰撞、invalid、cold bucket 等诊断；
- 单元测试覆盖重复时间戳、缺 metadata、短用户和新 item；
- 文档记录下载/许可，仓库不重新分发禁止分发的数据。
