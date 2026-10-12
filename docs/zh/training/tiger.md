# 训练 TIGER

TIGER 训练分为 RQ-VAE tokenizer 和 T5 推荐器两阶段。当前 trainer 为单进程设备路径，默认普通 beam search；不要照搬旧文档中的 Accelerate 多卡或 Trie 命令。

## 前置检查

```bash
test -f dataset/amazon/raw/beauty/reviews_Beauty_5.json.gz
test -f dataset/amazon/processed/beauty/item_emb_sentence-t5-xl.parquet
test -f out/tiger/amazon/beauty/rqvae/checkpoint_epoch_4999.pt
```

确认 RQ-VAE 是 TIGER 的 3×256、latent 32、hidden `[512,256,128,64]` 配置，并由同一 sentence-t5-xl cache 训练。

## Stage 1：tokenizer

```bash
python genrec/trainers/rqvae_trainer.py \
  config/tiger/amazon/rqvae.gin --split beauty
```

完成后先跑 codebook health 与 collision 检查，见 [RQ-VAE 训练](rqvae.md)。

## Stage 2：TIGER

```bash
python genrec/trainers/tiger_trainer.py \
  config/tiger/amazon/tiger.gin --split beauty \
  --gin 'train.wandb_logging=False'
```

当前默认：200 epochs、batch 256、Adam lr `1e-4`、50 item history、30 beams、valid Recall@10 patience 10。

如果 checkpoint 不在默认位置：

```bash
python genrec/trainers/tiger_trainer.py \
  config/tiger/amazon/tiger.gin --split beauty \
  --gin 'train.pretrained_rqvae_path="/abs/path/checkpoint.pt"'
```

## 训练样本与 token 数

每个 train user 的序列先去掉 valid/test 两项，再生成所有非空 prefix → next-item 样本。3-code SID 下：

```text
50 history items × 3 codes = 150 encoder tokens
target = 3 decoder labels
vocab = 3 × 256 + padding = 769
```

`t5_collate` 左 padding encoder token，并按 codebook position 加 offset。不要把 `max_seq_len` 当 token 数传给 `max_history_tokens`。

## 评测流程

每次 valid：

1. 生成 `beam_size` 个长度为 C 的 token 序列；
2. 去掉 decoder-start token，反解 position offset；
3. 记录 SID exact-match 诊断；
4. 用训练前缀流行度将每个 SID 解析到唯一 item；
5. 丢弃非法 SID、对 item 去重；
6. 计算 item-level Recall@5/10、NDCG@5/10；
7. 按 item Recall@10 选择 `best_model.pt`。

测试集在 valid 提升时运行，另可按 `eval_test_every_epoch` 周期诊断。`results.json` 的主 metrics 对应最佳 valid checkpoint 的 item-level test 结果。

## 当前没有硬约束解码

`Tiger.generate` 没有 Trie/prefix callback。要测候选有效率，可在 eval 中统计：

- 每个生成位置是否落在该 codebook 的 offset 区间；
- raw SID 是否存在于 catalog；
- beam 经解析和 item 去重后剩余多少候选。

增大 beam 能给后过滤更多候选，但不等价于修复非法生成，也会线性增加候选输出和更多 decoder 计算。

## 输出

```text
out/tiger/amazon/beauty/
├── best_model.pt          # model.state_dict only
├── tiger_<timestamp>.log
└── results.json
```

加载 `best_model.pt` 时要按 Gin 参数先构造相同 `Tiger(model_config)`；它不是 Hugging Face `save_pretrained` 目录。

## 可比实验规则

- `AmazonSeqDataset.add_disambiguation=False` 是当前默认；改变后是 tokenizer/ID 设计消融；
- `dedup_sinkhorn` 与 suffix 不应同时使用；
- `sem_id_dim`、codebook size、RQ-VAE 层数同时变化时不能归因于单一变量；
- SID-level 历史数字不进入 headline；
- 与 SASRec/HSTU 对比时统一 `max_seq_len=50`、split、seed 与 item-level catalog。

## 排错

- **RQ-VAE state shape mismatch**：下游 dataset 手动构造 RQ-VAE；检查五个几何参数，不要 `strict=False` 掩盖。
- **beam 输出 shape 错**：generation 预期 `[B*beam,C+1]`；第一个 token 是 start，再 reshape。
- **item metrics 全零但 SID 有命中**：检查 `target_item` 是否保留、item mapping 行序和 training-only resolver。
- **有效候选不足 10**：统计非法 SID 和 dedup 比率；mapper 会用 `-1` padding，不会把缺位当命中。
- **多 GPU 没有加速**：当前 TIGER trainer 没有 Accelerate/DDP 封装；需要单独实现和验证 distributed path。
- **旧文档代码找不到 `build_trie`**：`genrec.models.tiger` 没有这个函数。当前状态见 [TIGER 模型页](../models/tiger.md)。

## 推荐验证

```bash
pytest -q tests/test_honest_metrics.py tests/test_rqvae_checkpoint.py
python scripts/verify_t5_equivalence.py
python scripts/verify_pure_t5.py
```

后两个文件没有参数解析或 `--help` 模式；执行即会构造 Hugging Face 与本地 T5 并完成整次数值对齐检查，需要已安装 `transformers` 和 `torch`。它们是实现 parity 诊断，不是训练前必跑的轻量命令。

详细架构、offset 公式和实现差异见 [TIGER](../models/tiger.md)。
