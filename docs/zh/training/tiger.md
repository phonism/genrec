# TIGER 训练

本指南详细介绍如何训练 TIGER 模型。

## 前置条件

### 1. 预训练的 RQVAE 模型

TIGER 需要预训练的 RQVAE 模型来生成语义 ID：

```bash
# 确保 RQVAE 模型已训练完成
ls out/rqvae/p5_amazon/beauty/checkpoint_*.pt
```

如果没有，请先完成 [RQVAE 训练](rqvae.md)。

### 2. 数据准备

确保使用与 RQVAE 相同的数据集：

```bash
# 数据应该已经存在
ls dataset/amazon/
```

## 训练配置

### 默认配置

查看 TIGER 配置文件：

```bash
cat config/tiger/p5_amazon.gin
```

关键参数：

```gin
# 训练参数
train.epochs=5000               # 训练轮数
train.learning_rate=3e-4        # 学习率
train.batch_size=256            # 批量大小
train.weight_decay=0.035        # 权重衰减

# 模型参数
train.embedding_dim=128         # 嵌入维度
train.attn_dim=512             # 注意力维度
train.dropout=0.3              # Dropout率
train.num_heads=8              # 注意力头数
train.n_layers=8               # Transformer层数

# 序列参数
train.max_seq_len=512          # 最大序列长度
train.num_item_embeddings=256  # 物品嵌入数量
train.num_user_embeddings=2000 # 用户嵌入数量
train.sem_id_dim=3             # 语义ID维度

# 预训练模型路径
train.pretrained_rqvae_path="./out/rqvae/p5_amazon/beauty/checkpoint_299999.pt"
```

## 开始训练

### 基本训练命令

```bash
python genrec/trainers/tiger_trainer.py config/tiger/p5_amazon.gin
```

### 分布式训练

使用多GPU训练：

```bash
accelerate config
accelerate launch genrec/trainers/tiger_trainer.py config/tiger/p5_amazon.gin
```

### 训练过程

训练过程中会看到：

1. **数据加载**: 序列数据集加载和语义ID生成
2. **模型初始化**: Transformer模型初始化
3. **训练循环**: 损失下降和指标监控
4. **验证评估**: 周期性性能评估

## 自定义配置

### 创建自定义配置

```gin
# my_tiger_config.gin
import genrec.data.p5_amazon

# 调整模型规模
train.embedding_dim=256
train.attn_dim=1024
train.n_layers=12
train.num_heads=16

# 调整训练参数
train.learning_rate=1e-4
train.batch_size=128
train.epochs=10000

# 自定义路径
train.dataset_folder="my_dataset"
train.pretrained_rqvae_path="my_rqvae/checkpoint.pt"
train.save_dir_root="my_tiger_output/"

# 实验跟踪
train.wandb_logging=True
train.wandb_project="my_tiger_experiment"
```

## 模型架构解析

### Transformer 结构

TIGER 使用编码器-解码器架构：

```python
class Tiger(nn.Module):
    def __init__(self, config):
        # 用户和物品嵌入
        self.user_embedding = UserIdEmbedding(...)
        self.item_embedding = SemIdEmbedding(...)
        
        # Transformer 编码器-解码器
        self.transformer = TransformerEncoderDecoder(...)
        
        # 输出投影
        self.output_projection = nn.Linear(...)
```

### 语义ID映射

TIGER 将物品转换为语义ID序列：

```python
# 物品 -> 语义ID序列
item_id = 123
semantic_ids = rqvae.get_semantic_ids(item_features[item_id])
# semantic_ids: [45, 67, 89]  # 长度为 sem_id_dim
```

## 训练监控

### 关键指标

- **训练损失**: 序列建模损失
- **验证损失**: 验证集性能
- **Recall@K**: Top-K 召回率
- **NDCG@K**: 归一化折扣累积增益

### Weights & Biases 集成

启用实验跟踪：

```gin
train.wandb_logging=True
train.wandb_project="tiger_p5_amazon"
train.wandb_log_interval=100
```

查看训练曲线：
- 访问 [wandb.ai](https://wandb.ai)
- 找到你的项目和实验

## 模型评估

### 推荐质量评估

```python
from genrec.models.tiger import Tiger
from genrec.modules.metrics import TopKAccumulator

# 加载模型
model = Tiger.load_from_checkpoint("out/tiger/checkpoint.pt")

# 创建评估器
evaluator = TopKAccumulator(k=[5, 10, 20])

# 在测试集上评估
test_dataloader = DataLoader(test_dataset, batch_size=256)
metrics = evaluator.evaluate(model, test_dataloader)

print(f"Recall@10: {metrics['recall@10']:.4f}")
print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
```

### 生成式推荐

```python
def generate_recommendations(model, user_sequence, top_k=10):
    """为用户生成推荐"""
    model.eval()
    
    with torch.no_grad():
        # 编码用户序列
        sequence_embedding = model.encode_sequence(user_sequence)
        
        # 生成推荐
        logits = model.generate(sequence_embedding, max_length=top_k)
        
        # 获取Top-K物品
        recommendations = torch.topk(logits, top_k).indices
    
    return recommendations.tolist()

# 使用示例
user_history = [item1_semantic_ids, item2_semantic_ids, ...]
recommendations = generate_recommendations(model, user_history, top_k=10)
```

## 高级功能

### Trie约束生成

TIGER 支持基于Trie的约束生成：

```python
from genrec.models.tiger import build_trie

# 构建有效物品ID的Trie
valid_items = torch.tensor([[1, 2, 3], [4, 5, 6], ...])  # 语义ID序列
trie = build_trie(valid_items)

# 约束生成
constrained_output = model.generate_with_trie(
    user_sequence, 
    trie=trie,
    max_length=10
)
```

### 序列增强

训练时支持序列增强：

```gin
train.subsample=True  # 动态子采样
train.augmentation=True  # 序列增强
```

## 故障排除

### 常见问题

**Q: RQVAE检查点找不到？**

A: 检查路径是否正确：
```bash
# 确认文件存在
ls -la out/rqvae/p5_amazon/beauty/checkpoint_299999.pt

# 更新配置文件中的路径
train.pretrained_rqvae_path="实际的检查点路径"
```

**Q: 训练速度慢？**

A: 优化建议：
- 增加批量大小：`train.batch_size=512`
- 减少序列长度：`train.max_seq_len=256`
- 使用多GPU训练

**Q: 推荐效果差？**

A: 调优建议：
- 增加模型规模：`train.n_layers=12`
- 调整学习率：`train.learning_rate=1e-4`
- 增加训练轮数：`train.epochs=10000`

### 调试技巧

1. **检查语义ID生成**：
```python
# 验证RQVAE是否正常工作
rqvae = RqVae.load_from_checkpoint(pretrained_path)
sample_item = dataset[0]
semantic_ids = rqvae.get_semantic_ids(sample_item)
print(f"Semantic IDs: {semantic_ids}")
```

2. **监控注意力权重**：
```python
# 检查模型是否学到有意义的注意力模式
attention_weights = model.get_attention_weights(user_sequence)
print(f"Attention shape: {attention_weights.shape}")
```

## 性能优化

### 内存优化

```gin
# 减少内存使用
train.gradient_accumulate_every=4  # 梯度累积
train.batch_size=64               # 较小批量
train.max_seq_len=256            # 较短序列
```

### 混合精度训练

```gin
train.mixed_precision_type="fp16"  # 使用半精度
```

## 实验建议

### 超参数网格搜索

```python
# 建议的超参数范围
learning_rates = [1e-4, 3e-4, 1e-3]
batch_sizes = [128, 256, 512]
model_dims = [128, 256, 512]
n_layers = [6, 8, 12]

for lr in learning_rates:
    for bs in batch_sizes:
        # 创建配置并训练
        config = create_config(lr=lr, batch_size=bs)
        train_model(config)
```

### A/B测试

比较不同架构：

```gin
# 实验A: 标准TIGER
train.n_layers=8
train.num_heads=8

# 实验B: 更深的模型
train.n_layers=12
train.num_heads=16

# 实验C: 更宽的模型
train.embedding_dim=256
train.attn_dim=1024
```

## 下一步

训练完成后：

1. [评估推荐效果](../models/tiger.md#评估指标)
2. [部署到生产环境](../deployment.md)
3. [尝试其他数据集](../dataset/custom.md)