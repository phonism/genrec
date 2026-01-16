# P5 Amazon 数据集

P5 Amazon 数据集是一个大规模产品推荐数据集，包含来自 Amazon 的用户评论和产品元数据。

## 概述

P5 Amazon 来源于 Amazon 产品数据，包含丰富的产品文本信息以及用户交互历史。它专门为训练生成式推荐模型而设计。

**主要特点：**
- 多个产品类别（美容、电子产品、运动等）
- 丰富的产品元数据（标题、品牌、类别、价格、描述）
- 带时间戳的用户交互序列
- 适合神经模型的预处理文本特征

## 数据集结构

### 数据文件

下载后，数据集包含：
```
dataset/amazon/
├── raw/
│   ├── beauty_5.json.gz          # 原始交互数据
│   ├── meta_beauty.json.gz       # 产品元数据
│   └── ...
├── processed/
│   ├── items.parquet             # 处理后的物品特征
│   ├── interactions.parquet      # 处理后的交互数据
│   └── mappings.pkl              # ID 映射
└── cache/
    └── text_embeddings/          # 缓存的文本嵌入
```

### 数据格式

**物品数据框：**
| 列名 | 类型 | 描述 |
|------|------|------|
| item_id | int | 唯一物品标识符 |
| title | str | 产品标题 |
| brand | str | 产品品牌 |
| category | str | 产品类别 |
| price | float | 产品价格 |
| features | List[float] | 文本嵌入特征（768维） |
| text | str | 格式化文本（用于参考） |

**交互数据框：**
| 列名 | 类型 | 描述 |
|------|------|------|
| user_id | int | 唯一用户标识符 |
| item_id | int | 物品标识符 |
| rating | float | 用户评分（1-5） |
| timestamp | int | 交互时间戳 |

## 可用类别

### 美容 (Beauty)
- **大小**: ~52K 产品，~1.2M 交互
- **描述**: 化妆品、护肤品、护发产品
- **特点**: 丰富的品牌和类别信息

### 电子产品 (Electronics)
- **大小**: ~63K 产品，~1.7M 交互
- **描述**: 电子设备、配件、小工具
- **特点**: 描述中包含技术规格

### 运动 (Sports)
- **大小**: ~35K 产品，~296K 交互
- **描述**: 运动设备、户外装备、健身产品
- **特点**: 活动特定的分类

### 所有类别
- **总计**: 29 个可用类别
- **合并大小**: 多 GB 数据集
- **用途**: 大规模实验

## 使用方法

### 基本加载

```python
from genrec.data.p5_amazon import P5AmazonItemDataset

# 加载美容类别用于物品级训练
dataset = P5AmazonItemDataset(
    root="dataset/amazon",
    split="beauty",
    train_test_split="train"
)

print(f"数据集大小: {len(dataset)}")
print(f"特征维度: {dataset[0].shape}")
```

### 序列数据

```python
from genrec.data.p5_amazon import P5AmazonSequenceDataset

# 加载用于序列建模（需要预训练的 RQVAE）
seq_dataset = P5AmazonSequenceDataset(
    root="dataset/amazon", 
    split="beauty",
    train_test_split="train",
    pretrained_rqvae_path="checkpoints/rqvae_beauty.ckpt"
)

# 获取样本序列
sample = seq_dataset[0]
print(f"输入序列: {sample['input_ids']}")
print(f"目标序列: {sample['labels']}")
```

### 配置选项

```python
from genrec.data.p5_amazon import P5AmazonItemDataset

dataset = P5AmazonItemDataset(
    root="dataset/amazon",
    split="beauty",
    train_test_split="train",
    
    # 文本编码选项
    encoder_model_name="sentence-transformers/all-MiniLM-L6-v2",
    force_reload=False,  # 如果可用，使用缓存的嵌入
    
    # 数据过滤
    min_user_interactions=5,
    min_item_interactions=5,
    
    # 文本模板
    text_template="标题: {title}; 品牌: {brand}; 类别: {category}"
)
```

## 数据处理流水线

### 1. 下载和解压

```python
# 从 UCSD 仓库自动下载
dataset = P5AmazonItemDataset(root="dataset/amazon", split="beauty")
# 美容类别下载约 500MB
```

### 2. 文本处理

框架使用句子变换器自动处理产品文本：

```python
# 默认文本模板
template = "Title: {title}; Brand: {brand}; Category: {category}; Price: {price}"

# 处理后的文本示例
"Title: Maybelline Mascara; Brand: Maybelline; Category: Beauty; Price: $8.99"
```

### 3. 特征提取

- **文本嵌入**: 来自句子变换器的 768 维向量
- **缓存**: 嵌入被缓存以加快后续加载
- **标准化**: 默认应用 L2 标准化

### 4. 序列构建

对于 TIGER 训练，交互被转换为序列：

```python
# 用户交互历史
user_history = [item1, item2, item3, item4]

# 使用 RQVAE 转换为语义 ID 序列
semantic_sequence = [1, 45, 123, 67, 234, 189, 45, 123, 567, 234, 88, 192]
#                   |--item1--| |--item2--| |--item3--| |--item4--|
```

## 统计信息

### 美容类别
```
物品数: 52,024
用户数: 40,226  
交互数: 1,235,316
密度: 0.059%
每用户平均物品数: 30.7
每物品平均用户数: 23.7
```

### 电子产品类别
```
物品数: 63,001
用户数: 192,403
交互数: 1,689,188
密度: 0.014%
每用户平均物品数: 8.8
每物品平均用户数: 26.8
```

## 数据质量

### 预处理步骤

1. **去重**: 移除重复的用户-物品交互
2. **低活跃度过滤**: 过滤交互少于 5 次的用户/物品
3. **文本清洗**: 标准化标题，处理缺失的品牌/类别
4. **价格处理**: 清洗和标准化价格格式
5. **ID 重映射**: 创建连续的 ID 映射

### 质量检查

```python
from genrec.data.p5_amazon import P5AmazonItemDataset

dataset = P5AmazonItemDataset(root="dataset/amazon", split="beauty")

# 检查数据质量
items_df, interactions_df = dataset.base_dataset.get_dataset()

print("数据质量报告:")
print(f"缺少标题的物品: {items_df['title'].isna().sum()}")
print(f"缺少品牌的物品: {items_df['brand'].isna().sum()}")
print(f"有效评分的交互: {(interactions_df['rating'] > 0).sum()}")
print(f"特征向量维度: {len(items_df.iloc[0]['features'])}")
```

## 高级用法

### 自定义文本模板

```python
# 以产品为中心的模板
template = "产品: {brand} 的 {title}，属于 {category} 类别"

# 价格感知模板  
template = "购买 {brand} 的 {title}，价格 ${price}，属于 {category}"

# 简化模板
template = "{title} - {brand}"
```

### 批处理

```python
from torch.utils.data import DataLoader

dataset = P5AmazonItemDataset(root="dataset/amazon", split="beauty")
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

for batch in dataloader:
    # batch 形状: (128, 768)
    features = batch
    # 处理批次...
```

### 多类别加载

```python
# 加载多个类别
categories = ["beauty", "electronics", "sports"]
datasets = []

for category in categories:
    dataset = P5AmazonItemDataset(
        root="dataset/amazon",
        split=category,
        train_test_split="train"
    )
    datasets.append(dataset)

# 合并数据集
from torch.utils.data import ConcatDataset
combined_dataset = ConcatDataset(datasets)
```

## 故障排除

### 常见问题

**Q: 下载失败，出现网络错误**
A: 检查网络连接并重试。文件很大（100MB-2GB）。

**Q: 文本编码耗时很长**
A: 设置 `force_reload=False` 使用缓存的嵌入，并确保缓存目录可写。

**Q: 加载时内存不足**
A: 减少批次大小或使用较小的类别如 "beauty" 而不是 "all"。

**Q: 缺少品牌/类别信息**
A: 这是正常的 - 数据集用 "Unknown" 填充缺失值。

### 性能提示

```python
# 使用缓存加快后续加载
dataset = P5AmazonItemDataset(
    root="dataset/amazon",
    split="beauty", 
    force_reload=False  # 使用缓存
)

# 使用更轻的文本编码器加快处理
dataset = P5AmazonItemDataset(
    root="dataset/amazon",
    split="beauty",
    encoder_model_name="sentence-transformers/all-MiniLM-L6-v2"  # 更小的模型
)

# 使用更小的批次处理
from genrec.data.configs import TextEncodingConfig
text_config = TextEncodingConfig(batch_size=16)  # 从默认的 32 减少
```

## 引用

如果您使用 P5 Amazon 数据集，请引用：

```bibtex
@article{geng2022recommendation,
  title={Recommendation as language processing (rlp): A unified pretrain, personalized prompt \& predict paradigm (p5)},
  author={Geng, Shijie and Liu, Shuchang and Fu, Zuohui and Ge, Yingqiang and Zhang, Yongfeng},
  journal={arXiv preprint arXiv:2203.13366},
  year={2022}
}
```

## 相关文档

- [数据集概述](overview.md) - 通用数据集概念
- [自定义数据集](custom.md) - 创建您自己的数据集  
- [RQVAE 训练](../training/rqvae.md) - 训练物品编码器
- [TIGER 训练](../training/tiger.md) - 训练序列模型