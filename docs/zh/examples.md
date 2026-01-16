# 示例代码

本页面包含使用 genrec 的实用示例。

## 基础使用示例

### 从零开始训练 RQVAE

```python
import torch
from genrec.models.rqvae import RqVae, QuantizeForwardMode
from genrec.data.p5_amazon import P5AmazonItemDataset
from torch.utils.data import DataLoader

# 创建数据集
dataset = P5AmazonItemDataset(
    root="dataset/amazon",
    split="beauty",
    train_test_split="train"
)

# 创建模型
model = RqVae(
    input_dim=768,
    embed_dim=32,
    hidden_dims=[512, 256, 128],
    codebook_size=256,
    n_layers=3,
    commitment_weight=0.25,
    codebook_mode=QuantizeForwardMode.ROTATION_TRICK
)

# 训练循环
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

for epoch in range(100):
    for batch in dataloader:
        optimizer.zero_grad()
        
        outputs = model(torch.tensor(batch))
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### 使用数据集工厂

```python
from genrec.data.dataset_factory import DatasetFactory

# 创建物品数据集
item_dataset = DatasetFactory.create_item_dataset(
    "p5_amazon",
    "dataset/amazon",
    split="train"
)

# 创建序列数据集
sequence_dataset = DatasetFactory.create_sequence_dataset(
    "p5_amazon", 
    "dataset/amazon",
    split="train",
    pretrained_rqvae_path="./checkpoints/rqvae.pt"
)
```

### 自定义配置

```python
from genrec.data.configs import P5AmazonConfig, TextEncodingConfig

# 自定义文本编码配置
text_config = TextEncodingConfig(
    encoder_model="sentence-transformers/all-MiniLM-L6-v2",
    template="产品: {title} | 品牌: {brand} | 类别: {categories}",
    batch_size=32
)

# 自定义数据集配置
dataset_config = P5AmazonConfig(
    root_dir="my_data",
    split="electronics",
    text_config=text_config
)
```

## 高级示例

### 多 GPU 训练

```python
from accelerate import Accelerator

def train_with_accelerate():
    accelerator = Accelerator()
    
    # 模型、优化器、数据加载器
    model = RqVae(...)
    optimizer = torch.optim.AdamW(model.parameters())
    dataloader = DataLoader(...)
    
    # 准备分布式训练
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            
            with accelerator.autocast():
                outputs = model(batch)
                loss = outputs.loss
                
            accelerator.backward(loss)
            optimizer.step()
```

### 自定义数据集实现

```python
from genrec.data.base_dataset import BaseRecommenderDataset

class MyCustomDataset(BaseRecommenderDataset):
    def download(self):
        # 实现数据下载逻辑
        pass
        
    def load_raw_data(self):
        # 加载原始数据文件
        return {"items": items_df, "interactions": interactions_df}
        
    def preprocess_data(self, raw_data):
        # 自定义预处理
        return processed_data
        
    def extract_items(self, processed_data):
        return processed_data["items"]
        
    def extract_interactions(self, processed_data):
        return processed_data["interactions"]
```

## 集成示例

### Weights & Biases 集成

```python
import wandb

# 初始化 wandb
wandb.init(
    project="my-recommendation-project",
    config={
        "learning_rate": 0.0005,
        "batch_size": 64,
        "model_type": "rqvae"
    }
)

# 训练过程中记录指标
for epoch in range(epochs):
    # ... 训练代码 ...
    
    wandb.log({
        "epoch": epoch,
        "loss": loss.item(),
        "reconstruction_loss": recon_loss.item(),
        "quantization_loss": quant_loss.item()
    })
```

### 超参数调优

```python
import optuna

def objective(trial):
    # 建议超参数
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    embed_dim = trial.suggest_categorical("embed_dim", [16, 32, 64])
    
    # 使用建议的参数训练模型
    model = RqVae(embed_dim=embed_dim, ...)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # 训练循环
    val_loss = train_and_evaluate(model, optimizer, batch_size)
    
    return val_loss

# 运行优化
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
```

## 评估示例

### 模型评估

```python
def evaluate_model(model, test_dataloader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            
            total_loss += outputs.loss.item() * len(batch)
            total_samples += len(batch)
    
    return total_loss / total_samples

# 评估 RQVAE
test_loss = evaluate_model(rqvae_model, test_dataloader, device)
print(f"测试重构损失: {test_loss:.4f}")
```

### 推荐生成

```python
def generate_recommendations(tiger_model, user_sequence, top_k=10):
    """为用户序列生成 Top-K 推荐"""
    tiger_model.eval()
    
    with torch.no_grad():
        # 编码用户序列
        logits = tiger_model.generate(user_sequence, max_length=top_k)
        
        # 获取 Top-K 物品
        top_items = torch.topk(logits, top_k).indices
        
    return top_items.tolist()

# 生成推荐
user_seq = [1, 5, 23, 45]  # 用户交互历史
recommendations = generate_recommendations(tiger_model, user_seq, top_k=10)
print(f"推荐物品: {recommendations}")
```

## 实用工具

### 数据分析

```python
from genrec.data.processors.sequence_processor import SequenceStatistics

# 分析序列统计信息
stats = SequenceStatistics.compute_sequence_stats(sequence_data)
print(f"平均序列长度: {stats['avg_seq_length']:.2f}")
print(f"唯一物品数量: {stats['num_unique_items']}")
```

### 模型检查

```python
def inspect_codebook_usage(rqvae_model, dataloader):
    """分析码本利用率"""
    used_codes = set()
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = rqvae_model(batch)
            semantic_ids = outputs.sem_ids
            used_codes.update(semantic_ids.flatten().tolist())
    
    usage_rate = len(used_codes) / rqvae_model.codebook_size
    print(f"码本利用率: {usage_rate:.2%}")
    
    return used_codes

used_codes = inspect_codebook_usage(model, dataloader)
```

## 技巧和最佳实践

### 内存优化

```python
# 为大型模型启用梯度检查点
model.gradient_checkpointing_enable()

# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(batch)
        loss = outputs.loss
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 调试

```python
# 启用详细日志
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 记录模型统计信息
def log_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,}")
```