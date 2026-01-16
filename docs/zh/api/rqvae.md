# RQVAE API 参考

残差量化变分自编码器 (RQVAE) 的详细 API 文档。

## 核心类

### RqVae

主要的 RQVAE 模型类。

```python
class RqVae(LightningModule):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        num_embeddings: int = 1024,
        commitment_cost: float = 0.25,
        learning_rate: float = 1e-3
    )
```

**参数:**
- `input_dim`: 输入特征维度
- `hidden_dim`: 隐藏层维度
- `latent_dim`: 潜在空间维度
- `num_embeddings`: 嵌入向量数量
- `commitment_cost`: 承诺损失权重
- `learning_rate`: 学习率

**方法:**

#### forward(features)

前向传播计算。

```python
def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        features: 输入特征 (batch_size, input_dim)
    
    Returns:
        reconstructed: 重构特征 (batch_size, input_dim)
        commitment_loss: 承诺损失
        embedding_loss: 嵌入损失
        semantic_ids: 语义ID (batch_size,)
    """
```

#### encode(features)

编码特征为潜在表示。

```python
def encode(self, features: torch.Tensor) -> torch.Tensor:
    """
    Args:
        features: 输入特征 (batch_size, input_dim)
    
    Returns:
        encoded: 编码后的潜在表示 (batch_size, latent_dim)
    """
```

#### generate_semantic_ids(features)

生成语义ID。

```python
def generate_semantic_ids(self, features: torch.Tensor) -> torch.Tensor:
    """
    Args:
        features: 输入特征 (batch_size, input_dim)
    
    Returns:
        semantic_ids: 语义ID (batch_size,)
    """
```

## 组件类

### VectorQuantizer

向量量化层实现。

```python
class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25
    )
```

**参数:**
- `num_embeddings`: 嵌入向量数量
- `embedding_dim`: 嵌入维度
- `commitment_cost`: 承诺损失权重

**方法:**

#### forward(inputs)

量化输入向量。

```python
def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        inputs: 输入向量 (batch_size, embedding_dim)
    
    Returns:
        quantized: 量化后的向量
        commitment_loss: 承诺损失
        embedding_loss: 嵌入损失
        encoding_indices: 编码索引
    """
```

### Encoder

编码器网络。

```python
class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int
    )
```

### Decoder

解码器网络。

```python
class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int
    )
```

## 训练接口

### 训练步骤

```python
def training_step(self, batch, batch_idx):
    """训练步骤"""
    features = batch['features']
    
    # 前向传播
    reconstructed, commitment_loss, embedding_loss, semantic_ids = self(features)
    
    # 计算损失
    recon_loss = F.mse_loss(reconstructed, features)
    total_loss = recon_loss + commitment_loss + embedding_loss
    
    # 记录指标
    self.log('train_loss', total_loss)
    self.log('train_recon_loss', recon_loss)
    self.log('train_commitment_loss', commitment_loss)
    self.log('train_embedding_loss', embedding_loss)
    
    return total_loss
```

### 验证步骤

```python
def validation_step(self, batch, batch_idx):
    """验证步骤"""
    features = batch['features']
    
    # 前向传播
    reconstructed, commitment_loss, embedding_loss, semantic_ids = self(features)
    
    # 计算损失
    recon_loss = F.mse_loss(reconstructed, features)
    total_loss = recon_loss + commitment_loss + embedding_loss
    
    # 记录指标
    self.log('val_loss', total_loss)
    self.log('val_recon_loss', recon_loss)
    
    return total_loss
```

## 配置接口

### 优化器配置

```python
def configure_optimizers(self):
    """配置优化器"""
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'monitor': 'val_loss'
        }
    }
```

## 工具函数

### 模型保存和加载

```python
# 保存模型
model.save_pretrained("path/to/model")

# 加载模型
model = RqVae.load_from_checkpoint("path/to/checkpoint.ckpt")
```

### 批量推理

```python
def batch_inference(model, dataloader, device='cuda'):
    """批量推理生成语义ID"""
    model.eval()
    model.to(device)
    
    all_semantic_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            semantic_ids = model.generate_semantic_ids(features)
            all_semantic_ids.append(semantic_ids.cpu())
    
    return torch.cat(all_semantic_ids, dim=0)
```

## 评估接口

### 重构质量评估

```python
def evaluate_reconstruction(model, dataloader, device='cuda'):
    """评估重构质量"""
    model.eval()
    model.to(device)
    
    total_mse = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            reconstructed, _, _, _ = model(features)
            
            mse = F.mse_loss(reconstructed, features, reduction='sum')
            total_mse += mse.item()
            total_samples += features.size(0)
    
    avg_mse = total_mse / total_samples
    return {'mse': avg_mse, 'rmse': avg_mse ** 0.5}
```

### 量化质量评估

```python
def evaluate_quantization(model, dataloader, device='cuda'):
    """评估量化质量"""
    model.eval()
    model.to(device)
    
    all_indices = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            _, _, _, semantic_ids = model(features)
            all_indices.append(semantic_ids.cpu())
    
    all_indices = torch.cat(all_indices, dim=0)
    
    # 计算使用统计
    unique_codes = len(torch.unique(all_indices))
    total_codes = model.quantizer.num_embeddings
    usage_rate = unique_codes / total_codes
    
    # 计算困惑度
    counts = torch.bincount(all_indices, minlength=total_codes).float()
    probs = counts / counts.sum()
    perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
    
    return {
        'usage_rate': usage_rate,
        'unique_codes': unique_codes,
        'perplexity': perplexity.item()
    }
```

## 使用示例

### 基本训练

```python
from genrec.models.rqvae import RqVae
from genrec.data.p5_amazon import P5AmazonItemDataset
import pytorch_lightning as pl

# 创建数据集
dataset = P5AmazonItemDataset(
    root="dataset/amazon",
    split="beauty",
    train_test_split="train"
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 创建模型
model = RqVae(
    input_dim=768,
    hidden_dim=512,
    latent_dim=256,
    num_embeddings=1024,
    learning_rate=1e-3
)

# 训练模型
trainer = pl.Trainer(max_epochs=100, gpus=1)
trainer.fit(model, dataloader)
```

### 语义ID生成

```python
# 加载训练好的模型
model = RqVae.load_from_checkpoint("checkpoints/rqvae.ckpt")
model.eval()

# 生成语义ID
with torch.no_grad():
    features = torch.randn(10, 768)  # 示例特征
    semantic_ids = model.generate_semantic_ids(features)
    print(f"Semantic IDs: {semantic_ids}")
```