# TIGER API 参考

基于 Transformer 的生成式检索模型 (TIGER) 的详细 API 文档。

## 核心类

### Tiger

主要的 TIGER 模型类。

```python
class Tiger(LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        attn_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 1024,
        learning_rate: float = 1e-4
    )
```

**参数:**
- `vocab_size`: 词汇表大小
- `embedding_dim`: 嵌入维度
- `num_heads`: 注意力头数
- `num_layers`: Transformer 层数
- `attn_dim`: 注意力维度
- `dropout`: Dropout 概率
- `max_seq_length`: 最大序列长度
- `learning_rate`: 学习率

**方法:**

#### forward(input_ids, attention_mask=None)

前向传播计算。

```python
def forward(
    self, 
    input_ids: torch.Tensor, 
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Args:
        input_ids: 输入序列 (batch_size, seq_len)
        attention_mask: 注意力掩码 (batch_size, seq_len)
    
    Returns:
        logits: 输出 logits (batch_size, seq_len, vocab_size)
    """
```

#### generate(input_ids, max_length=50, temperature=1.0, top_k=None, top_p=None)

生成推荐序列。

```python
def generate(
    self,
    input_ids: torch.Tensor,
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> torch.Tensor:
    """
    Args:
        input_ids: 输入序列
        max_length: 最大生成长度
        temperature: 温度参数
        top_k: Top-k 采样
        top_p: Top-p 采样
    
    Returns:
        generated: 生成的序列
    """
```

#### generate_with_trie(input_ids, trie, max_length=50)

使用 Trie 约束生成。

```python
def generate_with_trie(
    self,
    input_ids: torch.Tensor,
    trie: TrieNode,
    max_length: int = 50
) -> torch.Tensor:
    """
    Args:
        input_ids: 输入序列
        trie: Trie 约束结构
        max_length: 最大生成长度
    
    Returns:
        generated: 约束生成的序列
    """
```

## 组件类

### TransformerBlock

Transformer 块实现。

```python
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        attn_dim: int,
        dropout: float = 0.1
    )
```

### MultiHeadAttention

多头注意力机制。

```python
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1
    )
```

### PositionalEncoding

位置编码。

```python
class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_seq_length: int = 5000
    )
```

## 数据结构

### TrieNode

Trie 节点用于约束生成。

```python
class TrieNode(defaultdict):
    def __init__(self):
        super().__init__(TrieNode)
        self.is_end = False
        
    def add_sequence(self, sequence: List[int]):
        """添加序列到 Trie"""
        node = self
        for token in sequence:
            node = node[token]
        node.is_end = True
        
    def get_valid_tokens(self) -> List[int]:
        """获取当前节点的有效 token"""
        return list(self.keys())
```

### 构建 Trie

```python
def build_trie(valid_sequences: List[List[int]]) -> TrieNode:
    """构建有效序列的 Trie"""
    root = TrieNode()
    for sequence in valid_sequences:
        root.add_sequence(sequence)
    return root
```

## 训练接口

### 训练步骤

```python
def training_step(self, batch, batch_idx):
    """训练步骤"""
    input_ids = batch['input_ids']
    labels = batch['labels']
    attention_mask = batch.get('attention_mask', None)
    
    # 前向传播
    logits = self(input_ids, attention_mask)
    
    # 计算损失
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    # 记录指标
    self.log('train_loss', loss)
    
    return loss
```

### 验证步骤

```python
def validation_step(self, batch, batch_idx):
    """验证步骤"""
    input_ids = batch['input_ids']
    labels = batch['labels']
    attention_mask = batch.get('attention_mask', None)
    
    # 前向传播
    logits = self(input_ids, attention_mask)
    
    # 计算损失
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    # 记录指标
    self.log('val_loss', loss)
    
    return loss
```

## 推理接口

### 批量生成

```python
def batch_generate(
    model: Tiger,
    input_sequences: List[torch.Tensor],
    max_length: int = 50,
    device: str = 'cuda'
) -> List[torch.Tensor]:
    """批量生成推荐"""
    model.eval()
    model.to(device)
    
    results = []
    
    with torch.no_grad():
        for input_seq in input_sequences:
            input_seq = input_seq.to(device)
            generated = model.generate(input_seq, max_length=max_length)
            results.append(generated.cpu())
    
    return results
```

### 约束生成

```python
def constrained_generate(
    model: Tiger,
    input_ids: torch.Tensor,
    valid_item_sequences: List[List[int]],
    max_length: int = 50
) -> torch.Tensor:
    """约束生成推荐"""
    # 构建 Trie
    trie = build_trie(valid_item_sequences)
    
    # 约束生成
    return model.generate_with_trie(input_ids, trie, max_length)
```

## 评估接口

### Top-K 推荐评估

```python
def evaluate_recommendation(
    model: Tiger,
    test_dataloader: DataLoader,
    k_values: List[int] = [5, 10, 20],
    device: str = 'cuda'
) -> Dict[str, float]:
    """评估推荐性能"""
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets']
            
            # 生成推荐
            generated = model.generate(input_ids, max_length=50)
            
            all_predictions.extend(generated.cpu().tolist())
            all_targets.extend(targets.tolist())
    
    # 计算指标
    metrics = {}
    for k in k_values:
        recall_k = compute_recall_at_k(all_predictions, all_targets, k)
        ndcg_k = compute_ndcg_at_k(all_predictions, all_targets, k)
        
        metrics[f'recall@{k}'] = recall_k
        metrics[f'ndcg@{k}'] = ndcg_k
    
    return metrics
```

### 困惑度评估

```python
def evaluate_perplexity(
    model: Tiger,
    test_dataloader: DataLoader,
    device: str = 'cuda'
) -> float:
    """评估困惑度"""
    model.eval()
    model.to(device)
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch.get('attention_mask', None)
            
            logits = model(input_ids, attention_mask)
            
            # 计算损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # 统计有效 token 数量
            valid_tokens = (shift_labels != -100).sum()
            
            total_loss += loss.item()
            total_tokens += valid_tokens.item()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity
```

## 工具函数

### 序列处理

```python
def pad_sequences(
    sequences: List[torch.Tensor],
    pad_token_id: int = 0,
    max_length: Optional[int] = None
) -> torch.Tensor:
    """填充序列到相同长度"""
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded = []
    for seq in sequences:
        if len(seq) < max_length:
            pad_length = max_length - len(seq)
            padded_seq = torch.cat([
                seq, 
                torch.full((pad_length,), pad_token_id, dtype=seq.dtype)
            ])
        else:
            padded_seq = seq[:max_length]
        padded.append(padded_seq)
    
    return torch.stack(padded)
```

### 采样策略

```python
def top_k_top_p_sampling(
    logits: torch.Tensor,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: float = 1.0
) -> torch.Tensor:
    """Top-k 和 Top-p 采样"""
    logits = logits / temperature
    
    # Top-k 采样
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k)
        logits[logits < values[..., [-1]]] = float('-inf')
    
    # Top-p 采样
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 找到累积概率超过 top_p 的位置
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
    
    # 采样
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, 1)
    
    return next_token
```

## 使用示例

### 基本训练

```python
from genrec.models.tiger import Tiger
from genrec.data.p5_amazon import P5AmazonSequenceDataset
import pytorch_lightning as pl

# 创建数据集
dataset = P5AmazonSequenceDataset(
    root="dataset/amazon",
    split="beauty",
    train_test_split="train",
    pretrained_rqvae_path="checkpoints/rqvae.ckpt"
)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 创建模型
model = Tiger(
    vocab_size=1024,
    embedding_dim=512,
    num_heads=8,
    num_layers=6,
    learning_rate=1e-4
)

# 训练模型
trainer = pl.Trainer(max_epochs=50, gpus=1)
trainer.fit(model, dataloader)
```

### 推荐生成

```python
# 加载训练好的模型
model = Tiger.load_from_checkpoint("checkpoints/tiger.ckpt")
model.eval()

# 用户历史序列
user_sequence = torch.tensor([10, 25, 67, 89])  # 语义ID序列

# 生成推荐
with torch.no_grad():
    recommendations = model.generate(
        user_sequence.unsqueeze(0),
        max_length=20,
        temperature=0.8,
        top_k=50
    )
    
print(f"Recommendations: {recommendations.squeeze().tolist()}")
```