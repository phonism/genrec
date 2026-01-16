# 模块 API 参考

核心构建模块的详细文档，包括编码器、损失函数、指标等。

## 编码器模块

### TransformerEncoder

基于 Transformer 的编码器。

```python
class TransformerEncoder(nn.Module):
    """Transformer 编码器"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        attn_dim: int,
        dropout: float = 0.1,
        max_seq_length: int = 1024
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=attn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
```

**方法:**

#### forward(input_ids, attention_mask)

前向传播。

```python
def forward(
    self, 
    input_ids: torch.Tensor, 
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    前向传播
    
    Args:
        input_ids: 输入序列 (batch_size, seq_len)
        attention_mask: 注意力掩码 (batch_size, seq_len)
        
    Returns:
        编码后的序列 (batch_size, seq_len, embedding_dim)
    """
    # 嵌入和位置编码
    embeddings = self.embedding(input_ids)
    embeddings = self.pos_encoding(embeddings)
    embeddings = self.dropout(embeddings)
    
    # 创建填充掩码
    if attention_mask is not None:
        # 转换为 Transformer 期望的格式
        src_key_padding_mask = (attention_mask == 0)
    else:
        src_key_padding_mask = None
    
    # Transformer 编码
    encoded = self.transformer(
        embeddings,
        src_key_padding_mask=src_key_padding_mask
    )
    
    return encoded
```

### PositionalEncoding

位置编码模块。

```python
class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, embedding_dim: int, max_seq_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                           (-math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: 输入嵌入 (batch_size, seq_len, embedding_dim)
            
        Returns:
            添加位置编码后的嵌入
        """
        return x + self.pe[:, :x.size(1)]
```

### MultiHeadAttention

多头注意力机制。

```python
class MultiHeadAttention(nn.Module):
    """多头注意力"""
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)
        self.w_o = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        多头注意力计算
        
        Args:
            query: 查询向量 (batch_size, seq_len, embedding_dim)
            key: 键向量 (batch_size, seq_len, embedding_dim)
            value: 值向量 (batch_size, seq_len, embedding_dim)
            mask: 注意力掩码 (batch_size, seq_len, seq_len)
            
        Returns:
            (attention_output, attention_weights): 注意力输出和权重
        """
        batch_size, seq_len, _ = query.size()
        
        # 线性变换
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores.masked_fill_(mask == 0, -1e9)
        
        # 注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 注意力输出
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embedding_dim
        )
        
        output = self.w_o(attention_output)
        
        return output, attention_weights
```

## 损失函数

### VQVAELoss

VQVAE 损失函数。

```python
class VQVAELoss(nn.Module):
    """VQVAE 损失函数"""
    
    def __init__(
        self,
        commitment_cost: float = 0.25,
        beta: float = 1.0
    ):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.beta = beta
        
    def forward(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        commitment_loss: torch.Tensor,
        embedding_loss: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算 VQVAE 损失
        
        Args:
            x: 原始输入
            x_recon: 重构输出
            commitment_loss: 承诺损失
            embedding_loss: 嵌入损失
            
        Returns:
            损失字典
        """
        # 重构损失
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # 总损失
        total_loss = (
            recon_loss + 
            self.commitment_cost * commitment_loss + 
            self.beta * embedding_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'commitment_loss': commitment_loss,
            'embedding_loss': embedding_loss
        }
```

### SequenceLoss

序列建模损失函数。

```python
class SequenceLoss(nn.Module):
    """序列建模损失函数"""
    
    def __init__(
        self,
        vocab_size: int,
        ignore_index: int = -100,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=ignore_index,
                label_smoothing=label_smoothing
            )
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算序列建模损失
        
        Args:
            logits: 模型输出 (batch_size, seq_len, vocab_size)
            labels: 目标标签 (batch_size, seq_len)
            attention_mask: 注意力掩码 (batch_size, seq_len)
            
        Returns:
            损失字典
        """
        # 展平张量
        flat_logits = logits.view(-1, self.vocab_size)
        flat_labels = labels.view(-1)
        
        # 计算损失
        loss = self.criterion(flat_logits, flat_labels)
        
        # 计算准确率
        with torch.no_grad():
            predictions = torch.argmax(flat_logits, dim=-1)
            mask = (flat_labels != self.ignore_index)
            correct = (predictions == flat_labels) & mask
            accuracy = correct.sum().float() / mask.sum().float()
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
```

### ContrastiveLoss

对比学习损失函数。

```python
class ContrastiveLoss(nn.Module):
    """对比学习损失函数"""
    
    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 0.2
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            anchor: 锚点嵌入 (batch_size, embedding_dim)
            positive: 正样本嵌入 (batch_size, embedding_dim)
            negative: 负样本嵌入 (batch_size, num_negatives, embedding_dim)
            
        Returns:
            对比损失
        """
        # 标准化嵌入
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)
        
        # 计算相似度
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature
        neg_sim = torch.bmm(negative, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature
        
        # 计算对比损失
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss
```

## 评估指标

### RecommendationMetrics

推荐系统评估指标。

```python
class RecommendationMetrics:
    """推荐系统评估指标"""
    
    @staticmethod
    def recall_at_k(predictions: List[List[int]], targets: List[List[int]], k: int) -> float:
        """
        计算 Recall@K
        
        Args:
            predictions: 预测列表
            targets: 目标列表
            k: Top-K
            
        Returns:
            Recall@K 值
        """
        recall_scores = []
        
        for pred, target in zip(predictions, targets):
            if len(target) == 0:
                continue
                
            top_k_pred = set(pred[:k])
            target_set = set(target)
            
            recall = len(top_k_pred & target_set) / len(target_set)
            recall_scores.append(recall)
        
        return np.mean(recall_scores) if recall_scores else 0.0
    
    @staticmethod
    def precision_at_k(predictions: List[List[int]], targets: List[List[int]], k: int) -> float:
        """计算 Precision@K"""
        precision_scores = []
        
        for pred, target in zip(predictions, targets):
            if k == 0:
                continue
                
            top_k_pred = set(pred[:k])
            target_set = set(target)
            
            precision = len(top_k_pred & target_set) / k
            precision_scores.append(precision)
        
        return np.mean(precision_scores) if precision_scores else 0.0
    
    @staticmethod
    def ndcg_at_k(predictions: List[List[int]], targets: List[List[int]], k: int) -> float:
        """计算 NDCG@K"""
        ndcg_scores = []
        
        for pred, target in zip(predictions, targets):
            if len(target) == 0:
                continue
            
            # 计算 DCG
            dcg = 0
            for i, item in enumerate(pred[:k]):
                if item in target:
                    dcg += 1 / np.log2(i + 2)
            
            # 计算 IDCG
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(target), k)))
            
            # 计算 NDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    @staticmethod
    def hit_rate_at_k(predictions: List[List[int]], targets: List[List[int]], k: int) -> float:
        """计算 Hit Rate@K"""
        hits = 0
        total = 0
        
        for pred, target in zip(predictions, targets):
            if len(target) == 0:
                continue
                
            top_k_pred = set(pred[:k])
            target_set = set(target)
            
            if len(top_k_pred & target_set) > 0:
                hits += 1
            total += 1
        
        return hits / total if total > 0 else 0.0
    
    @staticmethod
    def coverage(predictions: List[List[int]], total_items: int) -> float:
        """计算物品覆盖度"""
        recommended_items = set()
        for pred in predictions:
            recommended_items.update(pred)
        
        return len(recommended_items) / total_items
    
    @staticmethod
    def diversity(predictions: List[List[int]]) -> float:
        """计算推荐多样性（平均 Jaccard 距离）"""
        if len(predictions) < 2:
            return 0.0
        
        distances = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                set_i = set(predictions[i])
                set_j = set(predictions[j])
                
                if len(set_i | set_j) > 0:
                    jaccard_sim = len(set_i & set_j) / len(set_i | set_j)
                    jaccard_dist = 1 - jaccard_sim
                    distances.append(jaccard_dist)
        
        return np.mean(distances) if distances else 0.0
```

## 工具模块

### AttentionVisualization

注意力可视化工具。

```python
class AttentionVisualization:
    """注意力权重可视化"""
    
    @staticmethod
    def plot_attention_heatmap(
        attention_weights: torch.Tensor,
        input_tokens: List[str],
        output_tokens: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """
        绘制注意力热力图
        
        Args:
            attention_weights: 注意力权重 (seq_len_out, seq_len_in)
            input_tokens: 输入标记列表
            output_tokens: 输出标记列表
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        
        # 创建热力图
        sns.heatmap(
            attention_weights.cpu().numpy(),
            xticklabels=input_tokens,
            yticklabels=output_tokens,
            cmap='Blues',
            annot=True,
            fmt='.2f'
        )
        
        plt.title('Attention Weights')
        plt.xlabel('Input Tokens')
        plt.ylabel('Output Tokens')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
```

### ModelUtils

模型工具函数。

```python
class ModelUtils:
    """模型工具函数"""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def get_model_size(model: nn.Module) -> str:
        """获取模型大小（以 MB 为单位）"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return f"{size_mb:.2f} MB"
    
    @staticmethod
    def freeze_layers(model: nn.Module, layer_names: List[str]) -> None:
        """冻结指定层的参数"""
        for name, param in model.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = False
                    break
    
    @staticmethod
    def unfreeze_layers(model: nn.Module, layer_names: List[str]) -> None:
        """解冻指定层的参数"""
        for name, param in model.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = True
                    break
    
    @staticmethod
    def initialize_weights(model: nn.Module, init_type: str = 'xavier') -> None:
        """初始化模型权重"""
        for name, param in model.named_parameters():
            if 'weight' in name:
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(param)
                elif init_type == 'kaiming':
                    nn.init.kaiming_uniform_(param)
                elif init_type == 'normal':
                    nn.init.normal_(param, mean=0, std=0.02)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
```

## 使用示例

### 使用编码器

```python
from genrec.modules import TransformerEncoder

# 创建编码器
encoder = TransformerEncoder(
    vocab_size=1000,
    embedding_dim=512,
    num_heads=8,
    num_layers=6,
    attn_dim=2048
)

# 编码序列
input_ids = torch.randint(0, 1000, (32, 50))  # (batch_size, seq_len)
attention_mask = torch.ones_like(input_ids)

encoded = encoder(input_ids, attention_mask)
print(f"Encoded shape: {encoded.shape}")  # (32, 50, 512)
```

### 使用损失函数

```python
from genrec.modules import VQVAELoss, SequenceLoss

# VQVAE 损失
vqvae_loss = VQVAELoss(commitment_cost=0.25)
losses = vqvae_loss(x, x_recon, commitment_loss, embedding_loss)

# 序列损失
seq_loss = SequenceLoss(vocab_size=1000, label_smoothing=0.1)
losses = seq_loss(logits, labels, attention_mask)
```

### 计算评估指标

```python
from genrec.modules import RecommendationMetrics

# 示例数据
predictions = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
targets = [[1, 3, 5], [7, 9]]

# 计算指标
recall_5 = RecommendationMetrics.recall_at_k(predictions, targets, 5)
ndcg_5 = RecommendationMetrics.ndcg_at_k(predictions, targets, 5)
hit_rate = RecommendationMetrics.hit_rate_at_k(predictions, targets, 5)

print(f"Recall@5: {recall_5:.4f}")
print(f"NDCG@5: {ndcg_5:.4f}")
print(f"Hit Rate@5: {hit_rate:.4f}")
```

### 模型工具

```python
from genrec.modules import ModelUtils

# 模型信息
param_count = ModelUtils.count_parameters(model)
model_size = ModelUtils.get_model_size(model)

print(f"Parameters: {param_count:,}")
print(f"Model size: {model_size}")

# 冻结/解冻层
ModelUtils.freeze_layers(model, ['embedding', 'pos_encoding'])
ModelUtils.unfreeze_layers(model, ['transformer'])

# 权重初始化
ModelUtils.initialize_weights(model, init_type='xavier')
```