# TIGER 模型

TIGER (Recommender Systems with Generative Retrieval) 是一种基于 Transformer 的生成式推荐模型，通过序列建模的方式进行物品推荐。

## 模型架构

### 核心思想

TIGER 将推荐问题转化为序列生成问题：
- 输入：用户历史交互序列（语义ID表示）
- 输出：推荐物品的语义ID序列

```mermaid
graph TD
    A[用户历史序列] --> B[语义ID编码]
    B --> C[Transformer编码器]
    C --> D[Transformer解码器]
    D --> E[生成推荐序列]
    E --> F[语义ID解码]
    F --> G[推荐物品]
```

### 模型组件

#### 1. 嵌入层

```python
class SemIdEmbedding(nn.Module):
    """语义ID嵌入层"""
    def __init__(self, num_embeddings, embedding_dim, sem_id_dim=3):
        super().__init__()
        self.sem_id_dim = sem_id_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
    def forward(self, sem_ids):
        # sem_ids: (batch_size, seq_len, sem_id_dim)
        embedded = self.embedding(sem_ids)
        return embedded.mean(dim=-2)  # 聚合语义ID维度
```

#### 2. Transformer 架构

```python
class TransformerEncoderDecoder(nn.Module):
    """Transformer 编码器-解码器"""
    def __init__(self, config):
        super().__init__()
        
        # 编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.attn_dim,
            dropout=config.dropout
        )
        self.encoder = TransformerEncoder(encoder_layer, config.n_layers)
        
        # 解码器  
        decoder_layer = TransformerDecoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.attn_dim,
            dropout=config.dropout
        )
        self.decoder = TransformerDecoder(decoder_layer, config.n_layers)
```

#### 3. 约束生成

TIGER 使用 Trie 数据结构约束生成过程：

```python
class TrieNode(defaultdict):
    """Trie 节点"""
    def __init__(self):
        super().__init__(TrieNode)
        self.is_end = False

def build_trie(valid_item_ids):
    """构建有效物品ID的Trie"""
    root = TrieNode()
    for seq in valid_item_ids.tolist():
        node = root
        for token in seq:
            node = node[token]
        node.is_end = True
    return root
```

## 语义ID映射

### 从物品到语义ID

TIGER 使用预训练的 RQVAE 将物品特征映射为语义ID：

```python
# 物品特征 -> 语义ID
item_features = torch.tensor([...])  # (768,)
rqvae_output = rqvae(item_features)
semantic_ids = rqvae_output.sem_ids  # (3,) 三个语义ID
```

### 语义ID序列

用户交互历史转换为语义ID序列：

```python
user_history = [item1, item2, item3, ...]
semantic_sequence = []

for item_id in user_history:
    item_sem_ids = rqvae.get_semantic_ids(item_features[item_id])
    semantic_sequence.extend(item_sem_ids.tolist())

# 结果：[id1, id2, id3, id4, id5, id6, ...]
```

## 训练过程

### 数据准备

```python
class SeqData(NamedTuple):
    """序列数据格式"""
    user_id: int
    item_ids: List[int]      # 输入序列（语义ID）
    target_ids: List[int]    # 目标序列（语义ID）
```

### 损失函数

使用交叉熵损失进行序列建模：

```python
def compute_loss(logits, target_ids, mask):
    """计算序列建模损失"""
    # logits: (batch_size, seq_len, vocab_size)
    # target_ids: (batch_size, seq_len)
    # mask: (batch_size, seq_len)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    
    # 重塑为 2D
    logits_flat = logits.view(-1, logits.size(-1))
    target_flat = target_ids.view(-1)
    
    # 计算损失
    loss = loss_fn(logits_flat, target_flat)
    
    return loss
```

### 训练循环

```python
def train_step(model, batch, optimizer):
    """单步训练"""
    optimizer.zero_grad()
    
    # 前向传播
    user_ids = batch["user_input_ids"]
    item_ids = batch["item_input_ids"] 
    target_ids = batch["target_input_ids"]
    
    logits = model(user_ids, item_ids)
    
    # 计算损失
    loss = compute_loss(logits, target_ids, batch["seq_mask"])
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

## 推理过程

### 生成式推荐

```python
def generate_recommendations(model, user_sequence, max_length=10):
    """生成推荐序列"""
    model.eval()
    
    with torch.no_grad():
        # 编码用户序列
        encoded = model.encode_sequence(user_sequence)
        
        # 自回归生成
        generated = []
        current_input = encoded
        
        for _ in range(max_length):
            # 预测下一个token
            logits = model.decode_step(current_input)
            next_token = torch.argmax(logits, dim=-1)
            
            generated.append(next_token.item())
            
            # 更新输入
            current_input = torch.cat([current_input, next_token.unsqueeze(0)])
    
    return generated
```

### Trie约束生成

```python
def generate_with_trie_constraint(model, user_sequence, trie, max_length=10):
    """使用Trie约束的生成"""
    model.eval()
    
    generated = []
    current_node = trie
    
    with torch.no_grad():
        for step in range(max_length):
            # 获取有效的下一个tokens
            valid_tokens = list(current_node.keys())
            if not valid_tokens:
                break
                
            # 预测并约束
            logits = model.decode_step(user_sequence + generated)
            masked_logits = mask_invalid_tokens(logits, valid_tokens)
            
            next_token = torch.argmax(masked_logits, dim=-1).item()
            generated.append(next_token)
            
            # 更新Trie位置
            current_node = current_node[next_token]
            
            # 检查是否是完整的物品ID
            if current_node.is_end:
                break
    
    return generated
```

## 评估指标 {#评估指标}

### Top-K 推荐指标

```python
def compute_recall_at_k(predictions, targets, k=10):
    """计算 Recall@K"""
    recall_scores = []
    
    for pred, target in zip(predictions, targets):
        # 获取 top-k 预测
        top_k_pred = set(pred[:k])
        target_set = set(target)
        
        # 计算召回率
        if len(target_set) > 0:
            recall = len(top_k_pred & target_set) / len(target_set)
            recall_scores.append(recall)
    
    return np.mean(recall_scores)

def compute_ndcg_at_k(predictions, targets, k=10):
    """计算 NDCG@K"""
    ndcg_scores = []
    
    for pred, target in zip(predictions, targets):
        # 计算DCG
        dcg = 0
        for i, item in enumerate(pred[:k]):
            if item in target:
                dcg += 1 / np.log2(i + 2)  # +2 因为log2(1)=0
        
        # 计算IDCG
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(target), k)))
        
        # 计算NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores)
```

### 多样性指标

```python
def compute_diversity(recommendations):
    """计算推荐多样性"""
    all_items = set()
    for rec_list in recommendations:
        all_items.update(rec_list)
    
    # 物品覆盖度
    coverage = len(all_items) / total_items
    
    # 基尼系数
    item_counts = defaultdict(int)
    for rec_list in recommendations:
        for item in rec_list:
            item_counts[item] += 1
    
    counts = list(item_counts.values())
    gini = compute_gini_coefficient(counts)
    
    return {"coverage": coverage, "gini": gini}
```

## 模型优化

### 位置编码

```python
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

### 注意力机制优化

```python
class MultiHeadAttention(nn.Module):
    """优化的多头注意力"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention, weights = scaled_dot_product_attention(Q, K, V, mask, self.dropout)
        
        # Concatenate heads
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.d_k
        )
        
        output = self.w_o(attention)
        return output, weights
```

## 高级功能

### 用户行为建模

```python
class UserBehaviorEncoder(nn.Module):
    """用户行为编码器"""
    def __init__(self, config):
        super().__init__()
        
        # 时间编码
        self.time_encoder = nn.Linear(1, config.embedding_dim)
        
        # 行为类型编码
        self.behavior_embedding = nn.Embedding(
            config.num_behavior_types, 
            config.embedding_dim
        )
        
    def forward(self, sequences, timestamps, behavior_types):
        # 序列编码
        seq_encoded = self.item_encoder(sequences)
        
        # 时间编码
        time_encoded = self.time_encoder(timestamps.unsqueeze(-1))
        
        # 行为编码
        behavior_encoded = self.behavior_embedding(behavior_types)
        
        # 融合特征
        combined = seq_encoded + time_encoded + behavior_encoded
        
        return combined
```

### 冷启动处理

```python
class ColdStartHandler:
    """冷启动处理器"""
    
    def __init__(self, model, item_features):
        self.model = model
        self.item_features = item_features
        
    def recommend_for_new_user(self, user_profile, k=10):
        """为新用户推荐"""
        # 基于用户画像找相似物品
        similar_items = self.find_similar_items(user_profile)
        
        # 使用物品特征进行推荐
        recommendations = self.model.recommend_by_content(similar_items, k)
        
        return recommendations
    
    def recommend_new_item(self, item_features, k=10):
        """推荐新物品"""
        # 找到特征相似的现有物品
        similar_existing = self.find_similar_existing_items(item_features)
        
        # 基于相似物品的用户进行推荐
        target_users = self.get_users_who_liked(similar_existing)
        
        return target_users[:k]
```

## 实际应用

### 在线服务

```python
class TIGERRecommendationService:
    """TIGER 推荐服务"""
    
    def __init__(self, model_path, device='cuda'):
        self.model = Tiger.load_from_checkpoint(model_path)
        self.model.to(device)
        self.model.eval()
        self.device = device
        
    def get_recommendations(self, user_id, user_history, k=10):
        """获取推荐结果"""
        # 预处理用户历史
        semantic_sequence = self.preprocess_user_history(user_history)
        
        # 生成推荐
        with torch.no_grad():
            recommendations = self.model.generate_recommendations(
                semantic_sequence, max_length=k
            )
        
        # 后处理：语义ID -> 物品ID
        item_recommendations = self.semantic_to_items(recommendations)
        
        return item_recommendations
    
    def batch_recommend(self, user_requests):
        """批量推荐"""
        batch_results = []
        
        for user_id, user_history in user_requests:
            recommendations = self.get_recommendations(user_id, user_history)
            batch_results.append((user_id, recommendations))
        
        return batch_results
```

### A/B测试支持

```python
class ABTestingFramework:
    """A/B测试框架"""
    
    def __init__(self, model_a, model_b):
        self.model_a = model_a  # 控制组模型
        self.model_b = model_b  # 实验组模型
        
    def recommend_with_ab_test(self, user_id, user_history, test_group=None):
        """A/B测试推荐"""
        if test_group is None:
            # 随机分组
            test_group = 'A' if hash(user_id) % 2 == 0 else 'B'
        
        if test_group == 'A':
            return self.model_a.recommend(user_history), 'A'
        else:
            return self.model_b.recommend(user_history), 'B'
    
    def collect_metrics(self, user_id, recommendations, group, feedback):
        """收集A/B测试指标"""
        # 记录用户反馈和推荐结果
        metrics_data = {
            'user_id': user_id,
            'group': group,
            'recommendations': recommendations,
            'feedback': feedback,
            'timestamp': time.time()
        }
        
        # 存储到数据库或日志系统
        self.save_metrics(metrics_data)
```

TIGER 模型通过生成式的方法解决推荐问题，具有强大的序列建模能力和灵活的生成机制，特别适合处理复杂的用户行为序列和多样化的推荐场景。