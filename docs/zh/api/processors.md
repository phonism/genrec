# 处理器 API 参考

文本和序列处理工具的详细文档。

## 文本处理器

### TextProcessor

文本编码和处理的核心类。

```python
class TextProcessor:
    def __init__(self, config: TextEncodingConfig):
        self.config = config
        self.model = None
        self.device = config.device
        self.cache_manager = CacheManager(config.cache_dir)
```

**参数:**
- `config`: 文本编码配置对象

**方法:**

#### load_model()

加载文本编码模型。

```python
def load_model(self) -> None:
    """
    加载 Sentence Transformer 模型
    """
    if self.model is None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.config.encoder_model)
        self.model.to(self.device)
        print(f"Loaded text encoder: {self.config.encoder_model}")
```

#### encode_texts(texts, cache_key, force_reload)

编码文本列表。

```python
def encode_texts(
    self,
    texts: List[str],
    cache_key: Optional[str] = None,
    force_reload: bool = False
) -> np.ndarray:
    """
    编码文本列表为嵌入向量
    
    Args:
        texts: 文本列表
        cache_key: 缓存键，如果提供将尝试使用缓存
        force_reload: 是否强制重新计算
        
    Returns:
        嵌入矩阵 (num_texts, embedding_dim)
    """
    # 检查缓存
    if cache_key and not force_reload and self.cache_manager.exists(cache_key):
        print(f"Loading embeddings from cache: {cache_key}")
        return self.cache_manager.load(cache_key)
    
    # 加载模型
    self.load_model()
    
    # 批量编码
    print(f"Encoding {len(texts)} texts with {self.config.encoder_model}")
    embeddings = []
    
    for i in range(0, len(texts), self.config.batch_size):
        batch_texts = texts[i:i + self.config.batch_size]
        batch_embeddings = self.model.encode(
            batch_texts,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=True
        )
        embeddings.append(batch_embeddings)
    
    # 合并结果
    embeddings = np.vstack(embeddings)
    
    # 保存缓存
    if cache_key:
        self.cache_manager.save(cache_key, embeddings)
        print(f"Saved embeddings to cache: {cache_key}")
    
    return embeddings
```

#### encode_item_features(items_df, cache_key, force_reload)

编码物品特征。

```python
def encode_item_features(
    self,
    items_df: pd.DataFrame,
    cache_key: Optional[str] = None,
    force_reload: bool = False
) -> np.ndarray:
    """
    编码物品特征为嵌入向量
    
    Args:
        items_df: 物品数据框
        cache_key: 缓存键
        force_reload: 是否强制重新计算
        
    Returns:
        物品嵌入矩阵 (num_items, embedding_dim)
    """
    # 格式化文本
    texts = []
    for _, row in items_df.iterrows():
        text = self.config.format_text(row.to_dict())
        texts.append(text)
    
    return self.encode_texts(texts, cache_key, force_reload)
```

#### encode_single_text(text)

编码单个文本。

```python
def encode_single_text(self, text: str) -> np.ndarray:
    """
    编码单个文本
    
    Args:
        text: 输入文本
        
    Returns:
        文本嵌入向量 (embedding_dim,)
    """
    self.load_model()
    
    embedding = self.model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=self.config.normalize_embeddings
    )[0]
    
    return embedding
```

#### compute_similarity(text1, text2)

计算文本相似度。

```python
def compute_similarity(self, text1: str, text2: str) -> float:
    """
    计算两个文本的余弦相似度
    
    Args:
        text1: 第一个文本
        text2: 第二个文本
        
    Returns:
        余弦相似度值 [-1, 1]
    """
    embedding1 = self.encode_single_text(text1)
    embedding2 = self.encode_single_text(text2)
    
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
```

#### find_similar_texts(query_text, candidate_texts, top_k)

查找相似文本。

```python
def find_similar_texts(
    self,
    query_text: str,
    candidate_texts: List[str],
    top_k: int = 5
) -> List[Tuple[int, str, float]]:
    """
    查找与查询文本最相似的候选文本
    
    Args:
        query_text: 查询文本
        candidate_texts: 候选文本列表
        top_k: 返回前 k 个最相似的
        
    Returns:
        (索引, 文本, 相似度) 的列表，按相似度降序排列
    """
    query_embedding = self.encode_single_text(query_text)
    candidate_embeddings = self.encode_texts(candidate_texts)
    
    # 计算相似度
    similarities = np.dot(candidate_embeddings, query_embedding)
    
    # 获取 top-k
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append((idx, candidate_texts[idx], similarities[idx]))
    
    return results
```

## 序列处理器

### SequenceProcessor

序列数据处理的核心类。

```python
class SequenceProcessor:
    def __init__(self, config: SequenceConfig):
        self.config = config
        
    def build_user_sequences(
        self, 
        interactions_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        构建用户交互序列
        
        Args:
            interactions_df: 交互数据框，包含 user_id, item_id, timestamp
            
        Returns:
            用户序列列表，每个序列包含用户ID和物品序列
        """
        sequences = []
        
        # 按用户分组并按时间排序
        for user_id, group in interactions_df.groupby('user_id'):
            user_interactions = group.sort_values('timestamp')
            item_sequence = user_interactions['item_id'].tolist()
            
            # 过滤过短的序列
            if len(item_sequence) >= self.config.min_seq_length:
                sequences.append({
                    'user_id': user_id,
                    'item_sequence': item_sequence,
                    'timestamps': user_interactions['timestamp'].tolist() if self.config.include_timestamps else None
                })
        
        return sequences
```

#### create_training_samples(sequences)

创建训练样本。

```python
def create_training_samples(
    self,
    sequences: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    从用户序列创建训练样本
    
    Args:
        sequences: 用户序列列表
        
    Returns:
        训练样本列表，每个样本包含输入序列和目标序列
    """
    training_samples = []
    
    for seq_data in sequences:
        item_sequence = seq_data['item_sequence']
        
        # 创建多个子序列
        for i in range(0, len(item_sequence) - self.config.min_seq_length + 1, self.config.sequence_stride):
            # 确定子序列长度
            end_idx = min(i + self.config.max_seq_length, len(item_sequence))
            
            if end_idx - i >= self.config.min_seq_length:
                input_seq = item_sequence[i:end_idx-self.config.target_offset]
                target_seq = item_sequence[i+self.config.target_offset:end_idx]
                
                if len(input_seq) > 0 and len(target_seq) > 0:
                    sample = {
                        'user_id': seq_data['user_id'],
                        'input_sequence': input_seq,
                        'target_sequence': target_seq
                    }
                    
                    # 添加时间戳信息
                    if self.config.include_timestamps and seq_data['timestamps']:
                        sample['input_timestamps'] = seq_data['timestamps'][i:end_idx-self.config.target_offset]
                        sample['target_timestamps'] = seq_data['timestamps'][i+self.config.target_offset:end_idx]
                    
                    training_samples.append(sample)
    
    return training_samples
```

#### pad_and_truncate_sequence(sequence)

填充和截断序列。

```python
def pad_and_truncate_sequence(self, sequence: List[int]) -> List[int]:
    """
    填充和截断序列到指定长度
    
    Args:
        sequence: 输入序列
        
    Returns:
        处理后的序列
    """
    # 截断
    if len(sequence) > self.config.max_seq_length:
        sequence = self.config.truncate_sequence(sequence)
    
    # 填充
    if len(sequence) < self.config.max_seq_length:
        sequence = self.config.pad_sequence(sequence)
    
    return sequence
```

#### create_attention_mask(sequence)

创建注意力掩码。

```python
def create_attention_mask(self, sequence: List[int]) -> List[int]:
    """
    为序列创建注意力掩码
    
    Args:
        sequence: 输入序列
        
    Returns:
        注意力掩码，1 表示有效位置，0 表示填充位置
    """
    mask = []
    for token in sequence:
        if token == self.config.padding_token:
            mask.append(0)
        else:
            mask.append(1)
    
    return mask
```

#### encode_time_features(timestamps)

编码时间特征。

```python
def encode_time_features(self, timestamps: List[float]) -> np.ndarray:
    """
    将时间戳编码为特征向量
    
    Args:
        timestamps: 时间戳列表
        
    Returns:
        时间特征矩阵 (seq_len, time_encoding_dim)
    """
    if not timestamps:
        return np.zeros((0, self.config.time_encoding_dim))
    
    # 标准化时间戳
    timestamps = np.array(timestamps)
    min_time, max_time = timestamps.min(), timestamps.max()
    
    if max_time > min_time:
        normalized_times = (timestamps - min_time) / (max_time - min_time)
    else:
        normalized_times = np.zeros_like(timestamps)
    
    # 创建正弦和余弦编码
    time_features = []
    for i in range(self.config.time_encoding_dim // 2):
        freq = 1.0 / (10000 ** (2 * i / self.config.time_encoding_dim))
        sin_features = np.sin(normalized_times * freq)
        cos_features = np.cos(normalized_times * freq)
        time_features.extend([sin_features, cos_features])
    
    # 转置并截断到指定维度
    time_features = np.array(time_features[:self.config.time_encoding_dim]).T
    
    return time_features
```

## 数据增强处理器

### DataAugmentor

数据增强处理器。

```python
class DataAugmentor:
    def __init__(self, augmentation_config: Dict[str, Any]):
        self.config = augmentation_config
        
    def augment_sequence(self, sequence: List[int]) -> List[int]:
        """
        对序列进行数据增强
        
        Args:
            sequence: 原始序列
            
        Returns:
            增强后的序列
        """
        augmented = sequence.copy()
        
        # 随机删除
        if self.config.get('random_drop', False):
            drop_prob = self.config.get('drop_prob', 0.1)
            augmented = [item for item in augmented if random.random() > drop_prob]
        
        # 随机打乱
        if self.config.get('random_shuffle', False):
            shuffle_prob = self.config.get('shuffle_prob', 0.1)
            if random.random() < shuffle_prob:
                # 只打乱部分子序列
                start = random.randint(0, max(0, len(augmented) - 3))
                end = min(start + random.randint(2, 4), len(augmented))
                subseq = augmented[start:end]
                random.shuffle(subseq)
                augmented[start:end] = subseq
        
        # 随机替换
        if self.config.get('random_replace', False):
            replace_prob = self.config.get('replace_prob', 0.05)
            vocab_size = self.config.get('vocab_size', 1000)
            
            for i in range(len(augmented)):
                if random.random() < replace_prob:
                    augmented[i] = random.randint(1, vocab_size)
        
        return augmented
```

## 预处理管道

### PreprocessingPipeline

数据预处理管道。

```python
class PreprocessingPipeline:
    def __init__(
        self,
        text_processor: TextProcessor,
        sequence_processor: SequenceProcessor,
        augmentor: Optional[DataAugmentor] = None
    ):
        self.text_processor = text_processor
        self.sequence_processor = sequence_processor
        self.augmentor = augmentor
        
    def process_items(
        self,
        items_df: pd.DataFrame,
        cache_key: str = None
    ) -> pd.DataFrame:
        """
        处理物品数据
        
        Args:
            items_df: 物品数据框
            cache_key: 缓存键
            
        Returns:
            处理后的物品数据框，包含特征向量
        """
        print("Processing item features...")
        
        # 编码文本特征
        embeddings = self.text_processor.encode_item_features(
            items_df, cache_key=cache_key
        )
        
        # 添加特征到数据框
        processed_df = items_df.copy()
        processed_df['features'] = embeddings.tolist()
        
        return processed_df
        
    def process_interactions(
        self,
        interactions_df: pd.DataFrame,
        items_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        处理交互数据生成序列
        
        Args:
            interactions_df: 交互数据框
            items_df: 物品数据框
            
        Returns:
            处理后的序列数据
        """
        print("Building user sequences...")
        
        # 构建用户序列
        sequences = self.sequence_processor.build_user_sequences(interactions_df)
        
        # 创建训练样本
        training_samples = self.sequence_processor.create_training_samples(sequences)
        
        # 数据增强
        if self.augmentor:
            augmented_samples = []
            for sample in training_samples:
                # 原始样本
                augmented_samples.append(sample)
                
                # 增强样本
                aug_input = self.augmentor.augment_sequence(sample['input_sequence'])
                aug_target = self.augmentor.augment_sequence(sample['target_sequence'])
                
                augmented_sample = sample.copy()
                augmented_sample['input_sequence'] = aug_input
                augmented_sample['target_sequence'] = aug_target
                augmented_samples.append(augmented_sample)
            
            training_samples = augmented_samples
        
        return training_samples
```

## 工具函数

### compute_sequence_statistics(sequences)

计算序列统计信息。

```python
def compute_sequence_statistics(sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算序列数据的统计信息
    
    Args:
        sequences: 序列列表
        
    Returns:
        统计信息字典
    """
    if not sequences:
        return {}
    
    lengths = [len(seq['item_sequence']) for seq in sequences]
    unique_users = len(set(seq['user_id'] for seq in sequences))
    
    # 计算物品频率
    item_counts = {}
    for seq in sequences:
        for item_id in seq['item_sequence']:
            item_counts[item_id] = item_counts.get(item_id, 0) + 1
    
    stats = {
        'num_sequences': len(sequences),
        'num_unique_users': unique_users,
        'num_unique_items': len(item_counts),
        'avg_sequence_length': np.mean(lengths),
        'min_sequence_length': np.min(lengths),
        'max_sequence_length': np.max(lengths),
        'median_sequence_length': np.median(lengths),
        'total_interactions': sum(lengths),
        'most_popular_items': sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    }
    
    return stats
```

### visualize_embeddings(embeddings, labels, method)

可视化嵌入向量。

```python
def visualize_embeddings(
    embeddings: np.ndarray,
    labels: List[str] = None,
    method: str = 'tsne',
    save_path: str = None
) -> None:
    """
    可视化高维嵌入向量
    
    Args:
        embeddings: 嵌入矩阵 (n_samples, embedding_dim)
        labels: 样本标签
        method: 降维方法 ('tsne', 'pca', 'umap')
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    # 降维
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    elif method == 'umap':
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    if labels:
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                c=[colors[i]],
                label=label,
                alpha=0.7
            )
        plt.legend()
    else:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
    
    plt.title(f'Embedding Visualization ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
```

## 使用示例

### 文本处理

```python
from genrec.data.processors import TextProcessor
from genrec.data.configs import TextEncodingConfig

# 创建配置
config = TextEncodingConfig(
    encoder_model="sentence-transformers/all-MiniLM-L6-v2",
    template="Title: {title}; Category: {category}",
    batch_size=32
)

# 创建处理器
processor = TextProcessor(config)

# 编码文本
texts = ["Apple iPhone 13", "Samsung Galaxy S21", "Sony WH-1000XM4"]
embeddings = processor.encode_texts(texts, cache_key="sample_texts")

print(f"Embeddings shape: {embeddings.shape}")
```

### 序列处理

```python
from genrec.data.processors import SequenceProcessor
from genrec.data.configs import SequenceConfig

# 创建配置
config = SequenceConfig(
    max_seq_length=50,
    min_seq_length=3,
    target_offset=1
)

# 创建处理器
processor = SequenceProcessor(config)

# 处理交互数据
sequences = processor.build_user_sequences(interactions_df)
training_samples = processor.create_training_samples(sequences)

print(f"Generated {len(training_samples)} training samples")
```

### 完整预处理管道

```python
from genrec.data.processors import PreprocessingPipeline

# 创建管道
pipeline = PreprocessingPipeline(
    text_processor=text_processor,
    sequence_processor=sequence_processor
)

# 处理数据
processed_items = pipeline.process_items(items_df, cache_key="items_beauty")
processed_sequences = pipeline.process_interactions(interactions_df, processed_items)

# 查看统计信息
stats = compute_sequence_statistics(processed_sequences)
print(f"Dataset statistics: {stats}")
```