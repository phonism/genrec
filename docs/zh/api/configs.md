# 配置管理 API 参考

配置管理类的详细文档，用于管理数据处理和模型训练参数。

## 基础配置类

### DatasetConfig

数据集基础配置类。

```python
@dataclass
class DatasetConfig:
    root_dir: str
    split: str = "default"
    force_reload: bool = False
    text_config: Optional[TextEncodingConfig] = None
    sequence_config: Optional[SequenceConfig] = None
    processing_config: Optional[DataProcessingConfig] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.text_config is None:
            self.text_config = TextEncodingConfig()
        if self.sequence_config is None:
            self.sequence_config = SequenceConfig()
        if self.processing_config is None:
            self.processing_config = DataProcessingConfig()
```

**参数:**
- `root_dir`: 数据集根目录
- `split`: 数据分割标识
- `force_reload`: 是否强制重新加载
- `text_config`: 文本编码配置
- `sequence_config`: 序列处理配置
- `processing_config`: 数据处理配置

## 文本编码配置

### TextEncodingConfig

文本编码相关配置。

```python
@dataclass
class TextEncodingConfig:
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    template: str = "Title: {title}; Brand: {brand}; Category: {category}; Price: {price}"
    batch_size: int = 32
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir: str = "cache/text_embeddings"
    normalize_embeddings: bool = True
    
    def __post_init__(self):
        """验证配置参数"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
```

**参数:**
- `encoder_model`: 文本编码器模型名称
- `template`: 文本模板格式
- `batch_size`: 批处理大小
- `max_length`: 最大文本长度
- `device`: 计算设备
- `cache_dir`: 缓存目录
- `normalize_embeddings`: 是否标准化嵌入

**方法:**

#### get_cache_key(split, model_name)

生成缓存键。

```python
def get_cache_key(self, split: str, model_name: str = None) -> str:
    """
    生成缓存键
    
    Args:
        split: 数据分割
        model_name: 模型名称
        
    Returns:
        缓存键字符串
    """
    if model_name is None:
        model_name = self.encoder_model
    return f"{model_name}_{split}_{hash(self.template)}"
```

#### format_text(item_data)

格式化物品文本。

```python
def format_text(self, item_data: Dict[str, Any]) -> str:
    """
    使用模板格式化物品文本
    
    Args:
        item_data: 物品数据字典
        
    Returns:
        格式化后的文本
    """
    try:
        return self.template.format(**item_data)
    except KeyError as e:
        # 处理缺失字段
        available_fields = set(item_data.keys())
        template_fields = set(re.findall(r'\{(\w+)\}', self.template))
        missing_fields = template_fields - available_fields
        
        # 用默认值填充缺失字段
        filled_data = item_data.copy()
        for field in missing_fields:
            filled_data[field] = "Unknown"
            
        return self.template.format(**filled_data)
```

## 序列处理配置

### SequenceConfig

序列处理相关配置。

```python
@dataclass
class SequenceConfig:
    max_seq_length: int = 50
    min_seq_length: int = 3
    padding_token: int = 0
    truncate_strategy: str = "recent"  # "recent", "random", "oldest"
    sequence_stride: int = 1
    target_offset: int = 1
    include_timestamps: bool = False
    time_encoding_dim: int = 32
    
    def __post_init__(self):
        """验证配置参数"""
        if self.max_seq_length <= self.min_seq_length:
            raise ValueError("max_seq_length must be greater than min_seq_length")
        if self.truncate_strategy not in ["recent", "random", "oldest"]:
            raise ValueError("Invalid truncate_strategy")
        if self.target_offset <= 0:
            raise ValueError("target_offset must be positive")
```

**参数:**
- `max_seq_length`: 最大序列长度
- `min_seq_length`: 最小序列长度
- `padding_token`: 填充标记
- `truncate_strategy`: 截断策略
- `sequence_stride`: 序列步长
- `target_offset`: 目标偏移
- `include_timestamps`: 是否包含时间戳
- `time_encoding_dim`: 时间编码维度

**方法:**

#### truncate_sequence(sequence, strategy)

截断序列。

```python
def truncate_sequence(
    self, 
    sequence: List[Any], 
    strategy: str = None
) -> List[Any]:
    """
    根据策略截断序列
    
    Args:
        sequence: 输入序列
        strategy: 截断策略，如为 None 则使用配置中的策略
        
    Returns:
        截断后的序列
    """
    if len(sequence) <= self.max_seq_length:
        return sequence
        
    strategy = strategy or self.truncate_strategy
    
    if strategy == "recent":
        return sequence[-self.max_seq_length:]
    elif strategy == "oldest":
        return sequence[:self.max_seq_length]
    elif strategy == "random":
        start_idx = random.randint(0, len(sequence) - self.max_seq_length)
        return sequence[start_idx:start_idx + self.max_seq_length]
    else:
        raise ValueError(f"Unknown truncate strategy: {strategy}")
```

#### pad_sequence(sequence)

填充序列。

```python
def pad_sequence(self, sequence: List[Any]) -> List[Any]:
    """
    填充序列到最大长度
    
    Args:
        sequence: 输入序列
        
    Returns:
        填充后的序列
    """
    if len(sequence) >= self.max_seq_length:
        return sequence[:self.max_seq_length]
    
    pad_length = self.max_seq_length - len(sequence)
    return sequence + [self.padding_token] * pad_length
```

## 数据处理配置

### DataProcessingConfig

数据处理相关配置。

```python
@dataclass
class DataProcessingConfig:
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    remove_duplicates: bool = True
    normalize_ratings: bool = False
    rating_scale: Tuple[float, float] = (1.0, 5.0)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    
    def __post_init__(self):
        """验证配置参数"""
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        if any(ratio <= 0 for ratio in [self.train_ratio, self.val_ratio, self.test_ratio]):
            raise ValueError("All ratios must be positive")
        if self.min_user_interactions <= 0 or self.min_item_interactions <= 0:
            raise ValueError("Minimum interactions must be positive")
```

**参数:**
- `min_user_interactions`: 最少用户交互数
- `min_item_interactions`: 最少物品交互数
- `remove_duplicates`: 是否移除重复交互
- `normalize_ratings`: 是否标准化评分
- `rating_scale`: 评分范围
- `train_ratio`: 训练集比例
- `val_ratio`: 验证集比例
- `test_ratio`: 测试集比例
- `random_seed`: 随机种子

**方法:**

#### get_split_indices(total_size)

获取数据分割索引。

```python
def get_split_indices(self, total_size: int) -> Tuple[List[int], List[int], List[int]]:
    """
    根据配置比例获取数据分割索引
    
    Args:
        total_size: 总数据量
        
    Returns:
        (train_indices, val_indices, test_indices): 分割索引列表
    """
    indices = list(range(total_size))
    random.Random(self.random_seed).shuffle(indices)
    
    train_size = int(total_size * self.train_ratio)
    val_size = int(total_size * self.val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices
```

#### normalize_rating(rating)

标准化评分。

```python
def normalize_rating(self, rating: float) -> float:
    """
    标准化评分到 [0, 1] 范围
    
    Args:
        rating: 原始评分
        
    Returns:
        标准化后的评分
    """
    if not self.normalize_ratings:
        return rating
        
    min_rating, max_rating = self.rating_scale
    return (rating - min_rating) / (max_rating - min_rating)
```

## 特定数据集配置

### P5AmazonConfig

P5 Amazon 数据集专用配置。

```python
@dataclass
class P5AmazonConfig(DatasetConfig):
    category: str = "beauty"
    min_rating: float = 4.0
    include_price: bool = True
    include_brand: bool = True
    download_url: str = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/"
    
    def __post_init__(self):
        super().__post_init__()
        
        # 设置特定的文本模板
        if self.include_price and self.include_brand:
            template = "Title: {title}; Brand: {brand}; Category: {category}; Price: {price}"
        elif self.include_brand:
            template = "Title: {title}; Brand: {brand}; Category: {category}"
        else:
            template = "Title: {title}; Category: {category}"
            
        self.text_config.template = template
        
    def get_category_url(self) -> str:
        """获取特定类别的下载 URL"""
        return f"{self.download_url}{self.category}.json.gz"
```

**额外参数:**
- `category`: 产品类别
- `min_rating`: 最低评分阈值
- `include_price`: 是否包含价格信息
- `include_brand`: 是否包含品牌信息
- `download_url`: 下载基础 URL

## 配置验证和工具

### validate_config(config)

验证配置完整性。

```python
def validate_config(config: DatasetConfig) -> List[str]:
    """
    验证配置的有效性
    
    Args:
        config: 数据集配置
        
    Returns:
        错误信息列表，空列表表示配置有效
    """
    errors = []
    
    # 检查根目录
    if not config.root_dir:
        errors.append("root_dir cannot be empty")
    
    # 检查文本配置
    if config.text_config:
        if not config.text_config.encoder_model:
            errors.append("encoder_model cannot be empty")
        if config.text_config.batch_size <= 0:
            errors.append("batch_size must be positive")
    
    # 检查序列配置
    if config.sequence_config:
        if config.sequence_config.max_seq_length <= config.sequence_config.min_seq_length:
            errors.append("max_seq_length must be greater than min_seq_length")
    
    # 检查处理配置
    if config.processing_config:
        ratios_sum = (
            config.processing_config.train_ratio + 
            config.processing_config.val_ratio + 
            config.processing_config.test_ratio
        )
        if abs(ratios_sum - 1.0) > 1e-6:
            errors.append("Split ratios must sum to 1.0")
    
    return errors
```

### load_config_from_file(config_path)

从文件加载配置。

```python
def load_config_from_file(config_path: str) -> DatasetConfig:
    """
    从 YAML 或 JSON 文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        数据集配置对象
    """
    config_path = Path(config_path)
    
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    # 根据配置类型创建相应对象
    config_type = config_dict.pop('config_type', 'DatasetConfig')
    
    if config_type == 'P5AmazonConfig':
        return P5AmazonConfig(**config_dict)
    else:
        return DatasetConfig(**config_dict)
```

### save_config_to_file(config, config_path)

保存配置到文件。

```python
def save_config_to_file(config: DatasetConfig, config_path: str) -> None:
    """
    保存配置到 YAML 或 JSON 文件
    
    Args:
        config: 数据集配置对象
        config_path: 配置文件路径
    """
    config_path = Path(config_path)
    config_dict = asdict(config)
    
    # 添加配置类型信息
    config_dict['config_type'] = config.__class__.__name__
    
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
```

## 使用示例

### 基本配置创建

```python
from genrec.data.configs import (
    DatasetConfig, TextEncodingConfig, SequenceConfig, DataProcessingConfig
)

# 创建基本配置
config = DatasetConfig(
    root_dir="dataset/amazon",
    split="beauty",
    text_config=TextEncodingConfig(
        encoder_model="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=64
    ),
    sequence_config=SequenceConfig(
        max_seq_length=100,
        min_seq_length=5
    ),
    processing_config=DataProcessingConfig(
        min_user_interactions=10,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
)
```

### P5 Amazon 配置

```python
from genrec.data.configs import P5AmazonConfig

# 创建 P5 Amazon 配置
config = P5AmazonConfig(
    root_dir="dataset/amazon",
    category="beauty",
    min_rating=4.0,
    include_price=True,
    include_brand=True
)
```

### 配置文件操作

```python
# 保存配置到文件
save_config_to_file(config, "config/dataset_config.yaml")

# 从文件加载配置
loaded_config = load_config_from_file("config/dataset_config.yaml")

# 验证配置
errors = validate_config(loaded_config)
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Configuration is valid")
```