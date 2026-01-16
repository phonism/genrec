# 数据集 API 参考

genrec 数据集模块的详细 API 文档。

## 基础数据集类

### BaseRecommenderDataset

所有推荐系统数据集的抽象基类。

```python
class BaseRecommenderDataset:
    """推荐系统数据集基类"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.root_dir = Path(config.root_dir)
        self._items_df = None
        self._interactions_df = None
    
    @abstractmethod
    def download(self) -> None:
        """下载数据集"""
        pass
    
    @abstractmethod
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """加载原始数据"""
        pass
    
    @abstractmethod
    def preprocess_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """预处理数据"""
        pass
```

**主要方法：**

#### load_dataset()
加载数据集。

```python
def load_dataset(self, force_reload: bool = False) -> None:
    """
    加载数据集
    
    Args:
        force_reload: 是否强制重新加载
    """
```

#### get_items()
获取物品数据。

```python
def get_items(self) -> pd.DataFrame:
    """
    获取物品数据
    
    Returns:
        物品 DataFrame
    """
```

#### get_interactions()
获取交互数据。

```python
def get_interactions(self) -> pd.DataFrame:
    """
    获取用户-物品交互数据
    
    Returns:
        交互 DataFrame
    """
```

## 物品数据集类

### ItemDataset

用于物品编码和特征学习的数据集类。

```python
class ItemDataset(Dataset):
    """物品数据集类"""
    
    def __init__(
        self,
        base_dataset: BaseRecommenderDataset,
        split: str = "all",
        return_text: bool = False
    ):
        self.base_dataset = base_dataset
        self.split = split
        self.return_text = return_text
```

**参数：**
- `base_dataset`: 基础数据集实例
- `split`: 数据分割（"train", "val", "test", "all"）
- `return_text`: 是否返回文本特征

**方法：**

#### __getitem__(idx)
获取数据项。

```python
def __getitem__(self, idx: int) -> Dict[str, Any]:
    """
    获取指定索引的数据项
    
    Args:
        idx: 数据索引
        
    Returns:
        包含物品信息的字典
    """
```

## 序列数据集类

### SequenceDataset

用于序列生成训练的数据集类。

```python
class SequenceDataset(Dataset):
    """序列数据集类"""
    
    def __init__(
        self,
        base_dataset: BaseRecommenderDataset,
        split: str = "train",
        semantic_encoder: Optional[nn.Module] = None
    ):
        self.base_dataset = base_dataset
        self.split = split
        self.semantic_encoder = semantic_encoder
```

**参数：**
- `base_dataset`: 基础数据集实例
- `split`: 数据分割
- `semantic_encoder`: 语义编码器（如 RQVAE）

**方法：**

#### create_sequences()
创建用户序列。

```python
def create_sequences(self) -> List[Dict[str, Any]]:
    """
    创建用户交互序列
    
    Returns:
        序列列表
    """
```

#### encode_sequence()
编码序列。

```python
def encode_sequence(self, item_ids: List[int]) -> torch.Tensor:
    """
    将物品 ID 序列编码为语义表示
    
    Args:
        item_ids: 物品 ID 列表
        
    Returns:
        编码后的序列张量
    """
```

## 具体数据集实现

### P5AmazonDataset

P5 Amazon 数据集实现。

```python
@gin.configurable
class P5AmazonDataset(BaseRecommenderDataset):
    """P5 Amazon 数据集"""
    
    def __init__(self, config: P5AmazonConfig):
        super().__init__(config)
        self.category = config.category
        self.min_rating = config.min_rating
```

**特色功能：**
- 支持多个产品类别
- 自动下载和预处理
- 文本特征提取
- 评分过滤

### P5AmazonItemDataset

P5 Amazon 物品数据集封装。

```python
@gin.configurable
class P5AmazonItemDataset(ItemDataset):
    """P5 Amazon 物品数据集"""
    
    def __init__(
        self,
        root: str,
        split: str = "beauty",
        train_test_split: str = "all",
        return_text: bool = False,
        **kwargs
    ):
```

### P5AmazonSequenceDataset

P5 Amazon 序列数据集封装。

```python
@gin.configurable
class P5AmazonSequenceDataset(SequenceDataset):
    """P5 Amazon 序列数据集"""
    
    def __init__(
        self,
        root: str,
        split: str = "beauty", 
        train_test_split: str = "train",
        pretrained_rqvae_path: str = None,
        **kwargs
    ):
```

## 数据集工厂

### DatasetFactory

数据集工厂类，用于统一管理和创建数据集。

```python
class DatasetFactory:
    """数据集工厂"""
    
    _registered_datasets = {}
    
    @classmethod
    def register_dataset(
        cls,
        name: str,
        base_class: Type[BaseRecommenderDataset],
        item_class: Type[ItemDataset],
        sequence_class: Type[SequenceDataset]
    ) -> None:
        """注册数据集类"""
```

**使用示例：**

```python
# 注册数据集
DatasetFactory.register_dataset(
    "p5_amazon",
    P5AmazonDataset,
    P5AmazonItemDataset, 
    P5AmazonSequenceDataset
)

# 创建数据集
item_dataset = DatasetFactory.create_item_dataset(
    "p5_amazon",
    root="data/amazon",
    split="beauty"
)
```

## 数据处理器

### TextProcessor

文本处理器，用于物品文本特征编码。

```python
class TextProcessor:
    """文本处理器"""
    
    def __init__(self, config: TextEncodingConfig):
        self.config = config
        self.encoder = SentenceTransformer(config.encoder_model)
```

**方法：**

#### encode_item_features()
编码物品文本特征。

```python
def encode_item_features(
    self,
    items_df: pd.DataFrame,
    cache_key: str = None,
    force_reload: bool = False
) -> torch.Tensor:
    """
    编码物品文本特征
    
    Args:
        items_df: 物品数据框
        cache_key: 缓存键
        force_reload: 是否强制重新计算
        
    Returns:
        物品文本编码张量
    """
```

### SequenceProcessor

序列处理器，用于序列数据预处理。

```python
class SequenceProcessor:
    """序列处理器"""
    
    def __init__(self, config: SequenceConfig):
        self.config = config
```

**方法：**

#### process_user_sequence()
处理用户序列。

```python
def process_user_sequence(
    self,
    sequence: List[int],
    target_offset: int = 1
) -> Dict[str, torch.Tensor]:
    """
    处理用户交互序列
    
    Args:
        sequence: 原始序列
        target_offset: 目标偏移量
        
    Returns:
        处理后的序列数据
    """
```

## 使用示例

### 基础使用

```python
from genrec.data import P5AmazonDataset, P5AmazonConfig

# 创建配置
config = P5AmazonConfig(
    root_dir="data/amazon",
    split="beauty"
)

# 创建数据集
dataset = P5AmazonDataset(config)
dataset.load_dataset()

# 获取数据
items = dataset.get_items()
interactions = dataset.get_interactions()
```

### 物品数据集使用

```python
from genrec.data import P5AmazonItemDataset

# 创建物品数据集
item_dataset = P5AmazonItemDataset(
    root="data/amazon",
    split="beauty",
    return_text=True
)

# 使用 DataLoader
dataloader = DataLoader(item_dataset, batch_size=32, shuffle=True)
for batch in dataloader:
    item_ids = batch['item_id']
    text_features = batch['text_features']
    # 训练物品编码器...
```

### 序列数据集使用

```python
from genrec.data import P5AmazonSequenceDataset
from genrec.models import RqVae

# 加载预训练的 RQVAE
rqvae = RqVae.load_from_checkpoint("checkpoints/rqvae.ckpt")

# 创建序列数据集
seq_dataset = P5AmazonSequenceDataset(
    root="data/amazon",
    split="beauty",
    train_test_split="train",
    pretrained_rqvae_path="checkpoints/rqvae.ckpt"
)

# 使用 DataLoader
dataloader = DataLoader(seq_dataset, batch_size=16, shuffle=True)
for batch in dataloader:
    input_ids = batch['input_ids']
    target_ids = batch['target_ids']
    # 训练序列生成模型...
```

## 相关链接

- [配置管理](configs.md) - 数据集配置系统
- [处理器](processors.md) - 数据处理工具
- [数据集工厂](dataset-factory.md) - 工厂模式创建数据集
- [训练器](trainers.md) - 模型训练工具