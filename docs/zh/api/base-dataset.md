# 基础数据集 API 参考

数据集抽象基类和通用数据处理接口的详细文档。

## 抽象基类

### BaseRecommenderDataset

所有推荐数据集的抽象基类。

```python
class BaseRecommenderDataset(ABC):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.root_path = Path(config.root_dir)
        self.text_processor = TextProcessor(config.text_config)
```

**参数:**
- `config`: 数据集配置对象

**抽象方法:**

#### download()

下载数据集到本地。

```python
@abstractmethod
def download(self) -> None:
    """下载数据集到本地存储"""
    pass
```

#### load_raw_data()

加载原始数据文件。

```python
@abstractmethod
def load_raw_data(self) -> Dict[str, Any]:
    """
    加载原始数据文件
    
    Returns:
        包含原始数据的字典，通常包含 'items' 和 'interactions' 键
    """
    pass
```

#### preprocess_data(raw_data)

预处理原始数据。

```python
@abstractmethod
def preprocess_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    预处理原始数据
    
    Args:
        raw_data: 原始数据字典
        
    Returns:
        预处理后的数据字典
    """
    pass
```

#### extract_items(processed_data)

提取物品信息。

```python
@abstractmethod
def extract_items(self, processed_data: Dict[str, Any]) -> pd.DataFrame:
    """
    提取物品信息
    
    Args:
        processed_data: 预处理后的数据
        
    Returns:
        物品信息 DataFrame
    """
    pass
```

#### extract_interactions(processed_data)

提取用户交互信息。

```python
@abstractmethod
def extract_interactions(self, processed_data: Dict[str, Any]) -> pd.DataFrame:
    """
    提取用户交互信息
    
    Args:
        processed_data: 预处理后的数据
        
    Returns:
        交互信息 DataFrame
    """
    pass
```

**公共方法:**

#### get_dataset()

获取完整的数据集。

```python
def get_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    获取完整的数据集
    
    Returns:
        (items_df, interactions_df): 物品和交互数据框
    """
```

#### filter_low_interactions(interactions_df, min_user_interactions, min_item_interactions)

过滤低频用户和物品。

```python
def filter_low_interactions(
    self,
    interactions_df: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5
) -> pd.DataFrame:
    """
    过滤低频用户和物品
    
    Args:
        interactions_df: 交互数据框
        min_user_interactions: 最少用户交互数
        min_item_interactions: 最少物品交互数
        
    Returns:
        过滤后的交互数据框
    """
```

## 数据集包装器

### ItemDataset

用于物品级数据的数据集包装器，主要用于训练 RQVAE。

```python
class ItemDataset(Dataset):
    def __init__(
        self,
        base_dataset: BaseRecommenderDataset,
        split: str = "all",
        return_text: bool = False
    ):
```

**参数:**
- `base_dataset`: 基础数据集实例
- `split`: 数据分割 ("all", "train", "val", "test")
- `return_text`: 是否返回文本信息

**方法:**

#### __len__()

返回数据集大小。

```python
def __len__(self) -> int:
    """返回数据集中物品数量"""
```

#### __getitem__(idx)

获取单个数据样本。

```python
def __getitem__(self, idx: int) -> Union[torch.Tensor, Dict[str, Any]]:
    """
    获取单个物品数据
    
    Args:
        idx: 物品索引
        
    Returns:
        如果 return_text=False: 物品特征向量 (torch.Tensor)
        如果 return_text=True: 包含特征和文本的字典
    """
```

#### get_item_features(item_id)

根据物品 ID 获取特征。

```python
def get_item_features(self, item_id: int) -> torch.Tensor:
    """
    根据物品 ID 获取特征
    
    Args:
        item_id: 物品 ID
        
    Returns:
        物品特征向量
    """
```

### SequenceDataset

用于序列级数据的数据集包装器，主要用于训练 TIGER。

```python
class SequenceDataset(Dataset):
    def __init__(
        self,
        base_dataset: BaseRecommenderDataset,
        split: str = "train",
        semantic_encoder: Optional[torch.nn.Module] = None,
        sequence_config: Optional[SequenceConfig] = None
    ):
```

**参数:**
- `base_dataset`: 基础数据集实例
- `split`: 数据分割 ("train", "val", "test")
- `semantic_encoder`: 语义编码器 (RQVAE)
- `sequence_config`: 序列配置

**方法:**

#### __len__()

返回数据集大小。

```python
def __len__(self) -> int:
    """返回数据集中用户序列数量"""
```

#### __getitem__(idx)

获取单个序列数据。

```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    """
    获取单个用户序列数据
    
    Args:
        idx: 序列索引
        
    Returns:
        包含输入序列和目标序列的字典:
        - 'input_ids': 输入序列 (torch.Tensor)
        - 'labels': 目标序列 (torch.Tensor)  
        - 'attention_mask': 注意力掩码 (torch.Tensor)
    """
```

#### build_sequences()

构建用户交互序列。

```python
def build_sequences(self) -> List[Dict[str, Any]]:
    """
    构建用户交互序列
    
    Returns:
        序列列表，每个序列包含用户 ID 和物品 ID 列表
    """
```

#### encode_sequences_to_semantic_ids(sequences)

将序列编码为语义 ID。

```python
def encode_sequences_to_semantic_ids(
    self, 
    sequences: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    将物品序列编码为语义 ID 序列
    
    Args:
        sequences: 原始序列列表
        
    Returns:
        编码后的序列列表
    """
```

## 数据处理工具

### train_test_split(interactions_df, test_ratio, val_ratio)

分割训练、验证和测试集。

```python
def train_test_split(
    interactions_df: pd.DataFrame,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    time_based: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    分割训练、验证和测试集
    
    Args:
        interactions_df: 交互数据框
        test_ratio: 测试集比例
        val_ratio: 验证集比例
        time_based: 是否基于时间分割
        
    Returns:
        (train_df, val_df, test_df): 分割后的数据框
    """
```

### create_item_mapping(items_df)

创建物品 ID 映射。

```python
def create_item_mapping(items_df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    创建物品 ID 映射
    
    Args:
        items_df: 物品数据框
        
    Returns:
        (id_to_index, index_to_id): ID 映射字典
    """
```

### normalize_features(features)

标准化特征向量。

```python
def normalize_features(features: np.ndarray, method: str = "l2") -> np.ndarray:
    """
    标准化特征向量
    
    Args:
        features: 特征矩阵
        method: 标准化方法 ("l2", "minmax", "zscore")
        
    Returns:
        标准化后的特征矩阵
    """
```

## 缓存机制

### CacheManager

管理数据处理缓存。

```python
class CacheManager:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"
        
    def exists(self, cache_key: str) -> bool:
        """检查缓存是否存在"""
        return self.get_cache_path(cache_key).exists()
        
    def save(self, cache_key: str, data: Any) -> None:
        """保存数据到缓存"""
        cache_path = self.get_cache_path(cache_key)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
            
    def load(self, cache_key: str) -> Any:
        """从缓存加载数据"""
        cache_path = self.get_cache_path(cache_key)
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
```

## 数据验证

### validate_dataset(dataset)

验证数据集完整性。

```python
def validate_dataset(dataset: Dataset) -> Dict[str, Any]:
    """
    验证数据集完整性
    
    Args:
        dataset: 数据集实例
        
    Returns:
        验证结果字典
    """
    results = {
        'size': len(dataset),
        'sample_shapes': [],
        'data_types': [],
        'errors': []
    }
    
    try:
        # 检查数据集大小
        if len(dataset) == 0:
            results['errors'].append("Dataset is empty")
            
        # 检查样本形状和类型
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            if isinstance(sample, torch.Tensor):
                results['sample_shapes'].append(sample.shape)
                results['data_types'].append(sample.dtype)
            elif isinstance(sample, dict):
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        results['sample_shapes'].append((key, value.shape))
                        results['data_types'].append((key, value.dtype))
                        
    except Exception as e:
        results['errors'].append(f"Validation error: {str(e)}")
        
    return results
```

## 使用示例

### 创建自定义数据集

```python
from genrec.data.base_dataset import BaseRecommenderDataset
from genrec.data.configs import DatasetConfig

class MyDataset(BaseRecommenderDataset):
    def download(self):
        # 实现下载逻辑
        pass
        
    def load_raw_data(self):
        # 加载数据
        return {"items": items_df, "interactions": interactions_df}
        
    def preprocess_data(self, raw_data):
        # 预处理
        return raw_data
        
    def extract_items(self, processed_data):
        return processed_data["items"]
        
    def extract_interactions(self, processed_data):
        return processed_data["interactions"]

# 使用数据集
config = DatasetConfig(root_dir="data", split="default")
dataset = MyDataset(config)
items_df, interactions_df = dataset.get_dataset()
```

### 创建物品数据集

```python
from genrec.data.base_dataset import ItemDataset

# 创建物品数据集
item_dataset = ItemDataset(
    base_dataset=dataset,
    split="train",
    return_text=False
)

# 使用 DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(item_dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    features = batch  # (batch_size, feature_dim)
    # 训练逻辑
```

### 创建序列数据集

```python
from genrec.data.base_dataset import SequenceDataset
from genrec.models.rqvae import RqVae

# 加载语义编码器
semantic_encoder = RqVae.load_from_checkpoint("rqvae.ckpt")

# 创建序列数据集
sequence_dataset = SequenceDataset(
    base_dataset=dataset,
    split="train",
    semantic_encoder=semantic_encoder
)

# 使用 DataLoader
dataloader = DataLoader(sequence_dataset, batch_size=16, shuffle=True)

for batch in dataloader:
    input_ids = batch['input_ids']  # (batch_size, seq_len)
    labels = batch['labels']        # (batch_size, seq_len)
    # 训练逻辑
```