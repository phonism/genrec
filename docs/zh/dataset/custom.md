# 自定义数据集

本指南介绍如何为 genrec 框架添加自定义数据集。

## 基本概念

### 数据集类型

框架支持两种主要的数据集类型：

1. **ItemDataset**: 物品级数据集，用于训练 RQVAE
2. **SequenceDataset**: 序列级数据集，用于训练 TIGER

### 基础架构

所有数据集都继承自 `BaseRecommenderDataset`：

```python
from genrec.data.base_dataset import BaseRecommenderDataset

class MyCustomDataset(BaseRecommenderDataset):
    def __init__(self, config):
        super().__init__(config)
        # 初始化自定义参数
        
    def download(self):
        # 实现数据下载逻辑
        pass
        
    def load_raw_data(self):
        # 加载原始数据文件
        pass
        
    def preprocess_data(self, raw_data):
        # 预处理数据
        pass
        
    def extract_items(self, processed_data):
        # 提取物品信息
        pass
        
    def extract_interactions(self, processed_data):
        # 提取用户交互信息
        pass
```

## 实现步骤

### 1. 创建配置类

首先定义数据集特定的配置：

```python
from dataclasses import dataclass
from genrec.data.configs import DatasetConfig

@dataclass
class MyDatasetConfig(DatasetConfig):
    # 数据集特定参数
    api_key: str = ""
    data_format: str = "json"
    categories: List[str] = None
```

### 2. 实现数据下载

```python
def download(self):
    """下载数据集到本地"""
    if self._data_exists():
        return
        
    print("Downloading custom dataset...")
    
    # 示例：从 API 下载数据
    import requests
    response = requests.get(f"https://api.example.com/data?key={self.config.api_key}")
    
    # 保存到本地
    data_path = self.root_path / "raw" / "data.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(data_path, 'w') as f:
        json.dump(response.json(), f)
    
    print("Download completed.")

def _data_exists(self):
    """检查数据是否已存在"""
    data_path = self.root_path / "raw" / "data.json"
    return data_path.exists()
```

### 3. 实现数据加载

```python
def load_raw_data(self):
    """加载原始数据"""
    data_path = self.root_path / "raw" / "data.json"
    
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    
    # 解析数据结构
    users = raw_data.get('users', [])
    items = raw_data.get('items', [])
    interactions = raw_data.get('interactions', [])
    
    return {
        "users": pd.DataFrame(users),
        "items": pd.DataFrame(items), 
        "interactions": pd.DataFrame(interactions)
    }
```

### 4. 实现数据预处理

```python
def preprocess_data(self, raw_data):
    """预处理数据"""
    # 清洗物品数据
    items_df = self._clean_items(raw_data["items"])
    
    # 清洗交互数据
    interactions_df = self._clean_interactions(raw_data["interactions"])
    
    # 过滤低频用户和物品
    interactions_df = self.filter_low_interactions(
        interactions_df,
        min_user_interactions=self.config.processing_config.min_user_interactions,
        min_item_interactions=self.config.processing_config.min_item_interactions
    )
    
    # 处理物品特征
    items_df = self._process_item_features(items_df)
    
    return {
        "items": items_df,
        "interactions": interactions_df
    }

def _clean_items(self, items_df):
    """清洗物品数据"""
    # 填充缺失值
    items_df["title"] = items_df["title"].fillna("Unknown")
    items_df["category"] = items_df["category"].fillna("Unknown")
    
    # 标准化文本
    items_df["title"] = items_df["title"].str.strip()
    
    return items_df

def _clean_interactions(self, interactions_df):
    """清洗交互数据"""
    # 移除重复交互
    interactions_df = interactions_df.drop_duplicates(["user_id", "item_id"])
    
    # 添加时间戳（如果没有）
    if "timestamp" not in interactions_df.columns:
        interactions_df["timestamp"] = range(len(interactions_df))
    
    return interactions_df
```

### 5. 实现特征提取

```python
def _process_item_features(self, items_df):
    """处理物品特征"""
    # 使用文本处理器生成嵌入
    cache_key = f"custom_dataset_{self.config.split}"
    embeddings = self.text_processor.encode_item_features(
        items_df,
        cache_key=cache_key,
        force_reload=self.config.force_reload
    )
    
    # 添加嵌入特征
    items_df = items_df.copy()
    items_df["features"] = embeddings.tolist()
    
    # 创建文本字段用于参考
    texts = []
    for _, row in items_df.iterrows():
        text = f"Title: {row['title']}; Category: {row['category']}"
        texts.append(text)
    
    items_df["text"] = texts
    
    return items_df

def extract_items(self, processed_data):
    """提取物品信息"""
    return processed_data["items"]

def extract_interactions(self, processed_data):
    """提取交互信息"""
    return processed_data["interactions"]
```

## 创建数据集包装器

### 物品数据集

```python
from genrec.data.base_dataset import ItemDataset
import gin

@gin.configurable
class MyItemDataset(ItemDataset):
    """自定义物品数据集"""
    
    def __init__(
        self,
        root: str,
        split: str = "default",
        train_test_split: str = "all",
        api_key: str = "",
        **kwargs
    ):
        # 创建配置
        config = MyDatasetConfig(
            root_dir=root,
            split=split,
            api_key=api_key,
            **kwargs
        )
        
        # 创建基础数据集
        base_dataset = MyCustomDataset(config)
        
        # 初始化物品数据集
        super().__init__(
            base_dataset=base_dataset,
            split=train_test_split,
            return_text=False
        )
```

### 序列数据集

```python
from genrec.data.base_dataset import SequenceDataset

@gin.configurable
class MySequenceDataset(SequenceDataset):
    """自定义序列数据集"""
    
    def __init__(
        self,
        root: str,
        split: str = "default",
        train_test_split: str = "train",
        api_key: str = "",
        pretrained_rqvae_path: Optional[str] = None,
        **kwargs
    ):
        # 创建配置
        config = MyDatasetConfig(
            root_dir=root,
            split=split,
            api_key=api_key,
            **kwargs
        )
        
        # 加载语义编码器
        semantic_encoder = None
        if pretrained_rqvae_path:
            from genrec.models.rqvae import RqVae
            semantic_encoder = RqVae.load_from_checkpoint(pretrained_rqvae_path)
            semantic_encoder.eval()
        
        # 创建基础数据集
        base_dataset = MyCustomDataset(config)
        
        # 初始化序列数据集
        super().__init__(
            base_dataset=base_dataset,
            split=train_test_split,
            semantic_encoder=semantic_encoder
        )
```

## 注册数据集

### 使用工厂模式

```python
from genrec.data.dataset_factory import DatasetFactory

# 注册数据集
DatasetFactory.register_dataset(
    "my_dataset",
    base_class=MyCustomDataset,
    item_class=MyItemDataset,
    sequence_class=MySequenceDataset
)

# 使用工厂创建数据集
item_dataset = DatasetFactory.create_item_dataset(
    "my_dataset",
    root_dir="path/to/data",
    api_key="your_api_key"
)
```

## 配置文件集成

### Gin 配置文件

创建配置文件 `config/my_dataset.gin`：

```gin
import my_module.my_dataset

# 数据集参数
train.dataset_folder="dataset/my_dataset"
train.dataset=@MyItemDataset

# 自定义参数
MyItemDataset.api_key="your_api_key"
MyItemDataset.split="category_a"

# 文本编码参数
MyItemDataset.encoder_model_name="sentence-transformers/all-MiniLM-L6-v2"
```

## 测试和验证

### 单元测试

```python
import unittest
from my_dataset import MyCustomDataset, MyDatasetConfig

class TestMyDataset(unittest.TestCase):
    def setUp(self):
        self.config = MyDatasetConfig(
            root_dir="test_data",
            api_key="test_key"
        )
        self.dataset = MyCustomDataset(self.config)
    
    def test_data_loading(self):
        """测试数据加载"""
        # 模拟数据
        raw_data = self.dataset.load_raw_data()
        self.assertIn("items", raw_data)
        self.assertIn("interactions", raw_data)
    
    def test_preprocessing(self):
        """测试预处理"""
        raw_data = {"items": pd.DataFrame(), "interactions": pd.DataFrame()}
        processed = self.dataset.preprocess_data(raw_data)
        self.assertIn("items", processed)
        self.assertIn("interactions", processed)
```

### 数据质量验证

```python
def validate_dataset(dataset):
    """验证数据集质量"""
    # 检查数据完整性
    assert len(dataset) > 0, "数据集为空"
    
    # 检查特征维度
    sample = dataset[0]
    assert len(sample) == 768, f"特征维度错误: {len(sample)}"
    
    # 检查数据类型
    assert isinstance(sample, list), "数据类型错误"
    
    print("数据集验证通过")
```

## 最佳实践

### 1. 错误处理

```python
def load_raw_data(self):
    try:
        # 数据加载逻辑
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
    except Exception as e:
        raise RuntimeError(f"数据加载失败: {str(e)}")
```

### 2. 日志记录

```python
import logging

logger = logging.getLogger(__name__)

def preprocess_data(self, raw_data):
    logger.info("开始预处理数据")
    
    # 预处理逻辑
    
    logger.info(f"预处理完成，物品数量: {len(items_df)}, 交互数量: {len(interactions_df)}")
```

### 3. 配置验证

```python
def __post_init__(self):
    super().__post_init__()
    
    if not self.api_key:
        raise ValueError("API key 不能为空")
    
    if self.data_format not in ["json", "csv"]:
        raise ValueError(f"不支持的数据格式: {self.data_format}")
```

## 示例：MovieLens 数据集

完整的 MovieLens 数据集实现示例：

```python
@dataclass
class MovieLensConfig(DatasetConfig):
    """MovieLens 数据集配置"""
    version: str = "1m"  # 1m, 10m, 20m
    
class MovieLensDataset(BaseRecommenderDataset):
    """MovieLens 数据集实现"""
    
    URLS = {
        "1m": "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "10m": "http://files.grouplens.org/datasets/movielens/ml-10m.zip",
    }
    
    def download(self):
        if self._data_exists():
            return
            
        url = self.URLS[self.config.version]
        # 下载和解压逻辑
        
    def load_raw_data(self):
        # 加载 ratings.dat, movies.dat, users.dat
        pass
        
    def preprocess_data(self, raw_data):
        # MovieLens 特定的预处理
        pass
```

通过以上步骤，您可以成功为 genrec 框架添加自定义数据集支持。