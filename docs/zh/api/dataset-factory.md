# 数据集工厂 API 参考

数据集创建工厂模式的详细文档，用于统一管理和创建不同类型的数据集。

## 核心工厂类

### DatasetFactory

数据集工厂的核心类，管理数据集注册和创建。

```python
class DatasetFactory:
    """数据集工厂类"""
    
    _registered_datasets = {}
    
    @classmethod
    def register_dataset(
        cls,
        name: str,
        base_class: Type[BaseRecommenderDataset],
        item_class: Type[ItemDataset],
        sequence_class: Type[SequenceDataset]
    ) -> None:
        """
        注册数据集类
        
        Args:
            name: 数据集名称
            base_class: 基础数据集类
            item_class: 物品数据集类
            sequence_class: 序列数据集类
        """
        cls._registered_datasets[name] = {
            'base': base_class,
            'item': item_class,
            'sequence': sequence_class
        }
        print(f"Registered dataset: {name}")
```

**类方法:**

#### list_datasets()

列出所有已注册的数据集。

```python
@classmethod
def list_datasets(cls) -> List[str]:
    """
    列出所有已注册的数据集
    
    Returns:
        数据集名称列表
    """
    return list(cls._registered_datasets.keys())
```

#### get_dataset_info(name)

获取数据集信息。

```python
@classmethod
def get_dataset_info(cls, name: str) -> Dict[str, Type]:
    """
    获取指定数据集的类信息
    
    Args:
        name: 数据集名称
        
    Returns:
        包含数据集类的字典
        
    Raises:
        ValueError: 如果数据集未注册
    """
    if name not in cls._registered_datasets:
        available = ", ".join(cls.list_datasets())
        raise ValueError(f"Dataset '{name}' not registered. Available: {available}")
    
    return cls._registered_datasets[name]
```

#### create_base_dataset(name, **kwargs)

创建基础数据集实例。

```python
@classmethod
def create_base_dataset(cls, name: str, **kwargs) -> BaseRecommenderDataset:
    """
    创建基础数据集实例
    
    Args:
        name: 数据集名称
        **kwargs: 传递给数据集构造函数的参数
        
    Returns:
        基础数据集实例
    """
    dataset_info = cls.get_dataset_info(name)
    base_class = dataset_info['base']
    
    # 创建配置对象
    if 'config' not in kwargs:
        config_class = cls._get_config_class(base_class)
        kwargs['config'] = config_class(**kwargs)
    
    return base_class(kwargs['config'])
```

#### create_item_dataset(name, **kwargs)

创建物品数据集实例。

```python
@classmethod
def create_item_dataset(cls, name: str, **kwargs) -> ItemDataset:
    """
    创建物品数据集实例
    
    Args:
        name: 数据集名称
        **kwargs: 传递给数据集构造函数的参数
        
    Returns:
        物品数据集实例
    """
    dataset_info = cls.get_dataset_info(name)
    item_class = dataset_info['item']
    
    return item_class(**kwargs)
```

#### create_sequence_dataset(name, **kwargs)

创建序列数据集实例。

```python
@classmethod
def create_sequence_dataset(cls, name: str, **kwargs) -> SequenceDataset:
    """
    创建序列数据集实例
    
    Args:
        name: 数据集名称
        **kwargs: 传递给数据集构造函数的参数
        
    Returns:
        序列数据集实例
    """
    dataset_info = cls.get_dataset_info(name)
    sequence_class = dataset_info['sequence']
    
    return sequence_class(**kwargs)
```

#### _get_config_class(base_class)

获取数据集对应的配置类。

```python
@classmethod
def _get_config_class(cls, base_class: Type[BaseRecommenderDataset]) -> Type[DatasetConfig]:
    """
    根据基础数据集类获取对应的配置类
    
    Args:
        base_class: 基础数据集类
        
    Returns:
        配置类
    """
    # 通过类名或注解推断配置类
    if hasattr(base_class, '_config_class'):
        return base_class._config_class
    
    # 默认配置类映射
    config_mapping = {
        'P5AmazonDataset': P5AmazonConfig,
        'MovieLensDataset': MovieLensConfig,
        # 可以继续添加其他映射
    }
    
    class_name = base_class.__name__
    return config_mapping.get(class_name, DatasetConfig)
```

## 数据集注册器

### DatasetRegistry

数据集注册管理器。

```python
class DatasetRegistry:
    """数据集注册管理器"""
    
    def __init__(self):
        self.datasets = {}
        self.auto_register_builtin_datasets()
    
    def register(
        self,
        name: str,
        base_class: Type[BaseRecommenderDataset],
        item_class: Type[ItemDataset] = None,
        sequence_class: Type[SequenceDataset] = None,
        config_class: Type[DatasetConfig] = None
    ) -> None:
        """
        注册数据集
        
        Args:
            name: 数据集名称
            base_class: 基础数据集类
            item_class: 物品数据集类
            sequence_class: 序列数据集类
            config_class: 配置类
        """
        # 自动生成包装类（如果未提供）
        if item_class is None:
            item_class = self._create_item_wrapper(name, base_class)
        
        if sequence_class is None:
            sequence_class = self._create_sequence_wrapper(name, base_class)
        
        # 设置配置类
        if config_class:
            base_class._config_class = config_class
        
        self.datasets[name] = {
            'base': base_class,
            'item': item_class,
            'sequence': sequence_class,
            'config': config_class or DatasetConfig
        }
    
    def _create_item_wrapper(
        self, 
        name: str, 
        base_class: Type[BaseRecommenderDataset]
    ) -> Type[ItemDataset]:
        """动态创建物品数据集包装类"""
        
        class DynamicItemDataset(ItemDataset):
            def __init__(self, **kwargs):
                # 创建基础数据集
                config_class = getattr(base_class, '_config_class', DatasetConfig)
                config = config_class(**kwargs)
                base_dataset = base_class(config)
                
                # 初始化物品数据集
                super().__init__(
                    base_dataset=base_dataset,
                    split=kwargs.get('train_test_split', 'all'),
                    return_text=kwargs.get('return_text', False)
                )
        
        DynamicItemDataset.__name__ = f"{name.title()}ItemDataset"
        return DynamicItemDataset
    
    def _create_sequence_wrapper(
        self, 
        name: str, 
        base_class: Type[BaseRecommenderDataset]
    ) -> Type[SequenceDataset]:
        """动态创建序列数据集包装类"""
        
        class DynamicSequenceDataset(SequenceDataset):
            def __init__(self, **kwargs):
                # 创建基础数据集
                config_class = getattr(base_class, '_config_class', DatasetConfig)
                config = config_class(**kwargs)
                base_dataset = base_class(config)
                
                # 加载语义编码器
                semantic_encoder = None
                if 'pretrained_rqvae_path' in kwargs:
                    from genrec.models.rqvae import RqVae
                    semantic_encoder = RqVae.load_from_checkpoint(kwargs['pretrained_rqvae_path'])
                    semantic_encoder.eval()
                
                # 初始化序列数据集
                super().__init__(
                    base_dataset=base_dataset,
                    split=kwargs.get('train_test_split', 'train'),
                    semantic_encoder=semantic_encoder
                )
        
        DynamicSequenceDataset.__name__ = f"{name.title()}SequenceDataset"
        return DynamicSequenceDataset
    
    def auto_register_builtin_datasets(self) -> None:
        """自动注册内置数据集"""
        try:
            from genrec.data.p5_amazon import (
                P5AmazonDataset, P5AmazonItemDataset, P5AmazonSequenceDataset
            )
            from genrec.data.configs import P5AmazonConfig
            
            self.register(
                name="p5_amazon",
                base_class=P5AmazonDataset,
                item_class=P5AmazonItemDataset,
                sequence_class=P5AmazonSequenceDataset,
                config_class=P5AmazonConfig
            )
        except ImportError:
            pass
        
        # 可以继续添加其他内置数据集
```

## 配置构建器

### ConfigBuilder

配置对象构建器。

```python
class ConfigBuilder:
    """配置构建器"""
    
    def __init__(self, config_class: Type[DatasetConfig]):
        self.config_class = config_class
        self.params = {}
    
    def set_root_dir(self, root_dir: str) -> 'ConfigBuilder':
        """设置根目录"""
        self.params['root_dir'] = root_dir
        return self
    
    def set_split(self, split: str) -> 'ConfigBuilder':
        """设置数据分割"""
        self.params['split'] = split
        return self
    
    def set_text_config(self, **kwargs) -> 'ConfigBuilder':
        """设置文本配置"""
        self.params['text_config'] = TextEncodingConfig(**kwargs)
        return self
    
    def set_sequence_config(self, **kwargs) -> 'ConfigBuilder':
        """设置序列配置"""
        self.params['sequence_config'] = SequenceConfig(**kwargs)
        return self
    
    def set_processing_config(self, **kwargs) -> 'ConfigBuilder':
        """设置处理配置"""
        self.params['processing_config'] = DataProcessingConfig(**kwargs)
        return self
    
    def build(self) -> DatasetConfig:
        """构建配置对象"""
        return self.config_class(**self.params)
```

## 数据集管理器

### DatasetManager

数据集生命周期管理器。

```python
class DatasetManager:
    """数据集管理器"""
    
    def __init__(self):
        self.registry = DatasetRegistry()
        self.cache = {}
    
    def create_dataset(
        self,
        name: str,
        dataset_type: str = "item",
        cache_key: str = None,
        **kwargs
    ) -> Union[ItemDataset, SequenceDataset]:
        """
        创建数据集实例
        
        Args:
            name: 数据集名称
            dataset_type: 数据集类型 ("item" 或 "sequence")
            cache_key: 缓存键
            **kwargs: 数据集参数
            
        Returns:
            数据集实例
        """
        # 检查缓存
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]
        
        # 获取数据集信息
        if name not in self.registry.datasets:
            raise ValueError(f"Dataset '{name}' not registered")
        
        dataset_info = self.registry.datasets[name]
        
        # 创建数据集
        if dataset_type == "item":
            dataset = dataset_info['item'](**kwargs)
        elif dataset_type == "sequence":
            dataset = dataset_info['sequence'](**kwargs)
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")
        
        # 缓存结果
        if cache_key:
            self.cache[cache_key] = dataset
        
        return dataset
    
    def get_dataset_config(self, name: str, **kwargs) -> DatasetConfig:
        """
        获取数据集配置
        
        Args:
            name: 数据集名称
            **kwargs: 配置参数
            
        Returns:
            配置对象
        """
        if name not in self.registry.datasets:
            raise ValueError(f"Dataset '{name}' not registered")
        
        config_class = self.registry.datasets[name]['config']
        return config_class(**kwargs)
    
    def list_datasets(self) -> List[str]:
        """列出所有可用数据集"""
        return list(self.registry.datasets.keys())
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
```

## 工具函数

### register_dataset_from_module(module_path)

从模块注册数据集。

```python
def register_dataset_from_module(module_path: str) -> None:
    """
    从模块自动注册数据集
    
    Args:
        module_path: 模块路径，如 "my_package.my_dataset"
    """
    import importlib
    
    module = importlib.import_module(module_path)
    
    # 查找数据集类
    base_classes = []
    item_classes = []
    sequence_classes = []
    
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type):
            if issubclass(obj, BaseRecommenderDataset) and obj != BaseRecommenderDataset:
                base_classes.append(obj)
            elif issubclass(obj, ItemDataset) and obj != ItemDataset:
                item_classes.append(obj)
            elif issubclass(obj, SequenceDataset) and obj != SequenceDataset:
                sequence_classes.append(obj)
    
    # 自动匹配和注册
    for base_class in base_classes:
        dataset_name = base_class.__name__.lower().replace('dataset', '')
        
        # 查找对应的包装类
        item_class = None
        sequence_class = None
        
        for cls in item_classes:
            if dataset_name in cls.__name__.lower():
                item_class = cls
                break
        
        for cls in sequence_classes:
            if dataset_name in cls.__name__.lower():
                sequence_class = cls
                break
        
        # 注册数据集
        if item_class or sequence_class:
            DatasetFactory.register_dataset(
                name=dataset_name,
                base_class=base_class,
                item_class=item_class,
                sequence_class=sequence_class
            )
```

### create_dataset_from_config(config_path)

从配置文件创建数据集。

```python
def create_dataset_from_config(config_path: str) -> Union[ItemDataset, SequenceDataset]:
    """
    从配置文件创建数据集
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        数据集实例
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # 提取数据集信息
    dataset_name = config_data['dataset']['name']
    dataset_type = config_data['dataset']['type']
    dataset_params = config_data['dataset'].get('params', {})
    
    # 创建数据集
    manager = DatasetManager()
    return manager.create_dataset(
        name=dataset_name,
        dataset_type=dataset_type,
        **dataset_params
    )
```

## 使用示例

### 注册自定义数据集

```python
from genrec.data.dataset_factory import DatasetFactory
from my_package.my_dataset import MyDataset, MyItemDataset, MySequenceDataset

# 方式1：手动注册
DatasetFactory.register_dataset(
    name="my_dataset",
    base_class=MyDataset,
    item_class=MyItemDataset,
    sequence_class=MySequenceDataset
)

# 方式2：使用注册器
registry = DatasetRegistry()
registry.register(
    name="my_dataset",
    base_class=MyDataset,
    item_class=MyItemDataset,
    sequence_class=MySequenceDataset
)
```

### 创建数据集实例

```python
# 使用工厂方法
item_dataset = DatasetFactory.create_item_dataset(
    "p5_amazon",
    root="dataset/amazon",
    split="beauty",
    train_test_split="train"
)

sequence_dataset = DatasetFactory.create_sequence_dataset(
    "p5_amazon",
    root="dataset/amazon",
    split="beauty",
    train_test_split="train",
    pretrained_rqvae_path="checkpoints/rqvae.ckpt"
)
```

### 使用数据集管理器

```python
# 创建管理器
manager = DatasetManager()

# 列出可用数据集
print("Available datasets:", manager.list_datasets())

# 创建数据集
dataset = manager.create_dataset(
    name="p5_amazon",
    dataset_type="item",
    cache_key="amazon_beauty_train",
    root="dataset/amazon",
    split="beauty"
)
```

### 配置构建器

```python
# 使用构建器创建配置
config = (ConfigBuilder(P5AmazonConfig)
    .set_root_dir("dataset/amazon")
    .set_split("beauty")
    .set_text_config(
        encoder_model="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=64
    )
    .set_sequence_config(
        max_seq_length=100,
        min_seq_length=5
    )
    .build())

# 使用配置创建数据集
base_dataset = P5AmazonDataset(config)
```

### 从配置文件创建

```yaml
# dataset_config.yaml
dataset:
  name: p5_amazon
  type: item
  params:
    root: "dataset/amazon"
    split: "beauty"
    train_test_split: "train"
    encoder_model_name: "sentence-transformers/all-MiniLM-L6-v2"
```

```python
# 从配置文件加载
dataset = create_dataset_from_config("dataset_config.yaml")
```