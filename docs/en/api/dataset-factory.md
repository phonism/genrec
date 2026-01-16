# Dataset Factory API Reference

Detailed documentation for the dataset factory pattern used to uniformly manage and create different types of datasets.

## Core Factory Class

### DatasetFactory

Core class for dataset factory, managing dataset registration and creation.

```python
class DatasetFactory:
    """Dataset factory class"""
    
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
        Register dataset classes
        
        Args:
            name: Dataset name
            base_class: Base dataset class
            item_class: Item dataset class
            sequence_class: Sequence dataset class
        """
        cls._registered_datasets[name] = {
            'base': base_class,
            'item': item_class,
            'sequence': sequence_class
        }
        print(f"Registered dataset: {name}")
```

**Class Methods:**

#### list_datasets()

List all registered datasets.

```python
@classmethod
def list_datasets(cls) -> List[str]:
    """
    List all registered datasets
    
    Returns:
        Dataset name list
    """
    return list(cls._registered_datasets.keys())
```

#### get_dataset_info(name)

Get dataset information.

```python
@classmethod
def get_dataset_info(cls, name: str) -> Dict[str, Type]:
    """
    Get class information for specified dataset
    
    Args:
        name: Dataset name
        
    Returns:
        Dictionary containing dataset classes
        
    Raises:
        ValueError: If dataset is not registered
    """
    if name not in cls._registered_datasets:
        available = ", ".join(cls.list_datasets())
        raise ValueError(f"Dataset '{name}' not registered. Available: {available}")
    
    return cls._registered_datasets[name]
```

#### create_base_dataset(name, **kwargs)

Create base dataset instance.

```python
@classmethod
def create_base_dataset(cls, name: str, **kwargs) -> BaseRecommenderDataset:
    """
    Create base dataset instance
    
    Args:
        name: Dataset name
        **kwargs: Parameters passed to dataset constructor
        
    Returns:
        Base dataset instance
    """
    dataset_info = cls.get_dataset_info(name)
    base_class = dataset_info['base']
    
    # Create configuration object
    if 'config' not in kwargs:
        config_class = cls._get_config_class(base_class)
        kwargs['config'] = config_class(**kwargs)
    
    return base_class(kwargs['config'])
```

#### create_item_dataset(name, **kwargs)

Create item dataset instance.

```python
@classmethod
def create_item_dataset(cls, name: str, **kwargs) -> ItemDataset:
    """
    Create item dataset instance
    
    Args:
        name: Dataset name
        **kwargs: Parameters passed to dataset constructor
        
    Returns:
        Item dataset instance
    """
    dataset_info = cls.get_dataset_info(name)
    item_class = dataset_info['item']
    
    return item_class(**kwargs)
```

#### create_sequence_dataset(name, **kwargs)

Create sequence dataset instance.

```python
@classmethod
def create_sequence_dataset(cls, name: str, **kwargs) -> SequenceDataset:
    """
    Create sequence dataset instance
    
    Args:
        name: Dataset name
        **kwargs: Parameters passed to dataset constructor
        
    Returns:
        Sequence dataset instance
    """
    dataset_info = cls.get_dataset_info(name)
    sequence_class = dataset_info['sequence']
    
    return sequence_class(**kwargs)
```

#### _get_config_class(base_class)

Get configuration class corresponding to dataset.

```python
@classmethod
def _get_config_class(cls, base_class: Type[BaseRecommenderDataset]) -> Type[DatasetConfig]:
    """
    Get configuration class corresponding to base dataset class
    
    Args:
        base_class: Base dataset class
        
    Returns:
        Configuration class
    """
    # Infer configuration class through class name or annotations
    if hasattr(base_class, '_config_class'):
        return base_class._config_class
    
    # Default configuration class mapping
    config_mapping = {
        'P5AmazonDataset': P5AmazonConfig,
        'MovieLensDataset': MovieLensConfig,
        # Can continue adding other mappings
    }
    
    class_name = base_class.__name__
    return config_mapping.get(class_name, DatasetConfig)
```

## Dataset Registry

### DatasetRegistry

Dataset registration manager.

```python
class DatasetRegistry:
    """Dataset registration manager"""
    
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
        Register dataset
        
        Args:
            name: Dataset name
            base_class: Base dataset class
            item_class: Item dataset class
            sequence_class: Sequence dataset class
            config_class: Configuration class
        """
        # Auto-generate wrapper classes if not provided
        if item_class is None:
            item_class = self._create_item_wrapper(name, base_class)
        
        if sequence_class is None:
            sequence_class = self._create_sequence_wrapper(name, base_class)
        
        # Set configuration class
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
        """Dynamically create item dataset wrapper class"""
        
        class DynamicItemDataset(ItemDataset):
            def __init__(self, **kwargs):
                # Create base dataset
                config_class = getattr(base_class, '_config_class', DatasetConfig)
                config = config_class(**kwargs)
                base_dataset = base_class(config)
                
                # Initialize item dataset
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
        """Dynamically create sequence dataset wrapper class"""
        
        class DynamicSequenceDataset(SequenceDataset):
            def __init__(self, **kwargs):
                # Create base dataset
                config_class = getattr(base_class, '_config_class', DatasetConfig)
                config = config_class(**kwargs)
                base_dataset = base_class(config)
                
                # Load semantic encoder
                semantic_encoder = None
                if 'pretrained_rqvae_path' in kwargs:
                    from genrec.models.rqvae import RqVae
                    semantic_encoder = RqVae.load_from_checkpoint(kwargs['pretrained_rqvae_path'])
                    semantic_encoder.eval()
                
                # Initialize sequence dataset
                super().__init__(
                    base_dataset=base_dataset,
                    split=kwargs.get('train_test_split', 'train'),
                    semantic_encoder=semantic_encoder
                )
        
        DynamicSequenceDataset.__name__ = f"{name.title()}SequenceDataset"
        return DynamicSequenceDataset
    
    def auto_register_builtin_datasets(self) -> None:
        """Auto-register built-in datasets"""
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
        
        # Can continue adding other built-in datasets
```

## Configuration Builder

### ConfigBuilder

Configuration object builder.

```python
class ConfigBuilder:
    """Configuration builder"""
    
    def __init__(self, config_class: Type[DatasetConfig]):
        self.config_class = config_class
        self.params = {}
    
    def set_root_dir(self, root_dir: str) -> 'ConfigBuilder':
        """Set root directory"""
        self.params['root_dir'] = root_dir
        return self
    
    def set_split(self, split: str) -> 'ConfigBuilder':
        """Set data split"""
        self.params['split'] = split
        return self
    
    def set_text_config(self, **kwargs) -> 'ConfigBuilder':
        """Set text configuration"""
        self.params['text_config'] = TextEncodingConfig(**kwargs)
        return self
    
    def set_sequence_config(self, **kwargs) -> 'ConfigBuilder':
        """Set sequence configuration"""
        self.params['sequence_config'] = SequenceConfig(**kwargs)
        return self
    
    def set_processing_config(self, **kwargs) -> 'ConfigBuilder':
        """Set processing configuration"""
        self.params['processing_config'] = DataProcessingConfig(**kwargs)
        return self
    
    def build(self) -> DatasetConfig:
        """Build configuration object"""
        return self.config_class(**self.params)
```

## Dataset Manager

### DatasetManager

Dataset lifecycle manager.

```python
class DatasetManager:
    """Dataset manager"""
    
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
        Create dataset instance
        
        Args:
            name: Dataset name
            dataset_type: Dataset type ("item" or "sequence")
            cache_key: Cache key
            **kwargs: Dataset parameters
            
        Returns:
            Dataset instance
        """
        # Check cache
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Get dataset information
        if name not in self.registry.datasets:
            raise ValueError(f"Dataset '{name}' not registered")
        
        dataset_info = self.registry.datasets[name]
        
        # Create dataset
        if dataset_type == "item":
            dataset = dataset_info['item'](**kwargs)
        elif dataset_type == "sequence":
            dataset = dataset_info['sequence'](**kwargs)
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")
        
        # Cache result
        if cache_key:
            self.cache[cache_key] = dataset
        
        return dataset
    
    def get_dataset_config(self, name: str, **kwargs) -> DatasetConfig:
        """
        Get dataset configuration
        
        Args:
            name: Dataset name
            **kwargs: Configuration parameters
            
        Returns:
            Configuration object
        """
        if name not in self.registry.datasets:
            raise ValueError(f"Dataset '{name}' not registered")
        
        config_class = self.registry.datasets[name]['config']
        return config_class(**kwargs)
    
    def list_datasets(self) -> List[str]:
        """List all available datasets"""
        return list(self.registry.datasets.keys())
    
    def clear_cache(self) -> None:
        """Clear cache"""
        self.cache.clear()
```

## Utility Functions

### register_dataset_from_module(module_path)

Register dataset from module.

```python
def register_dataset_from_module(module_path: str) -> None:
    """
    Auto-register dataset from module
    
    Args:
        module_path: Module path, e.g., "my_package.my_dataset"
    """
    import importlib
    
    module = importlib.import_module(module_path)
    
    # Find dataset classes
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
    
    # Auto-match and register
    for base_class in base_classes:
        dataset_name = base_class.__name__.lower().replace('dataset', '')
        
        # Find corresponding wrapper classes
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
        
        # Register dataset
        if item_class or sequence_class:
            DatasetFactory.register_dataset(
                name=dataset_name,
                base_class=base_class,
                item_class=item_class,
                sequence_class=sequence_class
            )
```

### create_dataset_from_config(config_path)

Create dataset from configuration file.

```python
def create_dataset_from_config(config_path: str) -> Union[ItemDataset, SequenceDataset]:
    """
    Create dataset from configuration file
    
    Args:
        config_path: Configuration file path
        
    Returns:
        Dataset instance
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Extract dataset information
    dataset_name = config_data['dataset']['name']
    dataset_type = config_data['dataset']['type']
    dataset_params = config_data['dataset'].get('params', {})
    
    # Create dataset
    manager = DatasetManager()
    return manager.create_dataset(
        name=dataset_name,
        dataset_type=dataset_type,
        **dataset_params
    )
```

## Usage Examples

### Register Custom Dataset

```python
from genrec.data.dataset_factory import DatasetFactory
from my_package.my_dataset import MyDataset, MyItemDataset, MySequenceDataset

# Method 1: Manual registration
DatasetFactory.register_dataset(
    name="my_dataset",
    base_class=MyDataset,
    item_class=MyItemDataset,
    sequence_class=MySequenceDataset
)

# Method 2: Using registry
registry = DatasetRegistry()
registry.register(
    name="my_dataset",
    base_class=MyDataset,
    item_class=MyItemDataset,
    sequence_class=MySequenceDataset
)
```

### Create Dataset Instances

```python
# Using factory methods
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

### Using Dataset Manager

```python
# Create manager
manager = DatasetManager()

# List available datasets
print("Available datasets:", manager.list_datasets())

# Create dataset
dataset = manager.create_dataset(
    name="p5_amazon",
    dataset_type="item",
    cache_key="amazon_beauty_train",
    root="dataset/amazon",
    split="beauty"
)
```

### Configuration Builder

```python
# Use builder to create configuration
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

# Use configuration to create dataset
base_dataset = P5AmazonDataset(config)
```

### Create from Configuration File

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
# Load from configuration file
dataset = create_dataset_from_config("dataset_config.yaml")
```