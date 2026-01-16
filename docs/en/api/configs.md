# Configuration Management API Reference

Detailed documentation for configuration management classes used to manage data processing and model training parameters.

## Base Configuration Classes

### DatasetConfig

Base dataset configuration class.

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
        """Post-initialization processing"""
        if self.text_config is None:
            self.text_config = TextEncodingConfig()
        if self.sequence_config is None:
            self.sequence_config = SequenceConfig()
        if self.processing_config is None:
            self.processing_config = DataProcessingConfig()
```

**Parameters:**
- `root_dir`: Dataset root directory
- `split`: Data split identifier
- `force_reload`: Whether to force reload
- `text_config`: Text encoding configuration
- `sequence_config`: Sequence processing configuration
- `processing_config`: Data processing configuration

## Text Encoding Configuration

### TextEncodingConfig

Text encoding related configuration.

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
        """Validate configuration parameters"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
```

**Parameters:**
- `encoder_model`: Text encoder model name
- `template`: Text template format
- `batch_size`: Batch processing size
- `max_length`: Maximum text length
- `device`: Computing device
- `cache_dir`: Cache directory
- `normalize_embeddings`: Whether to normalize embeddings

**Methods:**

#### get_cache_key(split, model_name)

Generate cache key.

```python
def get_cache_key(self, split: str, model_name: str = None) -> str:
    """
    Generate cache key
    
    Args:
        split: Data split
        model_name: Model name
        
    Returns:
        Cache key string
    """
    if model_name is None:
        model_name = self.encoder_model
    return f"{model_name}_{split}_{hash(self.template)}"
```

#### format_text(item_data)

Format item text.

```python
def format_text(self, item_data: Dict[str, Any]) -> str:
    """
    Format item text using template
    
    Args:
        item_data: Item data dictionary
        
    Returns:
        Formatted text
    """
    try:
        return self.template.format(**item_data)
    except KeyError as e:
        # Handle missing fields
        available_fields = set(item_data.keys())
        template_fields = set(re.findall(r'\{(\w+)\}', self.template))
        missing_fields = template_fields - available_fields
        
        # Fill missing fields with default values
        filled_data = item_data.copy()
        for field in missing_fields:
            filled_data[field] = "Unknown"
            
        return self.template.format(**filled_data)
```

## Sequence Processing Configuration

### SequenceConfig

Sequence processing related configuration.

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
        """Validate configuration parameters"""
        if self.max_seq_length <= self.min_seq_length:
            raise ValueError("max_seq_length must be greater than min_seq_length")
        if self.truncate_strategy not in ["recent", "random", "oldest"]:
            raise ValueError("Invalid truncate_strategy")
        if self.target_offset <= 0:
            raise ValueError("target_offset must be positive")
```

**Parameters:**
- `max_seq_length`: Maximum sequence length
- `min_seq_length`: Minimum sequence length
- `padding_token`: Padding token
- `truncate_strategy`: Truncation strategy
- `sequence_stride`: Sequence stride
- `target_offset`: Target offset
- `include_timestamps`: Whether to include timestamps
- `time_encoding_dim`: Time encoding dimension

**Methods:**

#### truncate_sequence(sequence, strategy)

Truncate sequence.

```python
def truncate_sequence(
    self, 
    sequence: List[Any], 
    strategy: str = None
) -> List[Any]:
    """
    Truncate sequence according to strategy
    
    Args:
        sequence: Input sequence
        strategy: Truncation strategy, uses config strategy if None
        
    Returns:
        Truncated sequence
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

Pad sequence.

```python
def pad_sequence(self, sequence: List[Any]) -> List[Any]:
    """
    Pad sequence to maximum length
    
    Args:
        sequence: Input sequence
        
    Returns:
        Padded sequence
    """
    if len(sequence) >= self.max_seq_length:
        return sequence[:self.max_seq_length]
    
    pad_length = self.max_seq_length - len(sequence)
    return sequence + [self.padding_token] * pad_length
```

## Data Processing Configuration

### DataProcessingConfig

Data processing related configuration.

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
        """Validate configuration parameters"""
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        if any(ratio <= 0 for ratio in [self.train_ratio, self.val_ratio, self.test_ratio]):
            raise ValueError("All ratios must be positive")
        if self.min_user_interactions <= 0 or self.min_item_interactions <= 0:
            raise ValueError("Minimum interactions must be positive")
```

**Parameters:**
- `min_user_interactions`: Minimum user interactions
- `min_item_interactions`: Minimum item interactions
- `remove_duplicates`: Whether to remove duplicate interactions
- `normalize_ratings`: Whether to normalize ratings
- `rating_scale`: Rating range
- `train_ratio`: Training set ratio
- `val_ratio`: Validation set ratio
- `test_ratio`: Test set ratio
- `random_seed`: Random seed

**Methods:**

#### get_split_indices(total_size)

Get data split indices.

```python
def get_split_indices(self, total_size: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Get data split indices according to configured ratios
    
    Args:
        total_size: Total data size
        
    Returns:
        (train_indices, val_indices, test_indices): Split index lists
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

Normalize rating.

```python
def normalize_rating(self, rating: float) -> float:
    """
    Normalize rating to [0, 1] range
    
    Args:
        rating: Original rating
        
    Returns:
        Normalized rating
    """
    if not self.normalize_ratings:
        return rating
        
    min_rating, max_rating = self.rating_scale
    return (rating - min_rating) / (max_rating - min_rating)
```

## Specific Dataset Configurations

### P5AmazonConfig

P5 Amazon dataset specific configuration.

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
        
        # Set specific text template
        if self.include_price and self.include_brand:
            template = "Title: {title}; Brand: {brand}; Category: {category}; Price: {price}"
        elif self.include_brand:
            template = "Title: {title}; Brand: {brand}; Category: {category}"
        else:
            template = "Title: {title}; Category: {category}"
            
        self.text_config.template = template
        
    def get_category_url(self) -> str:
        """Get download URL for specific category"""
        return f"{self.download_url}{self.category}.json.gz"
```

**Additional Parameters:**
- `category`: Product category
- `min_rating`: Minimum rating threshold
- `include_price`: Whether to include price information
- `include_brand`: Whether to include brand information
- `download_url`: Download base URL

## Configuration Validation and Tools

### validate_config(config)

Validate configuration integrity.

```python
def validate_config(config: DatasetConfig) -> List[str]:
    """
    Validate configuration validity
    
    Args:
        config: Dataset configuration
        
    Returns:
        List of error messages, empty list means valid configuration
    """
    errors = []
    
    # Check root directory
    if not config.root_dir:
        errors.append("root_dir cannot be empty")
    
    # Check text configuration
    if config.text_config:
        if not config.text_config.encoder_model:
            errors.append("encoder_model cannot be empty")
        if config.text_config.batch_size <= 0:
            errors.append("batch_size must be positive")
    
    # Check sequence configuration
    if config.sequence_config:
        if config.sequence_config.max_seq_length <= config.sequence_config.min_seq_length:
            errors.append("max_seq_length must be greater than min_seq_length")
    
    # Check processing configuration
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

Load configuration from file.

```python
def load_config_from_file(config_path: str) -> DatasetConfig:
    """
    Load configuration from YAML or JSON file
    
    Args:
        config_path: Configuration file path
        
    Returns:
        Dataset configuration object
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
    
    # Create appropriate object based on configuration type
    config_type = config_dict.pop('config_type', 'DatasetConfig')
    
    if config_type == 'P5AmazonConfig':
        return P5AmazonConfig(**config_dict)
    else:
        return DatasetConfig(**config_dict)
```

### save_config_to_file(config, config_path)

Save configuration to file.

```python
def save_config_to_file(config: DatasetConfig, config_path: str) -> None:
    """
    Save configuration to YAML or JSON file
    
    Args:
        config: Dataset configuration object
        config_path: Configuration file path
    """
    config_path = Path(config_path)
    config_dict = asdict(config)
    
    # Add configuration type information
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

## Usage Examples

### Basic Configuration Creation

```python
from genrec.data.configs import (
    DatasetConfig, TextEncodingConfig, SequenceConfig, DataProcessingConfig
)

# Create basic configuration
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

### P5 Amazon Configuration

```python
from genrec.data.configs import P5AmazonConfig

# Create P5 Amazon configuration
config = P5AmazonConfig(
    root_dir="dataset/amazon",
    category="beauty",
    min_rating=4.0,
    include_price=True,
    include_brand=True
)
```

### Configuration File Operations

```python
# Save configuration to file
save_config_to_file(config, "config/dataset_config.yaml")

# Load configuration from file
loaded_config = load_config_from_file("config/dataset_config.yaml")

# Validate configuration
errors = validate_config(loaded_config)
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Configuration is valid")
```