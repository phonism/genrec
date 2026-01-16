# Dataset API Reference

Detailed API documentation for the genrec dataset module.

## Base Dataset Classes

### BaseRecommenderDataset

Abstract base class for all recommender system datasets.

```python
class BaseRecommenderDataset:
    """Base class for recommender system datasets"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.root_dir = Path(config.root_dir)
        self._items_df = None
        self._interactions_df = None
    
    @abstractmethod
    def download(self) -> None:
        """Download dataset"""
        pass
    
    @abstractmethod
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load raw data"""
        pass
    
    @abstractmethod
    def preprocess_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Preprocess data"""
        pass
```

**Main Methods:**

#### load_dataset()
Load the dataset.

```python
def load_dataset(self, force_reload: bool = False) -> None:
    """
    Load dataset
    
    Args:
        force_reload: Whether to force reload
    """
```

#### get_items()
Get item data.

```python
def get_items(self) -> pd.DataFrame:
    """
    Get item data
    
    Returns:
        Items DataFrame
    """
```

#### get_interactions()
Get interaction data.

```python
def get_interactions(self) -> pd.DataFrame:
    """
    Get user-item interaction data
    
    Returns:
        Interactions DataFrame
    """
```

## Item Dataset Class

### ItemDataset

Dataset class for item encoding and feature learning.

```python
class ItemDataset(Dataset):
    """Item dataset class"""
    
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

**Parameters:**
- `base_dataset`: Base dataset instance
- `split`: Data split ("train", "val", "test", "all")
- `return_text`: Whether to return text features

**Methods:**

#### __getitem__(idx)
Get data item.

```python
def __getitem__(self, idx: int) -> Dict[str, Any]:
    """
    Get data item at specified index
    
    Args:
        idx: Data index
        
    Returns:
        Dictionary containing item information
    """
```

## Sequence Dataset Class

### SequenceDataset

Dataset class for sequence generation training.

```python
class SequenceDataset(Dataset):
    """Sequence dataset class"""
    
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

**Parameters:**
- `base_dataset`: Base dataset instance
- `split`: Data split
- `semantic_encoder`: Semantic encoder (e.g., RQVAE)

**Methods:**

#### create_sequences()
Create user sequences.

```python
def create_sequences(self) -> List[Dict[str, Any]]:
    """
    Create user interaction sequences
    
    Returns:
        List of sequences
    """
```

#### encode_sequence()
Encode sequences.

```python
def encode_sequence(self, item_ids: List[int]) -> torch.Tensor:
    """
    Encode item ID sequence to semantic representation
    
    Args:
        item_ids: List of item IDs
        
    Returns:
        Encoded sequence tensor
    """
```

## Concrete Dataset Implementations

### P5AmazonDataset

P5 Amazon dataset implementation.

```python
@gin.configurable
class P5AmazonDataset(BaseRecommenderDataset):
    """P5 Amazon dataset"""
    
    def __init__(self, config: P5AmazonConfig):
        super().__init__(config)
        self.category = config.category
        self.min_rating = config.min_rating
```

**Key Features:**
- Supports multiple product categories
- Automatic download and preprocessing
- Text feature extraction
- Rating filtering

### P5AmazonItemDataset

P5 Amazon item dataset wrapper.

```python
@gin.configurable
class P5AmazonItemDataset(ItemDataset):
    """P5 Amazon item dataset"""
    
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

P5 Amazon sequence dataset wrapper.

```python
@gin.configurable
class P5AmazonSequenceDataset(SequenceDataset):
    """P5 Amazon sequence dataset"""
    
    def __init__(
        self,
        root: str,
        split: str = "beauty", 
        train_test_split: str = "train",
        pretrained_rqvae_path: str = None,
        **kwargs
    ):
```

## Dataset Factory

### DatasetFactory

Dataset factory class for unified dataset management and creation.

```python
class DatasetFactory:
    """Dataset factory"""
    
    _registered_datasets = {}
    
    @classmethod
    def register_dataset(
        cls,
        name: str,
        base_class: Type[BaseRecommenderDataset],
        item_class: Type[ItemDataset],
        sequence_class: Type[SequenceDataset]
    ) -> None:
        """Register dataset classes"""
```

**Usage Example:**

```python
# Register dataset
DatasetFactory.register_dataset(
    "p5_amazon",
    P5AmazonDataset,
    P5AmazonItemDataset, 
    P5AmazonSequenceDataset
)

# Create dataset
item_dataset = DatasetFactory.create_item_dataset(
    "p5_amazon",
    root="data/amazon",
    split="beauty"
)
```

## Data Processors

### TextProcessor

Text processor for item text feature encoding.

```python
class TextProcessor:
    """Text processor"""
    
    def __init__(self, config: TextEncodingConfig):
        self.config = config
        self.encoder = SentenceTransformer(config.encoder_model)
```

**Methods:**

#### encode_item_features()
Encode item text features.

```python
def encode_item_features(
    self,
    items_df: pd.DataFrame,
    cache_key: str = None,
    force_reload: bool = False
) -> torch.Tensor:
    """
    Encode item text features
    
    Args:
        items_df: Items dataframe
        cache_key: Cache key
        force_reload: Whether to force recomputation
        
    Returns:
        Item text encoding tensor
    """
```

### SequenceProcessor

Sequence processor for sequence data preprocessing.

```python
class SequenceProcessor:
    """Sequence processor"""
    
    def __init__(self, config: SequenceConfig):
        self.config = config
```

**Methods:**

#### process_user_sequence()
Process user sequences.

```python
def process_user_sequence(
    self,
    sequence: List[int],
    target_offset: int = 1
) -> Dict[str, torch.Tensor]:
    """
    Process user interaction sequence
    
    Args:
        sequence: Raw sequence
        target_offset: Target offset
        
    Returns:
        Processed sequence data
    """
```

## Usage Examples

### Basic Usage

```python
from genrec.data import P5AmazonDataset, P5AmazonConfig

# Create configuration
config = P5AmazonConfig(
    root_dir="data/amazon",
    split="beauty"
)

# Create dataset
dataset = P5AmazonDataset(config)
dataset.load_dataset()

# Get data
items = dataset.get_items()
interactions = dataset.get_interactions()
```

### Item Dataset Usage

```python
from genrec.data import P5AmazonItemDataset

# Create item dataset
item_dataset = P5AmazonItemDataset(
    root="data/amazon",
    split="beauty",
    return_text=True
)

# Use DataLoader
dataloader = DataLoader(item_dataset, batch_size=32, shuffle=True)
for batch in dataloader:
    item_ids = batch['item_id']
    text_features = batch['text_features']
    # Train item encoder...
```

### Sequence Dataset Usage

```python
from genrec.data import P5AmazonSequenceDataset
from genrec.models import RqVae

# Load pretrained RQVAE
rqvae = RqVae.load_from_checkpoint("checkpoints/rqvae.ckpt")

# Create sequence dataset
seq_dataset = P5AmazonSequenceDataset(
    root="data/amazon",
    split="beauty",
    train_test_split="train",
    pretrained_rqvae_path="checkpoints/rqvae.ckpt"
)

# Use DataLoader
dataloader = DataLoader(seq_dataset, batch_size=16, shuffle=True)
for batch in dataloader:
    input_ids = batch['input_ids']
    target_ids = batch['target_ids']
    # Train sequence generation model...
```

## Related Links

- [Configurations](configs.md) - Dataset configuration system
- [Processors](processors.md) - Data processing utilities
- [Dataset Factory](dataset-factory.md) - Factory pattern for dataset creation
- [Trainers](trainers.md) - Model training utilities