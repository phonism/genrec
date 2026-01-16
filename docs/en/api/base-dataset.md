# Base Dataset API Reference

Detailed documentation for abstract base classes and common data processing interfaces.

## Abstract Base Classes

### BaseRecommenderDataset

Abstract base class for all recommendation datasets.

```python
class BaseRecommenderDataset(ABC):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.root_path = Path(config.root_dir)
        self.text_processor = TextProcessor(config.text_config)
```

**Parameters:**
- `config`: Dataset configuration object

**Abstract Methods:**

#### download()

Download dataset to local storage.

```python
@abstractmethod
def download(self) -> None:
    """Download dataset to local storage"""
    pass
```

#### load_raw_data()

Load raw data files.

```python
@abstractmethod
def load_raw_data(self) -> Dict[str, Any]:
    """
    Load raw data files
    
    Returns:
        Dictionary containing raw data, typically with 'items' and 'interactions' keys
    """
    pass
```

#### preprocess_data(raw_data)

Preprocess raw data.

```python
@abstractmethod
def preprocess_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess raw data
    
    Args:
        raw_data: Raw data dictionary
        
    Returns:
        Preprocessed data dictionary
    """
    pass
```

#### extract_items(processed_data)

Extract item information.

```python
@abstractmethod
def extract_items(self, processed_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract item information
    
    Args:
        processed_data: Preprocessed data
        
    Returns:
        Items DataFrame
    """
    pass
```

#### extract_interactions(processed_data)

Extract user interaction information.

```python
@abstractmethod
def extract_interactions(self, processed_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract user interaction information
    
    Args:
        processed_data: Preprocessed data
        
    Returns:
        Interactions DataFrame
    """
    pass
```

**Public Methods:**

#### get_dataset()

Get complete dataset.

```python
def get_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get complete dataset
    
    Returns:
        (items_df, interactions_df): Item and interaction DataFrames
    """
```

#### filter_low_interactions(interactions_df, min_user_interactions, min_item_interactions)

Filter low-frequency users and items.

```python
def filter_low_interactions(
    self,
    interactions_df: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5
) -> pd.DataFrame:
    """
    Filter low-frequency users and items
    
    Args:
        interactions_df: Interactions DataFrame
        min_user_interactions: Minimum user interactions
        min_item_interactions: Minimum item interactions
        
    Returns:
        Filtered interactions DataFrame
    """
```

## Dataset Wrappers

### ItemDataset

Dataset wrapper for item-level data, primarily used for training RQVAE.

```python
class ItemDataset(Dataset):
    def __init__(
        self,
        base_dataset: BaseRecommenderDataset,
        split: str = "all",
        return_text: bool = False
    ):
```

**Parameters:**
- `base_dataset`: Base dataset instance
- `split`: Data split ("all", "train", "val", "test")
- `return_text`: Whether to return text information

**Methods:**

#### __len__()

Return dataset size.

```python
def __len__(self) -> int:
    """Return number of items in dataset"""
```

#### __getitem__(idx)

Get single data sample.

```python
def __getitem__(self, idx: int) -> Union[torch.Tensor, Dict[str, Any]]:
    """
    Get single item data
    
    Args:
        idx: Item index
        
    Returns:
        If return_text=False: Item feature vector (torch.Tensor)
        If return_text=True: Dictionary containing features and text
    """
```

#### get_item_features(item_id)

Get features by item ID.

```python
def get_item_features(self, item_id: int) -> torch.Tensor:
    """
    Get features by item ID
    
    Args:
        item_id: Item ID
        
    Returns:
        Item feature vector
    """
```

### SequenceDataset

Dataset wrapper for sequence-level data, primarily used for training TIGER.

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

**Parameters:**
- `base_dataset`: Base dataset instance
- `split`: Data split ("train", "val", "test")
- `semantic_encoder`: Semantic encoder (RQVAE)
- `sequence_config`: Sequence configuration

**Methods:**

#### __len__()

Return dataset size.

```python
def __len__(self) -> int:
    """Return number of user sequences in dataset"""
```

#### __getitem__(idx)

Get single sequence data.

```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    """
    Get single user sequence data
    
    Args:
        idx: Sequence index
        
    Returns:
        Dictionary containing input and target sequences:
        - 'input_ids': Input sequence (torch.Tensor)
        - 'labels': Target sequence (torch.Tensor)  
        - 'attention_mask': Attention mask (torch.Tensor)
    """
```

#### build_sequences()

Build user interaction sequences.

```python
def build_sequences(self) -> List[Dict[str, Any]]:
    """
    Build user interaction sequences
    
    Returns:
        List of sequences, each containing user ID and item ID list
    """
```

#### encode_sequences_to_semantic_ids(sequences)

Encode sequences to semantic IDs.

```python
def encode_sequences_to_semantic_ids(
    self, 
    sequences: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Encode item sequences to semantic ID sequences
    
    Args:
        sequences: Original sequence list
        
    Returns:
        Encoded sequence list
    """
```

## Data Processing Tools

### train_test_split(interactions_df, test_ratio, val_ratio)

Split train, validation and test sets.

```python
def train_test_split(
    interactions_df: pd.DataFrame,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    time_based: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split train, validation and test sets
    
    Args:
        interactions_df: Interactions DataFrame
        test_ratio: Test set ratio
        val_ratio: Validation set ratio
        time_based: Whether to split based on time
        
    Returns:
        (train_df, val_df, test_df): Split DataFrames
    """
```

### create_item_mapping(items_df)

Create item ID mapping.

```python
def create_item_mapping(items_df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Create item ID mapping
    
    Args:
        items_df: Items DataFrame
        
    Returns:
        (id_to_index, index_to_id): ID mapping dictionaries
    """
```

### normalize_features(features)

Normalize feature vectors.

```python
def normalize_features(features: np.ndarray, method: str = "l2") -> np.ndarray:
    """
    Normalize feature vectors
    
    Args:
        features: Feature matrix
        method: Normalization method ("l2", "minmax", "zscore")
        
    Returns:
        Normalized feature matrix
    """
```

## Caching Mechanism

### CacheManager

Manage data processing cache.

```python
class CacheManager:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.pkl"
        
    def exists(self, cache_key: str) -> bool:
        """Check if cache exists"""
        return self.get_cache_path(cache_key).exists()
        
    def save(self, cache_key: str, data: Any) -> None:
        """Save data to cache"""
        cache_path = self.get_cache_path(cache_key)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
            
    def load(self, cache_key: str) -> Any:
        """Load data from cache"""
        cache_path = self.get_cache_path(cache_key)
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
```

## Data Validation

### validate_dataset(dataset)

Validate dataset integrity.

```python
def validate_dataset(dataset: Dataset) -> Dict[str, Any]:
    """
    Validate dataset integrity
    
    Args:
        dataset: Dataset instance
        
    Returns:
        Validation results dictionary
    """
    results = {
        'size': len(dataset),
        'sample_shapes': [],
        'data_types': [],
        'errors': []
    }
    
    try:
        # Check dataset size
        if len(dataset) == 0:
            results['errors'].append("Dataset is empty")
            
        # Check sample shapes and types
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

## Usage Examples

### Create Custom Dataset

```python
from genrec.data.base_dataset import BaseRecommenderDataset
from genrec.data.configs import DatasetConfig

class MyDataset(BaseRecommenderDataset):
    def download(self):
        # Implement download logic
        pass
        
    def load_raw_data(self):
        # Load data
        return {"items": items_df, "interactions": interactions_df}
        
    def preprocess_data(self, raw_data):
        # Preprocess
        return raw_data
        
    def extract_items(self, processed_data):
        return processed_data["items"]
        
    def extract_interactions(self, processed_data):
        return processed_data["interactions"]

# Use dataset
config = DatasetConfig(root_dir="data", split="default")
dataset = MyDataset(config)
items_df, interactions_df = dataset.get_dataset()
```

### Create Item Dataset

```python
from genrec.data.base_dataset import ItemDataset

# Create item dataset
item_dataset = ItemDataset(
    base_dataset=dataset,
    split="train",
    return_text=False
)

# Use DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(item_dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    features = batch  # (batch_size, feature_dim)
    # Training logic
```

### Create Sequence Dataset

```python
from genrec.data.base_dataset import SequenceDataset
from genrec.models.rqvae import RqVae

# Load semantic encoder
semantic_encoder = RqVae.load_from_checkpoint("rqvae.ckpt")

# Create sequence dataset
sequence_dataset = SequenceDataset(
    base_dataset=dataset,
    split="train",
    semantic_encoder=semantic_encoder
)

# Use DataLoader
dataloader = DataLoader(sequence_dataset, batch_size=16, shuffle=True)

for batch in dataloader:
    input_ids = batch['input_ids']  # (batch_size, seq_len)
    labels = batch['labels']        # (batch_size, seq_len)
    # Training logic
```