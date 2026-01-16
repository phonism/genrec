# Custom Datasets

This guide explains how to add custom datasets to the genrec framework.

## Basic Concepts

### Dataset Types

The framework supports two main dataset types:

1. **ItemDataset**: Item-level datasets for training RQVAE
2. **SequenceDataset**: Sequence-level datasets for training TIGER

### Base Architecture

All datasets inherit from `BaseRecommenderDataset`:

```python
from genrec.data.base_dataset import BaseRecommenderDataset

class MyCustomDataset(BaseRecommenderDataset):
    def __init__(self, config):
        super().__init__(config)
        # Initialize custom parameters
        
    def download(self):
        # Implement data download logic
        pass
        
    def load_raw_data(self):
        # Load raw data files
        pass
        
    def preprocess_data(self, raw_data):
        # Preprocess data
        pass
        
    def extract_items(self, processed_data):
        # Extract item information
        pass
        
    def extract_interactions(self, processed_data):
        # Extract user interaction information
        pass
```

## Implementation Steps

### 1. Create Configuration Class

First define dataset-specific configuration:

```python
from dataclasses import dataclass
from genrec.data.configs import DatasetConfig

@dataclass
class MyDatasetConfig(DatasetConfig):
    # Dataset-specific parameters
    api_key: str = ""
    data_format: str = "json"
    categories: List[str] = None
```

### 2. Implement Data Download

```python
def download(self):
    """Download dataset to local storage"""
    if self._data_exists():
        return
        
    print("Downloading custom dataset...")
    
    # Example: Download from API
    import requests
    response = requests.get(f"https://api.example.com/data?key={self.config.api_key}")
    
    # Save locally
    data_path = self.root_path / "raw" / "data.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(data_path, 'w') as f:
        json.dump(response.json(), f)
    
    print("Download completed.")

def _data_exists(self):
    """Check if data already exists"""
    data_path = self.root_path / "raw" / "data.json"
    return data_path.exists()
```

### 3. Implement Data Loading

```python
def load_raw_data(self):
    """Load raw data"""
    data_path = self.root_path / "raw" / "data.json"
    
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    
    # Parse data structure
    users = raw_data.get('users', [])
    items = raw_data.get('items', [])
    interactions = raw_data.get('interactions', [])
    
    return {
        "users": pd.DataFrame(users),
        "items": pd.DataFrame(items), 
        "interactions": pd.DataFrame(interactions)
    }
```

### 4. Implement Data Preprocessing

```python
def preprocess_data(self, raw_data):
    """Preprocess data"""
    # Clean item data
    items_df = self._clean_items(raw_data["items"])
    
    # Clean interaction data
    interactions_df = self._clean_interactions(raw_data["interactions"])
    
    # Filter low-frequency users and items
    interactions_df = self.filter_low_interactions(
        interactions_df,
        min_user_interactions=self.config.processing_config.min_user_interactions,
        min_item_interactions=self.config.processing_config.min_item_interactions
    )
    
    # Process item features
    items_df = self._process_item_features(items_df)
    
    return {
        "items": items_df,
        "interactions": interactions_df
    }

def _clean_items(self, items_df):
    """Clean item data"""
    # Fill missing values
    items_df["title"] = items_df["title"].fillna("Unknown")
    items_df["category"] = items_df["category"].fillna("Unknown")
    
    # Standardize text
    items_df["title"] = items_df["title"].str.strip()
    
    return items_df

def _clean_interactions(self, interactions_df):
    """Clean interaction data"""
    # Remove duplicate interactions
    interactions_df = interactions_df.drop_duplicates(["user_id", "item_id"])
    
    # Add timestamp if missing
    if "timestamp" not in interactions_df.columns:
        interactions_df["timestamp"] = range(len(interactions_df))
    
    return interactions_df
```

### 5. Implement Feature Extraction

```python
def _process_item_features(self, items_df):
    """Process item features"""
    # Use text processor to generate embeddings
    cache_key = f"custom_dataset_{self.config.split}"
    embeddings = self.text_processor.encode_item_features(
        items_df,
        cache_key=cache_key,
        force_reload=self.config.force_reload
    )
    
    # Add embedding features
    items_df = items_df.copy()
    items_df["features"] = embeddings.tolist()
    
    # Create text fields for reference
    texts = []
    for _, row in items_df.iterrows():
        text = f"Title: {row['title']}; Category: {row['category']}"
        texts.append(text)
    
    items_df["text"] = texts
    
    return items_df

def extract_items(self, processed_data):
    """Extract item information"""
    return processed_data["items"]

def extract_interactions(self, processed_data):
    """Extract interaction information"""
    return processed_data["interactions"]
```

## Create Dataset Wrappers

### Item Dataset

```python
from genrec.data.base_dataset import ItemDataset
import gin

@gin.configurable
class MyItemDataset(ItemDataset):
    """Custom item dataset"""
    
    def __init__(
        self,
        root: str,
        split: str = "default",
        train_test_split: str = "all",
        api_key: str = "",
        **kwargs
    ):
        # Create configuration
        config = MyDatasetConfig(
            root_dir=root,
            split=split,
            api_key=api_key,
            **kwargs
        )
        
        # Create base dataset
        base_dataset = MyCustomDataset(config)
        
        # Initialize item dataset
        super().__init__(
            base_dataset=base_dataset,
            split=train_test_split,
            return_text=False
        )
```

### Sequence Dataset

```python
from genrec.data.base_dataset import SequenceDataset

@gin.configurable
class MySequenceDataset(SequenceDataset):
    """Custom sequence dataset"""
    
    def __init__(
        self,
        root: str,
        split: str = "default",
        train_test_split: str = "train",
        api_key: str = "",
        pretrained_rqvae_path: Optional[str] = None,
        **kwargs
    ):
        # Create configuration
        config = MyDatasetConfig(
            root_dir=root,
            split=split,
            api_key=api_key,
            **kwargs
        )
        
        # Load semantic encoder
        semantic_encoder = None
        if pretrained_rqvae_path:
            from genrec.models.rqvae import RqVae
            semantic_encoder = RqVae.load_from_checkpoint(pretrained_rqvae_path)
            semantic_encoder.eval()
        
        # Create base dataset
        base_dataset = MyCustomDataset(config)
        
        # Initialize sequence dataset
        super().__init__(
            base_dataset=base_dataset,
            split=train_test_split,
            semantic_encoder=semantic_encoder
        )
```

## Register Dataset

### Using Factory Pattern

```python
from genrec.data.dataset_factory import DatasetFactory

# Register dataset
DatasetFactory.register_dataset(
    "my_dataset",
    base_class=MyCustomDataset,
    item_class=MyItemDataset,
    sequence_class=MySequenceDataset
)

# Use factory to create dataset
item_dataset = DatasetFactory.create_item_dataset(
    "my_dataset",
    root_dir="path/to/data",
    api_key="your_api_key"
)
```

## Configuration File Integration

### Gin Configuration File

Create configuration file `config/my_dataset.gin`:

```gin
import my_module.my_dataset

# Dataset parameters
train.dataset_folder="dataset/my_dataset"
train.dataset=@MyItemDataset

# Custom parameters
MyItemDataset.api_key="your_api_key"
MyItemDataset.split="category_a"

# Text encoding parameters
MyItemDataset.encoder_model_name="sentence-transformers/all-MiniLM-L6-v2"
```

## Testing and Validation

### Unit Tests

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
        """Test data loading"""
        # Mock data
        raw_data = self.dataset.load_raw_data()
        self.assertIn("items", raw_data)
        self.assertIn("interactions", raw_data)
    
    def test_preprocessing(self):
        """Test preprocessing"""
        raw_data = {"items": pd.DataFrame(), "interactions": pd.DataFrame()}
        processed = self.dataset.preprocess_data(raw_data)
        self.assertIn("items", processed)
        self.assertIn("interactions", processed)
```

### Data Quality Validation

```python
def validate_dataset(dataset):
    """Validate dataset quality"""
    # Check data completeness
    assert len(dataset) > 0, "Dataset is empty"
    
    # Check feature dimensions
    sample = dataset[0]
    assert len(sample) == 768, f"Wrong feature dimension: {len(sample)}"
    
    # Check data types
    assert isinstance(sample, list), "Wrong data type"
    
    print("Dataset validation passed")
```

## Best Practices

### 1. Error Handling

```python
def load_raw_data(self):
    try:
        # Data loading logic
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {self.data_path}")
    except Exception as e:
        raise RuntimeError(f"Data loading failed: {str(e)}")
```

### 2. Logging

```python
import logging

logger = logging.getLogger(__name__)

def preprocess_data(self, raw_data):
    logger.info("Starting data preprocessing")
    
    # Preprocessing logic
    
    logger.info(f"Preprocessing completed, items: {len(items_df)}, interactions: {len(interactions_df)}")
```

### 3. Configuration Validation

```python
def __post_init__(self):
    super().__post_init__()
    
    if not self.api_key:
        raise ValueError("API key cannot be empty")
    
    if self.data_format not in ["json", "csv"]:
        raise ValueError(f"Unsupported data format: {self.data_format}")
```

## Example: MovieLens Dataset

Complete MovieLens dataset implementation example:

```python
@dataclass
class MovieLensConfig(DatasetConfig):
    """MovieLens dataset configuration"""
    version: str = "1m"  # 1m, 10m, 20m
    
class MovieLensDataset(BaseRecommenderDataset):
    """MovieLens dataset implementation"""
    
    URLS = {
        "1m": "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "10m": "http://files.grouplens.org/datasets/movielens/ml-10m.zip",
    }
    
    def download(self):
        if self._data_exists():
            return
            
        url = self.URLS[self.config.version]
        # Download and extract logic
        
    def load_raw_data(self):
        # Load ratings.dat, movies.dat, users.dat
        pass
        
    def preprocess_data(self, raw_data):
        # MovieLens-specific preprocessing
        pass
```

Through these steps, you can successfully add custom dataset support to the genrec framework.