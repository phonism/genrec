# P5 Amazon Dataset

The P5 Amazon dataset is a large-scale product recommendation dataset containing user reviews and product metadata from Amazon.

## Overview

P5 Amazon is derived from the Amazon product data and includes rich textual information about products along with user interaction histories. It's specifically designed for training generative recommendation models.

**Key Features:**
- Multiple product categories (Beauty, Electronics, Sports, etc.)
- Rich product metadata (title, brand, category, price, description)
- User interaction sequences with timestamps
- Pre-processed text features suitable for neural models

## Dataset Structure

### Data Files

When downloaded, the dataset contains:
```
dataset/amazon/
├── raw/
│   ├── beauty_5.json.gz          # Raw interaction data
│   ├── meta_beauty.json.gz       # Product metadata
│   └── ...
├── processed/
│   ├── items.parquet             # Processed item features
│   ├── interactions.parquet      # Processed interactions
│   └── mappings.pkl              # ID mappings
└── cache/
    └── text_embeddings/          # Cached text embeddings
```

### Data Format

**Items DataFrame:**
| Column | Type | Description |
|--------|------|-------------|
| item_id | int | Unique item identifier |
| title | str | Product title |
| brand | str | Product brand |
| category | str | Product category |
| price | float | Product price |
| features | List[float] | Text embedding features (768-dim) |
| text | str | Formatted text for reference |

**Interactions DataFrame:**
| Column | Type | Description |
|--------|------|-------------|
| user_id | int | Unique user identifier |
| item_id | int | Item identifier |
| rating | float | User rating (1-5) |
| timestamp | int | Interaction timestamp |

## Available Categories

### Beauty
- **Size**: ~52K products, ~1.2M interactions
- **Description**: Cosmetics, skincare, haircare products
- **Features**: Rich brand and category information

### Electronics  
- **Size**: ~63K products, ~1.7M interactions
- **Description**: Electronic devices, accessories, gadgets
- **Features**: Technical specifications in descriptions

### Sports
- **Size**: ~35K products, ~296K interactions  
- **Description**: Sports equipment, outdoor gear, fitness products
- **Features**: Activity-specific categorization

### All Categories
- **Total**: 29 categories available
- **Combined Size**: Multi-GB dataset
- **Use Case**: Large-scale experiments

## Usage

### Basic Loading

```python
from genrec.data.p5_amazon import P5AmazonItemDataset

# Load beauty category for item-level training
dataset = P5AmazonItemDataset(
    root="dataset/amazon",
    split="beauty",
    train_test_split="train"
)

print(f"Dataset size: {len(dataset)}")
print(f"Feature dimension: {dataset[0].shape}")
```

### Sequence Data

```python
from genrec.data.p5_amazon import P5AmazonSequenceDataset

# Load for sequence modeling (requires pre-trained RQVAE)
seq_dataset = P5AmazonSequenceDataset(
    root="dataset/amazon", 
    split="beauty",
    train_test_split="train",
    pretrained_rqvae_path="checkpoints/rqvae_beauty.ckpt"
)

# Get a sample sequence
sample = seq_dataset[0]
print(f"Input sequence: {sample['input_ids']}")
print(f"Target sequence: {sample['labels']}")
```

### Configuration Options

```python
from genrec.data.p5_amazon import P5AmazonItemDataset

dataset = P5AmazonItemDataset(
    root="dataset/amazon",
    split="beauty",
    train_test_split="train",
    
    # Text encoding options
    encoder_model_name="sentence-transformers/all-MiniLM-L6-v2",
    force_reload=False,  # Use cached embeddings if available
    
    # Data filtering
    min_user_interactions=5,
    min_item_interactions=5,
    
    # Text template
    text_template="Title: {title}; Brand: {brand}; Category: {category}"
)
```

## Data Processing Pipeline

### 1. Download and Extraction

```python
# Automatic download from UCSD repository
dataset = P5AmazonItemDataset(root="dataset/amazon", split="beauty")
# Downloads ~500MB for beauty category
```

### 2. Text Processing

The framework automatically processes product text using sentence transformers:

```python
# Default text template
template = "Title: {title}; Brand: {brand}; Category: {category}; Price: {price}"

# Example processed text
"Title: Maybelline Mascara; Brand: Maybelline; Category: Beauty; Price: $8.99"
```

### 3. Feature Extraction

- **Text Embeddings**: 768-dimensional vectors from sentence transformers
- **Caching**: Embeddings cached for faster subsequent loading
- **Normalization**: L2 normalization applied by default

### 4. Sequence Building

For TIGER training, interactions are converted to sequences:

```python
# User interaction history
user_history = [item1, item2, item3, item4]

# Converted to semantic ID sequence using RQVAE
semantic_sequence = [1, 45, 123, 67, 234, 189, 45, 123, 567, 234, 88, 192]
#                   |--item1--| |--item2--| |--item3--| |--item4--|
```

## Statistics

### Beauty Category
```
Items: 52,024
Users: 40,226  
Interactions: 1,235,316
Density: 0.059%
Avg items per user: 30.7
Avg users per item: 23.7
```

### Electronics Category
```
Items: 63,001
Users: 192,403
Interactions: 1,689,188
Density: 0.014%
Avg items per user: 8.8
Avg users per item: 26.8
```

## Data Quality

### Preprocessing Steps

1. **Duplicate Removal**: Remove duplicate user-item interactions
2. **Low Activity Filtering**: Filter users/items with < 5 interactions
3. **Text Cleaning**: Normalize titles, handle missing brands/categories
4. **Price Processing**: Clean and standardize price formats
5. **ID Remapping**: Create contiguous ID mappings

### Quality Checks

```python
from genrec.data.p5_amazon import P5AmazonItemDataset

dataset = P5AmazonItemDataset(root="dataset/amazon", split="beauty")

# Check data quality
items_df, interactions_df = dataset.base_dataset.get_dataset()

print("Data Quality Report:")
print(f"Items with missing titles: {items_df['title'].isna().sum()}")
print(f"Items with missing brands: {items_df['brand'].isna().sum()}")
print(f"Interactions with valid ratings: {(interactions_df['rating'] > 0).sum()}")
print(f"Feature vector dimension: {len(items_df.iloc[0]['features'])}")
```

## Advanced Usage

### Custom Text Templates

```python
# Product-focused template
template = "Product: {title} from {brand} in {category} category"

# Price-aware template  
template = "Buy {title} by {brand} for ${price} in {category}"

# Minimal template
template = "{title} - {brand}"
```

### Batch Processing

```python
from torch.utils.data import DataLoader

dataset = P5AmazonItemDataset(root="dataset/amazon", split="beauty")
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

for batch in dataloader:
    # batch shape: (128, 768)
    features = batch
    # Process batch...
```

### Multi-Category Loading

```python
# Load multiple categories
categories = ["beauty", "electronics", "sports"]
datasets = []

for category in categories:
    dataset = P5AmazonItemDataset(
        root="dataset/amazon",
        split=category,
        train_test_split="train"
    )
    datasets.append(dataset)

# Combine datasets
from torch.utils.data import ConcatDataset
combined_dataset = ConcatDataset(datasets)
```

## Troubleshooting

### Common Issues

**Q: Download fails with network error**
A: Check internet connection and try again. The files are large (100MB-2GB).

**Q: Text encoding takes very long**
A: Use cached embeddings by setting `force_reload=False` and ensure cache directory is writable.

**Q: Out of memory during loading**
A: Reduce batch size or use a smaller category like "beauty" instead of "all".

**Q: Missing brand/category information**
A: This is normal - the dataset fills missing values with "Unknown".

### Performance Tips

```python
# Use caching for faster subsequent loads
dataset = P5AmazonItemDataset(
    root="dataset/amazon",
    split="beauty", 
    force_reload=False  # Use cache
)

# Use lighter text encoder for faster processing
dataset = P5AmazonItemDataset(
    root="dataset/amazon",
    split="beauty",
    encoder_model_name="sentence-transformers/all-MiniLM-L6-v2"  # Smaller model
)

# Process in smaller batches
from genrec.data.configs import TextEncodingConfig
text_config = TextEncodingConfig(batch_size=16)  # Reduce from default 32
```

## Citation

If you use the P5 Amazon dataset, please cite:

```bibtex
@article{geng2022recommendation,
  title={Recommendation as language processing (rlp): A unified pretrain, personalized prompt \& predict paradigm (p5)},
  author={Geng, Shijie and Liu, Shuchang and Fu, Zuohui and Ge, Yingqiang and Zhang, Yongfeng},
  journal={arXiv preprint arXiv:2203.13366},
  year={2022}
}
```

## Related Documentation

- [Dataset Overview](overview.md) - General dataset concepts
- [Custom Datasets](custom.md) - Creating your own datasets  
- [RQVAE Training](../training/rqvae.md) - Training item encoders
- [TIGER Training](../training/tiger.md) - Training sequence models