# TIGER Training

This guide provides detailed instructions on how to train the TIGER model.

## Prerequisites

### 1. Pre-trained RQVAE Model

TIGER requires a pre-trained RQVAE model to generate semantic IDs:

```bash
# Ensure RQVAE model has been trained
ls out/rqvae/p5_amazon/beauty/checkpoint_*.pt
```

If not available, please complete [RQVAE training](rqvae.md) first.

### 2. Data Preparation

Ensure using the same dataset as RQVAE:

```bash
# Data should already exist
ls dataset/amazon/
```

## Training Configuration

### Default Configuration

View the TIGER configuration file:

```bash
cat config/tiger/p5_amazon.gin
```

Key parameters:

```gin
# Training parameters
train.epochs=5000               # Training epochs
train.learning_rate=3e-4        # Learning rate
train.batch_size=256            # Batch size
train.weight_decay=0.035        # Weight decay

# Model parameters
train.embedding_dim=128         # Embedding dimension
train.attn_dim=512             # Attention dimension
train.dropout=0.3              # Dropout rate
train.num_heads=8              # Number of attention heads
train.n_layers=8               # Number of transformer layers

# Sequence parameters
train.max_seq_len=512          # Maximum sequence length
train.num_item_embeddings=256  # Number of item embeddings
train.num_user_embeddings=2000 # Number of user embeddings
train.sem_id_dim=3             # Semantic ID dimension

# Pre-trained model path
train.pretrained_rqvae_path="./out/rqvae/p5_amazon/beauty/checkpoint_299999.pt"
```

## Start Training

### Basic Training Command

```bash
python genrec/trainers/tiger_trainer.py config/tiger/p5_amazon.gin
```

### Distributed Training

Using multi-GPU training:

```bash
accelerate config
accelerate launch genrec/trainers/tiger_trainer.py config/tiger/p5_amazon.gin
```

### Training Process

During training you'll see:

1. **Data Loading**: Sequence dataset loading and semantic ID generation
2. **Model Initialization**: Transformer model initialization
3. **Training Loop**: Loss reduction and metric monitoring
4. **Validation Evaluation**: Periodic performance assessment

## Custom Configuration

### Creating Custom Configuration

```gin
# my_tiger_config.gin
import genrec.data.p5_amazon

# Adjust model scale
train.embedding_dim=256
train.attn_dim=1024
train.n_layers=12
train.num_heads=16

# Adjust training parameters
train.learning_rate=1e-4
train.batch_size=128
train.epochs=10000

# Custom paths
train.dataset_folder="my_dataset"
train.pretrained_rqvae_path="my_rqvae/checkpoint.pt"
train.save_dir_root="my_tiger_output/"

# Experiment tracking
train.wandb_logging=True
train.wandb_project="my_tiger_experiment"
```

## Model Architecture Overview

### Transformer Structure

TIGER uses an encoder-decoder architecture:

```python
class Tiger(nn.Module):
    def __init__(self, config):
        # User and item embeddings
        self.user_embedding = UserIdEmbedding(...)
        self.item_embedding = SemIdEmbedding(...)
        
        # Transformer encoder-decoder
        self.transformer = TransformerEncoderDecoder(...)
        
        # Output projection
        self.output_projection = nn.Linear(...)
```

### Semantic ID Mapping

TIGER converts items to semantic ID sequences:

```python
# Item -> Semantic ID sequence
item_id = 123
semantic_ids = rqvae.get_semantic_ids(item_features[item_id])
# semantic_ids: [45, 67, 89]  # length = sem_id_dim
```

## Training Monitoring

### Key Metrics

- **Training Loss**: Sequence modeling loss
- **Validation Loss**: Validation set performance
- **Recall@K**: Top-K recall rate
- **NDCG@K**: Normalized Discounted Cumulative Gain

### Weights & Biases Integration

Enable experiment tracking:

```gin
train.wandb_logging=True
train.wandb_project="tiger_p5_amazon"
train.wandb_log_interval=100
```

View training curves:
- Visit [wandb.ai](https://wandb.ai)
- Find your project and experiment

## Model Evaluation

### Recommendation Quality Assessment

```python
from genrec.models.tiger import Tiger
from genrec.modules.metrics import TopKAccumulator

# Load model
model = Tiger.load_from_checkpoint("out/tiger/checkpoint.pt")

# Create evaluator
evaluator = TopKAccumulator(k=[5, 10, 20])

# Evaluate on test set
test_dataloader = DataLoader(test_dataset, batch_size=256)
metrics = evaluator.evaluate(model, test_dataloader)

print(f"Recall@10: {metrics['recall@10']:.4f}")
print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
```

### Generative Recommendation

```python
def generate_recommendations(model, user_sequence, top_k=10):
    """Generate recommendations for user"""
    model.eval()
    
    with torch.no_grad():
        # Encode user sequence
        sequence_embedding = model.encode_sequence(user_sequence)
        
        # Generate recommendations
        logits = model.generate(sequence_embedding, max_length=top_k)
        
        # Get Top-K items
        recommendations = torch.topk(logits, top_k).indices
    
    return recommendations.tolist()

# Usage example
user_history = [item1_semantic_ids, item2_semantic_ids, ...]
recommendations = generate_recommendations(model, user_history, top_k=10)
```

## Advanced Features

### Trie-Constrained Generation

TIGER supports Trie-based constrained generation:

```python
from genrec.models.tiger import build_trie

# Build Trie for valid item IDs
valid_items = torch.tensor([[1, 2, 3], [4, 5, 6], ...])  # Semantic ID sequences
trie = build_trie(valid_items)

# Constrained generation
constrained_output = model.generate_with_trie(
    user_sequence, 
    trie=trie,
    max_length=10
)
```

### Sequence Augmentation

Training supports sequence augmentation:

```gin
train.subsample=True  # Dynamic subsampling
train.augmentation=True  # Sequence augmentation
```

## Troubleshooting

### Common Issues

**Q: RQVAE checkpoint not found?**

A: Check if path is correct:
```bash
# Confirm file exists
ls -la out/rqvae/p5_amazon/beauty/checkpoint_299999.pt

# Update path in config file
train.pretrained_rqvae_path="actual_checkpoint_path"
```

**Q: Training is slow?**

A: Optimization suggestions:
- Increase batch size: `train.batch_size=512`
- Reduce sequence length: `train.max_seq_len=256`
- Use multi-GPU training

**Q: Poor recommendation performance?**

A: Tuning suggestions:
- Increase model size: `train.n_layers=12`
- Adjust learning rate: `train.learning_rate=1e-4`
- Increase training epochs: `train.epochs=10000`

### Debugging Tips

1. **Check semantic ID generation**:
```python
# Verify RQVAE is working correctly
rqvae = RqVae.load_from_checkpoint(pretrained_path)
sample_item = dataset[0]
semantic_ids = rqvae.get_semantic_ids(sample_item)
print(f"Semantic IDs: {semantic_ids}")
```

2. **Monitor attention weights**:
```python
# Check if model learns meaningful attention patterns
attention_weights = model.get_attention_weights(user_sequence)
print(f"Attention shape: {attention_weights.shape}")
```

## Performance Optimization

### Memory Optimization

```gin
# Reduce memory usage
train.gradient_accumulate_every=4  # Gradient accumulation
train.batch_size=64               # Smaller batch size
train.max_seq_len=256            # Shorter sequences
```

### Mixed Precision Training

```gin
train.mixed_precision_type="fp16"  # Use half precision
```

## Experiment Suggestions

### Hyperparameter Grid Search

```python
# Suggested hyperparameter ranges
learning_rates = [1e-4, 3e-4, 1e-3]
batch_sizes = [128, 256, 512]
model_dims = [128, 256, 512]
n_layers = [6, 8, 12]

for lr in learning_rates:
    for bs in batch_sizes:
        # Create config and train
        config = create_config(lr=lr, batch_size=bs)
        train_model(config)
```

### A/B Testing

Compare different architectures:

```gin
# Experiment A: Standard TIGER
train.n_layers=8
train.num_heads=8

# Experiment B: Deeper model
train.n_layers=12
train.num_heads=16

# Experiment C: Wider model
train.embedding_dim=256
train.attn_dim=1024
```

## Next Steps

After training completion:

1. [Evaluate recommendation effectiveness](../models/tiger.md#evaluation-metrics)
2. [Deploy to production environment](../deployment.md)
3. [Try other datasets](../dataset/custom.md)