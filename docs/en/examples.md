# Examples

This page contains practical examples for using genrec.

## Basic Usage Examples

### Training RQVAE from Scratch

```python
import torch
from genrec.models.rqvae import RqVae, QuantizeForwardMode
from genrec.data.p5_amazon import P5AmazonItemDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = P5AmazonItemDataset(
    root="dataset/amazon",
    split="beauty",
    train_test_split="train"
)

# Create model
model = RqVae(
    input_dim=768,
    embed_dim=32,
    hidden_dims=[512, 256, 128],
    codebook_size=256,
    n_layers=3,
    commitment_weight=0.25,
    codebook_mode=QuantizeForwardMode.ROTATION_TRICK
)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

for epoch in range(100):
    for batch in dataloader:
        optimizer.zero_grad()
        
        outputs = model(torch.tensor(batch))
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Using Dataset Factory

```python
from genrec.data.dataset_factory import DatasetFactory

# Create item dataset
item_dataset = DatasetFactory.create_item_dataset(
    "p5_amazon",
    "dataset/amazon",
    split="train"
)

# Create sequence dataset  
sequence_dataset = DatasetFactory.create_sequence_dataset(
    "p5_amazon", 
    "dataset/amazon",
    split="train",
    pretrained_rqvae_path="./checkpoints/rqvae.pt"
)
```

### Custom Configuration

```python
from genrec.data.configs import P5AmazonConfig, TextEncodingConfig

# Custom text encoding config
text_config = TextEncodingConfig(
    encoder_model="sentence-transformers/all-MiniLM-L6-v2",
    template="Product: {title} | Brand: {brand} | Category: {categories}",
    batch_size=32
)

# Custom dataset config
dataset_config = P5AmazonConfig(
    root_dir="my_data",
    split="electronics",
    text_config=text_config
)
```

## Advanced Examples

### Multi-GPU Training

```python
from accelerate import Accelerator

def train_with_accelerate():
    accelerator = Accelerator()
    
    # Model, optimizer, dataloader
    model = RqVae(...)
    optimizer = torch.optim.AdamW(model.parameters())
    dataloader = DataLoader(...)
    
    # Prepare for distributed training
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            
            with accelerator.autocast():
                outputs = model(batch)
                loss = outputs.loss
                
            accelerator.backward(loss)
            optimizer.step()
```

### Custom Dataset Implementation

```python
from genrec.data.base_dataset import BaseRecommenderDataset

class MyCustomDataset(BaseRecommenderDataset):
    def download(self):
        # Implement data download logic
        pass
        
    def load_raw_data(self):
        # Load your raw data files
        return {"items": items_df, "interactions": interactions_df}
        
    def preprocess_data(self, raw_data):
        # Custom preprocessing
        return processed_data
        
    def extract_items(self, processed_data):
        return processed_data["items"]
        
    def extract_interactions(self, processed_data):
        return processed_data["interactions"]
```

## Integration Examples

### Weights & Biases Integration

```python
import wandb

# Initialize wandb
wandb.init(
    project="my-recommendation-project",
    config={
        "learning_rate": 0.0005,
        "batch_size": 64,
        "model_type": "rqvae"
    }
)

# Log metrics during training
for epoch in range(epochs):
    # ... training code ...
    
    wandb.log({
        "epoch": epoch,
        "loss": loss.item(),
        "reconstruction_loss": recon_loss.item(),
        "quantization_loss": quant_loss.item()
    })
```

### Hyperparameter Tuning

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    embed_dim = trial.suggest_categorical("embed_dim", [16, 32, 64])
    
    # Train model with suggested parameters
    model = RqVae(embed_dim=embed_dim, ...)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Training loop
    val_loss = train_and_evaluate(model, optimizer, batch_size)
    
    return val_loss

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
```

## Evaluation Examples

### Model Evaluation

```python
def evaluate_model(model, test_dataloader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            
            total_loss += outputs.loss.item() * len(batch)
            total_samples += len(batch)
    
    return total_loss / total_samples

# Evaluate RQVAE
test_loss = evaluate_model(rqvae_model, test_dataloader, device)
print(f"Test reconstruction loss: {test_loss:.4f}")
```

### Recommendation Generation

```python
def generate_recommendations(tiger_model, user_sequence, top_k=10):
    """Generate top-K recommendations for a user sequence"""
    tiger_model.eval()
    
    with torch.no_grad():
        # Encode user sequence
        logits = tiger_model.generate(user_sequence, max_length=top_k)
        
        # Get top-K items
        top_items = torch.topk(logits, top_k).indices
        
    return top_items.tolist()

# Generate recommendations
user_seq = [1, 5, 23, 45]  # User's interaction history
recommendations = generate_recommendations(tiger_model, user_seq, top_k=10)
print(f"Recommended items: {recommendations}")
```

## Utilities and Helpers

### Data Analysis

```python
from genrec.data.processors.sequence_processor import SequenceStatistics

# Analyze sequence statistics
stats = SequenceStatistics.compute_sequence_stats(sequence_data)
print(f"Average sequence length: {stats['avg_seq_length']:.2f}")
print(f"Number of unique items: {stats['num_unique_items']}")
```

### Model Inspection

```python
def inspect_codebook_usage(rqvae_model, dataloader):
    """Analyze codebook utilization"""
    used_codes = set()
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = rqvae_model(batch)
            semantic_ids = outputs.sem_ids
            used_codes.update(semantic_ids.flatten().tolist())
    
    usage_rate = len(used_codes) / rqvae_model.codebook_size
    print(f"Codebook usage: {usage_rate:.2%}")
    
    return used_codes

used_codes = inspect_codebook_usage(model, dataloader)
```

## Tips and Best Practices

### Memory Optimization

```python
# Enable gradient checkpointing for large models
model.gradient_checkpointing_enable()

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(batch)
        loss = outputs.loss
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Debugging

```python
# Enable detailed logging
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Log model statistics
def log_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
```