# RQVAE API Reference

Detailed API documentation for the Residual Quantized Variational Autoencoder (RQVAE).

## Core Classes

### RqVae

Main RQVAE model class.

```python
class RqVae(LightningModule):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        num_embeddings: int = 1024,
        commitment_cost: float = 0.25,
        learning_rate: float = 1e-3
    )
```

**Parameters:**
- `input_dim`: Input feature dimension
- `hidden_dim`: Hidden layer dimension
- `latent_dim`: Latent space dimension
- `num_embeddings`: Number of embedding vectors
- `commitment_cost`: Commitment loss weight
- `learning_rate`: Learning rate

**Methods:**

#### forward(features)

Forward pass computation.

```python
def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        features: Input features (batch_size, input_dim)
    
    Returns:
        reconstructed: Reconstructed features (batch_size, input_dim)
        commitment_loss: Commitment loss
        embedding_loss: Embedding loss
        semantic_ids: Semantic IDs (batch_size,)
    """
```

#### encode(features)

Encode features to latent representation.

```python
def encode(self, features: torch.Tensor) -> torch.Tensor:
    """
    Args:
        features: Input features (batch_size, input_dim)
    
    Returns:
        encoded: Encoded latent representation (batch_size, latent_dim)
    """
```

#### generate_semantic_ids(features)

Generate semantic IDs.

```python
def generate_semantic_ids(self, features: torch.Tensor) -> torch.Tensor:
    """
    Args:
        features: Input features (batch_size, input_dim)
    
    Returns:
        semantic_ids: Semantic IDs (batch_size,)
    """
```

## Component Classes

### VectorQuantizer

Vector quantization layer implementation.

```python
class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25
    )
```

**Parameters:**
- `num_embeddings`: Number of embedding vectors
- `embedding_dim`: Embedding dimension
- `commitment_cost`: Commitment loss weight

**Methods:**

#### forward(inputs)

Quantize input vectors.

```python
def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        inputs: Input vectors (batch_size, embedding_dim)
    
    Returns:
        quantized: Quantized vectors
        commitment_loss: Commitment loss
        embedding_loss: Embedding loss
        encoding_indices: Encoding indices
    """
```

### Encoder

Encoder network.

```python
class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int
    )
```

### Decoder

Decoder network.

```python
class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int
    )
```

## Training Interface

### Training Step

```python
def training_step(self, batch, batch_idx):
    """Training step"""
    features = batch['features']
    
    # Forward pass
    reconstructed, commitment_loss, embedding_loss, semantic_ids = self(features)
    
    # Compute losses
    recon_loss = F.mse_loss(reconstructed, features)
    total_loss = recon_loss + commitment_loss + embedding_loss
    
    # Log metrics
    self.log('train_loss', total_loss)
    self.log('train_recon_loss', recon_loss)
    self.log('train_commitment_loss', commitment_loss)
    self.log('train_embedding_loss', embedding_loss)
    
    return total_loss
```

### Validation Step

```python
def validation_step(self, batch, batch_idx):
    """Validation step"""
    features = batch['features']
    
    # Forward pass
    reconstructed, commitment_loss, embedding_loss, semantic_ids = self(features)
    
    # Compute losses
    recon_loss = F.mse_loss(reconstructed, features)
    total_loss = recon_loss + commitment_loss + embedding_loss
    
    # Log metrics
    self.log('val_loss', total_loss)
    self.log('val_recon_loss', recon_loss)
    
    return total_loss
```

## Configuration Interface

### Optimizer Configuration

```python
def configure_optimizers(self):
    """Configure optimizers"""
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'monitor': 'val_loss'
        }
    }
```

## Utility Functions

### Model Save and Load

```python
# Save model
model.save_pretrained("path/to/model")

# Load model
model = RqVae.load_from_checkpoint("path/to/checkpoint.ckpt")
```

### Batch Inference

```python
def batch_inference(model, dataloader, device='cuda'):
    """Batch inference for semantic ID generation"""
    model.eval()
    model.to(device)
    
    all_semantic_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            semantic_ids = model.generate_semantic_ids(features)
            all_semantic_ids.append(semantic_ids.cpu())
    
    return torch.cat(all_semantic_ids, dim=0)
```

## Evaluation Interface

### Reconstruction Quality Evaluation

```python
def evaluate_reconstruction(model, dataloader, device='cuda'):
    """Evaluate reconstruction quality"""
    model.eval()
    model.to(device)
    
    total_mse = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            reconstructed, _, _, _ = model(features)
            
            mse = F.mse_loss(reconstructed, features, reduction='sum')
            total_mse += mse.item()
            total_samples += features.size(0)
    
    avg_mse = total_mse / total_samples
    return {'mse': avg_mse, 'rmse': avg_mse ** 0.5}
```

### Quantization Quality Evaluation

```python
def evaluate_quantization(model, dataloader, device='cuda'):
    """Evaluate quantization quality"""
    model.eval()
    model.to(device)
    
    all_indices = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            _, _, _, semantic_ids = model(features)
            all_indices.append(semantic_ids.cpu())
    
    all_indices = torch.cat(all_indices, dim=0)
    
    # Compute usage statistics
    unique_codes = len(torch.unique(all_indices))
    total_codes = model.quantizer.num_embeddings
    usage_rate = unique_codes / total_codes
    
    # Compute perplexity
    counts = torch.bincount(all_indices, minlength=total_codes).float()
    probs = counts / counts.sum()
    perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
    
    return {
        'usage_rate': usage_rate,
        'unique_codes': unique_codes,
        'perplexity': perplexity.item()
    }
```

## Usage Examples

### Basic Training

```python
from genrec.models.rqvae import RqVae
from genrec.data.p5_amazon import P5AmazonItemDataset
import pytorch_lightning as pl

# Create dataset
dataset = P5AmazonItemDataset(
    root="dataset/amazon",
    split="beauty",
    train_test_split="train"
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create model
model = RqVae(
    input_dim=768,
    hidden_dim=512,
    latent_dim=256,
    num_embeddings=1024,
    learning_rate=1e-3
)

# Train model
trainer = pl.Trainer(max_epochs=100, gpus=1)
trainer.fit(model, dataloader)
```

### Semantic ID Generation

```python
# Load trained model
model = RqVae.load_from_checkpoint("checkpoints/rqvae.ckpt")
model.eval()

# Generate semantic IDs
with torch.no_grad():
    features = torch.randn(10, 768)  # Example features
    semantic_ids = model.generate_semantic_ids(features)
    print(f"Semantic IDs: {semantic_ids}")
```