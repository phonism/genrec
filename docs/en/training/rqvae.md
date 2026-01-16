# RQVAE Training

This guide provides detailed instructions on how to train the RQVAE model.

## Training Preparation

### 1. Data Preparation

Ensure the dataset is downloaded and placed in the correct location:

```bash
# Data will be automatically downloaded to the specified directory
mkdir -p dataset/amazon
```

### 2. Check Configuration File

View the default configuration:

```bash
cat config/rqvae/p5_amazon.gin
```

Key configuration parameters:

```gin
# Training parameters
train.iterations=400000          # Number of training iterations
train.learning_rate=0.0005      # Learning rate
train.batch_size=64             # Batch size
train.weight_decay=0.01         # Weight decay

# Model parameters
train.vae_input_dim=768         # Input dimension
train.vae_embed_dim=32          # Embedding dimension
train.vae_hidden_dims=[512, 256, 128]  # Hidden layer dimensions
train.vae_codebook_size=256     # Codebook size
train.vae_n_layers=3            # Number of quantization layers

# Quantization settings
train.vae_codebook_mode=%genrec.models.rqvae.QuantizeForwardMode.ROTATION_TRICK
train.commitment_weight=0.25    # Commitment loss weight
```

## Start Training

### Basic Training Command

```bash
python genrec/trainers/rqvae_trainer.py config/rqvae/p5_amazon.gin
```

### Training Monitoring

If Weights & Biases is enabled:

```gin
train.wandb_logging=True
train.wandb_project="my_rqvae_project"
```

### GPU Training

Using multi-GPU training:

```bash
accelerate config  # Configure on first run
accelerate launch genrec/trainers/rqvae_trainer.py config/rqvae/p5_amazon.gin
```

## Custom Configuration

### Creating Custom Configuration File

```gin
# my_rqvae_config.gin
import genrec.data.p5_amazon
import genrec.models.rqvae

# Custom training parameters
train.iterations=200000
train.batch_size=32
train.learning_rate=0.001

# Custom model architecture
train.vae_embed_dim=64
train.vae_hidden_dims=[512, 256, 128, 64]
train.vae_codebook_size=512

# Data paths
train.dataset_folder="path/to/my/dataset"
train.save_dir_root="path/to/my/output"

# Experiment tracking
train.wandb_logging=True
train.wandb_project="custom_rqvae_experiment"
```

Using custom configuration:

```bash
python genrec/trainers/rqvae_trainer.py my_rqvae_config.gin
```

## Training Monitoring

### Key Metrics

Monitor these metrics during training:

- **Total Loss**: Overall training loss
- **Reconstruction Loss**: Reconstruction quality
- **Quantization Loss**: Quantization effectiveness
- **Commitment Loss**: Encoder commitment

### Sample Log Output

```
Epoch 1000: Loss=2.3456, Recon=2.1234, Quant=0.1234, Commit=0.0988
Epoch 2000: Loss=1.9876, Recon=1.8234, Quant=0.0987, Commit=0.0655
...
```

## Model Evaluation

### Reconstruction Quality Assessment

```python
from genrec.models.rqvae import RqVae
from genrec.data.p5_amazon import P5AmazonItemDataset

# Load trained model
model = RqVae.load_from_checkpoint("out/rqvae/checkpoint_299999.pt")

# Evaluation dataset
eval_dataset = P5AmazonItemDataset(
    root="dataset/amazon",
    train_test_split="eval"
)

# Calculate reconstruction loss
model.eval()
with torch.no_grad():
    eval_loss = model.evaluate(eval_dataset)
    print(f"Evaluation loss: {eval_loss:.4f}")
```

### Codebook Utilization Analysis

```python
def analyze_codebook_usage(model, dataloader):
    used_codes = set()
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch)
            semantic_ids = outputs.sem_ids
            used_codes.update(semantic_ids.flatten().tolist())
    
    usage_rate = len(used_codes) / model.codebook_size
    print(f"Codebook usage: {usage_rate:.2%}")
    print(f"Used codes: {len(used_codes)}/{model.codebook_size}")
    
    return used_codes
```

## Troubleshooting

### Common Issues

**Q: Training loss doesn't converge?**

A: Try these solutions:
- Lower learning rate: `train.learning_rate=0.0001`
- Adjust commitment weight: `train.commitment_weight=0.1`
- Check if data preprocessing is correct

**Q: Codebook collapse (all samples use the same code)?**

A: 
- Use ROTATION_TRICK mode
- Increase commitment weight
- Reduce learning rate

**Q: GPU out of memory?**

A:
- Reduce batch size: `train.batch_size=32`
- Reduce model size: `train.vae_hidden_dims=[256, 128]`
- Enable mixed precision training

### Debugging Tips

1. **Gradient checking**:
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: {grad_norm:.6f}")
```

2. **Loss analysis**:
```python
# Print individual loss components
print(f"Reconstruction: {outputs.reconstruction_loss:.4f}")
print(f"Quantization: {outputs.quantization_loss:.4f}")
print(f"Commitment: {outputs.commitment_loss:.4f}")
```

## Best Practices

### Hyperparameter Tuning Recommendations

1. **Learning rate scheduling**:
```gin
# Use cosine annealing
train.scheduler="cosine"
train.min_lr=1e-6
```

2. **Early stopping strategy**:
```gin
train.early_stopping=True
train.patience=10000
```

3. **Model saving frequency**:
```gin
train.save_model_every=50000  # Save every 50k iterations
train.eval_every=10000        # Evaluate every 10k iterations
```

### Experiment Management

Recommended to use version control and experiment tracking:

```bash
# Create experiment branch
git checkout -b experiment/rqvae-large-codebook

# Modify configuration
vim config/rqvae/large_codebook.gin

# Run experiment
python genrec/trainers/rqvae_trainer.py config/rqvae/large_codebook.gin

# Record results
git add .
git commit -m "Experiment: large codebook (size=1024)"
```

## Next Steps

After training completion, you can:

1. Use the trained RQVAE for [TIGER training](tiger.md)
2. Analyze [model performance](../models/rqvae.md#evaluation-metrics)
3. Try [different datasets](../dataset/custom.md)