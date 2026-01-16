# Trainers API Reference

Detailed documentation for training utilities and scripts.

## Base Trainer

### BaseTrainer

Base class for all trainers.

```python
class BaseTrainer:
    """Base trainer class"""
    
    def __init__(
        self,
        model: LightningModule,
        config: TrainingConfig,
        logger: Optional[Logger] = None
    ):
        self.model = model
        self.config = config
        self.logger = logger or self._create_default_logger()
        self.trainer = None
        
    def _create_default_logger(self) -> Logger:
        """Create default logger"""
        from pytorch_lightning.loggers import TensorBoardLogger
        
        return TensorBoardLogger(
            save_dir=self.config.log_dir,
            name=self.config.experiment_name,
            version=self.config.version
        )
```

**Methods:**

#### setup_trainer()

Setup PyTorch Lightning trainer.

```python
def setup_trainer(self) -> None:
    """Setup trainer"""
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    
    # Callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=self.config.checkpoint_dir,
        filename='{epoch}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=self.config.save_top_k,
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    if self.config.early_stopping_patience > 0:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            mode='min'
        )
        callbacks.append(early_stop_callback)
    
    # Create trainer
    self.trainer = Trainer(
        max_epochs=self.config.max_epochs,
        gpus=self.config.gpus,
        precision=self.config.precision,
        gradient_clip_val=self.config.gradient_clip_val,
        accumulate_grad_batches=self.config.accumulate_grad_batches,
        val_check_interval=self.config.val_check_interval,
        callbacks=callbacks,
        logger=self.logger,
        deterministic=self.config.deterministic,
        enable_progress_bar=self.config.progress_bar
    )
```

#### train(train_dataloader, val_dataloader)

Execute training.

```python
def train(
    self,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None
) -> None:
    """
    Execute model training
    
    Args:
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
    """
    if self.trainer is None:
        self.setup_trainer()
    
    self.trainer.fit(self.model, train_dataloader, val_dataloader)
```

#### test(test_dataloader)

Execute testing.

```python
def test(self, test_dataloader: DataLoader) -> Dict[str, float]:
    """
    Execute model testing
    
    Args:
        test_dataloader: Test data loader
        
    Returns:
        Test results dictionary
    """
    if self.trainer is None:
        self.setup_trainer()
    
    results = self.trainer.test(self.model, test_dataloader)
    return results[0] if results else {}
```

## RQVAE Trainer

### RQVAETrainer

Trainer specifically for training RQVAE models.

```python
class RQVAETrainer(BaseTrainer):
    """RQVAE trainer"""
    
    def __init__(
        self,
        model: RqVae,
        config: RQVAETrainingConfig,
        dataset: ItemDataset,
        logger: Optional[Logger] = None
    ):
        super().__init__(model, config, logger)
        self.dataset = dataset
        
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create data loaders
        
        Returns:
            (train_loader, val_loader, test_loader): Data loader tuple
        """
        # Split dataset
        train_size = int(0.8 * len(self.dataset))
        val_size = int(0.1 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config.random_seed)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
```

#### train_model()

Train RQVAE model.

```python
def train_model(self) -> RqVae:
    """
    Train RQVAE model
    
    Returns:
        Trained model
    """
    # Create data loaders
    train_loader, val_loader, test_loader = self.create_dataloaders()
    
    # Execute training
    self.train(train_loader, val_loader)
    
    # Test model
    if self.config.run_test:
        test_results = self.test(test_loader)
        print(f"Test results: {test_results}")
    
    return self.model
```

#### evaluate_reconstruction(test_dataloader)

Evaluate reconstruction quality.

```python
def evaluate_reconstruction(self, test_dataloader: DataLoader) -> Dict[str, float]:
    """
    Evaluate reconstruction quality
    
    Args:
        test_dataloader: Test data loader
        
    Returns:
        Evaluation metrics dictionary
    """
    self.model.eval()
    device = next(self.model.parameters()).device
    
    total_mse = 0
    total_cosine_sim = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            if isinstance(batch, dict):
                features = batch['features'].to(device)
            else:
                features = batch.to(device)
            
            # Forward pass
            reconstructed, _, _, _ = self.model(features)
            
            # Compute metrics
            mse = F.mse_loss(reconstructed, features, reduction='sum')
            cosine_sim = F.cosine_similarity(reconstructed, features, dim=1).sum()
            
            total_mse += mse.item()
            total_cosine_sim += cosine_sim.item()
            total_samples += features.size(0)
    
    return {
        'mse': total_mse / total_samples,
        'rmse': (total_mse / total_samples) ** 0.5,
        'cosine_similarity': total_cosine_sim / total_samples
    }
```

#### generate_semantic_ids(dataloader)

Generate semantic IDs for dataset.

```python
def generate_semantic_ids(self, dataloader: DataLoader) -> torch.Tensor:
    """
    Generate semantic IDs for dataset
    
    Args:
        dataloader: Data loader
        
    Returns:
        Semantic ID tensor (num_samples,)
    """
    self.model.eval()
    device = next(self.model.parameters()).device
    
    all_semantic_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                features = batch['features'].to(device)
            else:
                features = batch.to(device)
            
            semantic_ids = self.model.generate_semantic_ids(features)
            all_semantic_ids.append(semantic_ids.cpu())
    
    return torch.cat(all_semantic_ids, dim=0)
```

## TIGER Trainer

### TIGERTrainer

Trainer specifically for training TIGER models.

```python
class TIGERTrainer(BaseTrainer):
    """TIGER trainer"""
    
    def __init__(
        self,
        model: Tiger,
        config: TIGERTrainingConfig,
        dataset: SequenceDataset,
        logger: Optional[Logger] = None
    ):
        super().__init__(model, config, logger)
        self.dataset = dataset
        self.collate_fn = self._create_collate_fn()
        
    def _create_collate_fn(self) -> Callable:
        """Create data collation function"""
        def collate_fn(batch):
            # Extract sequence data
            input_ids = [item['input_ids'] for item in batch]
            labels = [item['labels'] for item in batch]
            
            # Pad sequences
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
            labels = pad_sequence(labels, batch_first=True, padding_value=-100)
            
            # Create attention mask
            attention_mask = (input_ids != 0).float()
            
            return {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask
            }
        
        return collate_fn
```

#### create_dataloaders()

Create TIGER data loaders.

```python
def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders
    
    Returns:
        (train_loader, val_loader, test_loader): Data loader tuple
    """
    # Split dataset
    train_size = int(0.8 * len(self.dataset))
    val_size = int(0.1 * len(self.dataset))
    test_size = len(self.dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        self.dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(self.config.random_seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=self.config.batch_size,
        shuffle=True,
        num_workers=self.config.num_workers,
        collate_fn=self.collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=self.config.batch_size,
        shuffle=False,
        num_workers=self.config.num_workers,
        collate_fn=self.collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=self.config.batch_size,
        shuffle=False,
        num_workers=self.config.num_workers,
        collate_fn=self.collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
```

#### evaluate_generation(test_dataloader, k_values)

Evaluate generation quality.

```python
def evaluate_generation(
    self,
    test_dataloader: DataLoader,
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Evaluate generation quality
    
    Args:
        test_dataloader: Test data loader
        k_values: Top-K value list
        
    Returns:
        Evaluation metrics dictionary
    """
    self.model.eval()
    device = next(self.model.parameters()).device
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Generate recommendations
            generated = self.model.generate(
                input_ids,
                max_length=self.config.max_generation_length,
                temperature=self.config.generation_temperature,
                top_k=self.config.generation_top_k
            )
            
            # Extract target sequences
            targets = []
            for label_seq in labels:
                target = label_seq[label_seq != -100].cpu().tolist()
                targets.append(target)
            
            all_predictions.extend(generated.cpu().tolist())
            all_targets.extend(targets)
    
    # Compute metrics
    metrics = {}
    for k in k_values:
        recall_k = self._compute_recall_at_k(all_predictions, all_targets, k)
        ndcg_k = self._compute_ndcg_at_k(all_predictions, all_targets, k)
        
        metrics[f'recall@{k}'] = recall_k
        metrics[f'ndcg@{k}'] = ndcg_k
    
    return metrics
```

#### _compute_recall_at_k(predictions, targets, k)

Compute Recall@K.

```python
def _compute_recall_at_k(
    self,
    predictions: List[List[int]],
    targets: List[List[int]],
    k: int
) -> float:
    """Compute Recall@K"""
    recall_scores = []
    
    for pred, target in zip(predictions, targets):
        if len(target) == 0:
            continue
            
        top_k_pred = set(pred[:k])
        target_set = set(target)
        
        recall = len(top_k_pred & target_set) / len(target_set)
        recall_scores.append(recall)
    
    return np.mean(recall_scores) if recall_scores else 0.0
```

#### _compute_ndcg_at_k(predictions, targets, k)

Compute NDCG@K.

```python
def _compute_ndcg_at_k(
    self,
    predictions: List[List[int]],
    targets: List[List[int]],
    k: int
) -> float:
    """Compute NDCG@K"""
    ndcg_scores = []
    
    for pred, target in zip(predictions, targets):
        if len(target) == 0:
            continue
        
        # Compute DCG
        dcg = 0
        for i, item in enumerate(pred[:k]):
            if item in target:
                dcg += 1 / np.log2(i + 2)
        
        # Compute IDCG
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(target), k)))
        
        # Compute NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0
```

## Training Configurations

### TrainingConfig

Base training configuration.

```python
@dataclass
class TrainingConfig:
    # Basic settings
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # Hardware settings
    gpus: int = 1 if torch.cuda.is_available() else 0
    precision: int = 32
    num_workers: int = 4
    
    # Training strategy
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    val_check_interval: float = 1.0
    
    # Checkpoints and logging
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    experiment_name: str = "experiment"
    version: Optional[str] = None
    save_top_k: int = 3
    
    # Early stopping
    early_stopping_patience: int = 10
    
    # Others
    deterministic: bool = True
    random_seed: int = 42
    progress_bar: bool = True
    run_test: bool = True
```

### RQVAETrainingConfig

RQVAE training configuration.

```python
@dataclass
class RQVAETrainingConfig(TrainingConfig):
    # Model parameters
    input_dim: int = 768
    hidden_dim: int = 512
    latent_dim: int = 256
    num_embeddings: int = 1024
    commitment_cost: float = 0.25
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    
    # Dataset parameters
    dataset_name: str = "p5_amazon"
    dataset_split: str = "beauty"
    
    # Evaluation parameters
    eval_reconstruction: bool = True
    eval_quantization: bool = True
```

### TIGERTrainingConfig

TIGER training configuration.

```python
@dataclass
class TIGERTrainingConfig(TrainingConfig):
    # Model parameters
    vocab_size: int = 1024
    embedding_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    attn_dim: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 100
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 16
    max_epochs: int = 50
    
    # Generation parameters
    max_generation_length: int = 50
    generation_temperature: float = 1.0
    generation_top_k: int = 50
    generation_top_p: float = 0.9
    
    # Dataset parameters
    dataset_name: str = "p5_amazon"
    dataset_split: str = "beauty"
    pretrained_rqvae_path: str = "checkpoints/rqvae.ckpt"
    
    # Evaluation parameters
    eval_generation: bool = True
    eval_k_values: List[int] = field(default_factory=lambda: [5, 10, 20])
```

## Training Scripts

### Train RQVAE

```python
#!/usr/bin/env python3
"""Script for training RQVAE model"""

import argparse
from pathlib import Path

from genrec.models.rqvae import RqVae
from genrec.data.dataset_factory import DatasetFactory
from genrec.trainers import RQVAETrainer, RQVAETrainingConfig

def main():
    parser = argparse.ArgumentParser(description="Train RQVAE model")
    parser.add_argument("--dataset", default="p5_amazon", help="Dataset name")
    parser.add_argument("--split", default="beauty", help="Dataset split")
    parser.add_argument("--root", default="dataset", help="Dataset root directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Checkpoint directory")
    
    args = parser.parse_args()
    
    # Create configuration
    config = RQVAETrainingConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        dataset_name=args.dataset,
        dataset_split=args.split
    )
    
    # Create dataset
    dataset = DatasetFactory.create_item_dataset(
        args.dataset,
        root=args.root,
        split=args.split,
        train_test_split="all"
    )
    
    # Create model
    model = RqVae(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_embeddings=config.num_embeddings,
        commitment_cost=config.commitment_cost,
        learning_rate=config.learning_rate
    )
    
    # Create trainer
    trainer = RQVAETrainer(model, config, dataset)
    
    # Train model
    trained_model = trainer.train_model()
    
    print(f"Training completed. Model saved to {config.checkpoint_dir}")

if __name__ == "__main__":
    main()
```

### Train TIGER

```python
#!/usr/bin/env python3
"""Script for training TIGER model"""

import argparse
from pathlib import Path

from genrec.models.tiger import Tiger
from genrec.data.dataset_factory import DatasetFactory
from genrec.trainers import TIGERTrainer, TIGERTrainingConfig

def main():
    parser = argparse.ArgumentParser(description="Train TIGER model")
    parser.add_argument("--dataset", default="p5_amazon", help="Dataset name")
    parser.add_argument("--split", default="beauty", help="Dataset split")
    parser.add_argument("--root", default="dataset", help="Dataset root directory")
    parser.add_argument("--rqvae_path", required=True, help="Pretrained RQVAE checkpoint path")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Checkpoint directory")
    
    args = parser.parse_args()
    
    # Create configuration
    config = TIGERTrainingConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        dataset_name=args.dataset,
        dataset_split=args.split,
        pretrained_rqvae_path=args.rqvae_path
    )
    
    # Create dataset
    dataset = DatasetFactory.create_sequence_dataset(
        args.dataset,
        root=args.root,
        split=args.split,
        train_test_split="train",
        pretrained_rqvae_path=args.rqvae_path
    )
    
    # Create model
    model = Tiger(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        learning_rate=config.learning_rate
    )
    
    # Create trainer
    trainer = TIGERTrainer(model, config, dataset)
    
    # Train model
    trained_model = trainer.train_model()
    
    print(f"Training completed. Model saved to {config.checkpoint_dir}")

if __name__ == "__main__":
    main()
```

## Usage Examples

### Basic Training

```python
from genrec.trainers import RQVAETrainer, RQVAETrainingConfig
from genrec.models.rqvae import RqVae
from genrec.data.dataset_factory import DatasetFactory

# Create configuration
config = RQVAETrainingConfig(
    batch_size=64,
    max_epochs=100,
    learning_rate=1e-3
)

# Create dataset
dataset = DatasetFactory.create_item_dataset(
    "p5_amazon",
    root="dataset/amazon",
    split="beauty"
)

# Create model
model = RqVae(
    input_dim=768,
    hidden_dim=512,
    num_embeddings=1024
)

# Create trainer and train
trainer = RQVAETrainer(model, config, dataset)
trained_model = trainer.train_model()
```