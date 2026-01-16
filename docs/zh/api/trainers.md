# 训练器 API 参考

训练工具和脚本的详细文档。

## 基础训练器

### BaseTrainer

所有训练器的基础类。

```python
class BaseTrainer:
    """基础训练器类"""
    
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
        """创建默认日志记录器"""
        from pytorch_lightning.loggers import TensorBoardLogger
        
        return TensorBoardLogger(
            save_dir=self.config.log_dir,
            name=self.config.experiment_name,
            version=self.config.version
        )
```

**方法:**

#### setup_trainer()

设置 PyTorch Lightning 训练器。

```python
def setup_trainer(self) -> None:
    """设置训练器"""
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    
    # 回调函数
    callbacks = []
    
    # 检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=self.config.checkpoint_dir,
        filename='{epoch}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=self.config.save_top_k,
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # 早停回调
    if self.config.early_stopping_patience > 0:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            mode='min'
        )
        callbacks.append(early_stop_callback)
    
    # 创建训练器
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

执行训练。

```python
def train(
    self,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None
) -> None:
    """
    执行模型训练
    
    Args:
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
    """
    if self.trainer is None:
        self.setup_trainer()
    
    self.trainer.fit(self.model, train_dataloader, val_dataloader)
```

#### test(test_dataloader)

执行测试。

```python
def test(self, test_dataloader: DataLoader) -> Dict[str, float]:
    """
    执行模型测试
    
    Args:
        test_dataloader: 测试数据加载器
        
    Returns:
        测试结果字典
    """
    if self.trainer is None:
        self.setup_trainer()
    
    results = self.trainer.test(self.model, test_dataloader)
    return results[0] if results else {}
```

## RQVAE 训练器

### RQVAETrainer

专门用于训练 RQVAE 模型的训练器。

```python
class RQVAETrainer(BaseTrainer):
    """RQVAE 训练器"""
    
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
        创建数据加载器
        
        Returns:
            (train_loader, val_loader, test_loader): 数据加载器元组
        """
        # 分割数据集
        train_size = int(0.8 * len(self.dataset))
        val_size = int(0.1 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config.random_seed)
        )
        
        # 创建数据加载器
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

训练 RQVAE 模型。

```python
def train_model(self) -> RqVae:
    """
    训练 RQVAE 模型
    
    Returns:
        训练好的模型
    """
    # 创建数据加载器
    train_loader, val_loader, test_loader = self.create_dataloaders()
    
    # 执行训练
    self.train(train_loader, val_loader)
    
    # 测试模型
    if self.config.run_test:
        test_results = self.test(test_loader)
        print(f"Test results: {test_results}")
    
    return self.model
```

#### evaluate_reconstruction(test_dataloader)

评估重构质量。

```python
def evaluate_reconstruction(self, test_dataloader: DataLoader) -> Dict[str, float]:
    """
    评估重构质量
    
    Args:
        test_dataloader: 测试数据加载器
        
    Returns:
        评估指标字典
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
            
            # 前向传播
            reconstructed, _, _, _ = self.model(features)
            
            # 计算指标
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

为数据集生成语义 ID。

```python
def generate_semantic_ids(self, dataloader: DataLoader) -> torch.Tensor:
    """
    为数据集生成语义 ID
    
    Args:
        dataloader: 数据加载器
        
    Returns:
        语义 ID 张量 (num_samples,)
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

## TIGER 训练器

### TIGERTrainer

专门用于训练 TIGER 模型的训练器。

```python
class TIGERTrainer(BaseTrainer):
    """TIGER 训练器"""
    
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
        """创建数据整理函数"""
        def collate_fn(batch):
            # 提取序列数据
            input_ids = [item['input_ids'] for item in batch]
            labels = [item['labels'] for item in batch]
            
            # 填充序列
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
            labels = pad_sequence(labels, batch_first=True, padding_value=-100)
            
            # 创建注意力掩码
            attention_mask = (input_ids != 0).float()
            
            return {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask
            }
        
        return collate_fn
```

#### create_dataloaders()

创建 TIGER 数据加载器。

```python
def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Returns:
        (train_loader, val_loader, test_loader): 数据加载器元组
    """
    # 分割数据集
    train_size = int(0.8 * len(self.dataset))
    val_size = int(0.1 * len(self.dataset))
    test_size = len(self.dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        self.dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(self.config.random_seed)
    )
    
    # 创建数据加载器
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

评估生成质量。

```python
def evaluate_generation(
    self,
    test_dataloader: DataLoader,
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    评估生成质量
    
    Args:
        test_dataloader: 测试数据加载器
        k_values: Top-K 值列表
        
    Returns:
        评估指标字典
    """
    self.model.eval()
    device = next(self.model.parameters()).device
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # 生成推荐
            generated = self.model.generate(
                input_ids,
                max_length=self.config.max_generation_length,
                temperature=self.config.generation_temperature,
                top_k=self.config.generation_top_k
            )
            
            # 提取目标序列
            targets = []
            for label_seq in labels:
                target = label_seq[label_seq != -100].cpu().tolist()
                targets.append(target)
            
            all_predictions.extend(generated.cpu().tolist())
            all_targets.extend(targets)
    
    # 计算指标
    metrics = {}
    for k in k_values:
        recall_k = self._compute_recall_at_k(all_predictions, all_targets, k)
        ndcg_k = self._compute_ndcg_at_k(all_predictions, all_targets, k)
        
        metrics[f'recall@{k}'] = recall_k
        metrics[f'ndcg@{k}'] = ndcg_k
    
    return metrics
```

#### _compute_recall_at_k(predictions, targets, k)

计算 Recall@K。

```python
def _compute_recall_at_k(
    self,
    predictions: List[List[int]],
    targets: List[List[int]],
    k: int
) -> float:
    """计算 Recall@K"""
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

计算 NDCG@K。

```python
def _compute_ndcg_at_k(
    self,
    predictions: List[List[int]],
    targets: List[List[int]],
    k: int
) -> float:
    """计算 NDCG@K"""
    ndcg_scores = []
    
    for pred, target in zip(predictions, targets):
        if len(target) == 0:
            continue
        
        # 计算 DCG
        dcg = 0
        for i, item in enumerate(pred[:k]):
            if item in target:
                dcg += 1 / np.log2(i + 2)
        
        # 计算 IDCG
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(target), k)))
        
        # 计算 NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0
```

## 训练配置

### TrainingConfig

基础训练配置。

```python
@dataclass
class TrainingConfig:
    # 基础设置
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # 硬件设置
    gpus: int = 1 if torch.cuda.is_available() else 0
    precision: int = 32
    num_workers: int = 4
    
    # 训练策略
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    val_check_interval: float = 1.0
    
    # 检查点和日志
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    experiment_name: str = "experiment"
    version: Optional[str] = None
    save_top_k: int = 3
    
    # 早停
    early_stopping_patience: int = 10
    
    # 其他
    deterministic: bool = True
    random_seed: int = 42
    progress_bar: bool = True
    run_test: bool = True
```

### RQVAETrainingConfig

RQVAE 训练配置。

```python
@dataclass
class RQVAETrainingConfig(TrainingConfig):
    # 模型参数
    input_dim: int = 768
    hidden_dim: int = 512
    latent_dim: int = 256
    num_embeddings: int = 1024
    commitment_cost: float = 0.25
    
    # 训练参数
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    
    # 数据集参数
    dataset_name: str = "p5_amazon"
    dataset_split: str = "beauty"
    
    # 评估参数
    eval_reconstruction: bool = True
    eval_quantization: bool = True
```

### TIGERTrainingConfig

TIGER 训练配置。

```python
@dataclass
class TIGERTrainingConfig(TrainingConfig):
    # 模型参数
    vocab_size: int = 1024
    embedding_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    attn_dim: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 100
    
    # 训练参数
    learning_rate: float = 1e-4
    batch_size: int = 16
    max_epochs: int = 50
    
    # 生成参数
    max_generation_length: int = 50
    generation_temperature: float = 1.0
    generation_top_k: int = 50
    generation_top_p: float = 0.9
    
    # 数据集参数
    dataset_name: str = "p5_amazon"
    dataset_split: str = "beauty"
    pretrained_rqvae_path: str = "checkpoints/rqvae.ckpt"
    
    # 评估参数
    eval_generation: bool = True
    eval_k_values: List[int] = field(default_factory=lambda: [5, 10, 20])
```

## 训练脚本

### 训练 RQVAE

```python
#!/usr/bin/env python3
"""训练 RQVAE 模型的脚本"""

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
    
    # 创建配置
    config = RQVAETrainingConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        dataset_name=args.dataset,
        dataset_split=args.split
    )
    
    # 创建数据集
    dataset = DatasetFactory.create_item_dataset(
        args.dataset,
        root=args.root,
        split=args.split,
        train_test_split="all"
    )
    
    # 创建模型
    model = RqVae(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_embeddings=config.num_embeddings,
        commitment_cost=config.commitment_cost,
        learning_rate=config.learning_rate
    )
    
    # 创建训练器
    trainer = RQVAETrainer(model, config, dataset)
    
    # 训练模型
    trained_model = trainer.train_model()
    
    print(f"Training completed. Model saved to {config.checkpoint_dir}")

if __name__ == "__main__":
    main()
```

### 训练 TIGER

```python
#!/usr/bin/env python3
"""训练 TIGER 模型的脚本"""

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
    
    # 创建配置
    config = TIGERTrainingConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        dataset_name=args.dataset,
        dataset_split=args.split,
        pretrained_rqvae_path=args.rqvae_path
    )
    
    # 创建数据集
    dataset = DatasetFactory.create_sequence_dataset(
        args.dataset,
        root=args.root,
        split=args.split,
        train_test_split="train",
        pretrained_rqvae_path=args.rqvae_path
    )
    
    # 创建模型
    model = Tiger(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        learning_rate=config.learning_rate
    )
    
    # 创建训练器
    trainer = TIGERTrainer(model, config, dataset)
    
    # 训练模型
    trained_model = trainer.train_model()
    
    print(f"Training completed. Model saved to {config.checkpoint_dir}")

if __name__ == "__main__":
    main()
```

## 使用示例

### 基本训练

```python
from genrec.trainers import RQVAETrainer, RQVAETrainingConfig
from genrec.models.rqvae import RqVae
from genrec.data.dataset_factory import DatasetFactory

# 创建配置
config = RQVAETrainingConfig(
    batch_size=64,
    max_epochs=100,
    learning_rate=1e-3
)

# 创建数据集
dataset = DatasetFactory.create_item_dataset(
    "p5_amazon",
    root="dataset/amazon",
    split="beauty"
)

# 创建模型
model = RqVae(
    input_dim=768,
    hidden_dim=512,
    num_embeddings=1024
)

# 创建训练器并训练
trainer = RQVAETrainer(model, config, dataset)
trained_model = trainer.train_model()
```