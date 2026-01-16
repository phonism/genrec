# TIGER API Reference

Detailed API documentation for the Transformer-based generative retrieval model (TIGER).

## Core Classes

### Tiger

Main TIGER model class.

```python
class Tiger(LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        attn_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 1024,
        learning_rate: float = 1e-4
    )
```

**Parameters:**
- `vocab_size`: Vocabulary size
- `embedding_dim`: Embedding dimension
- `num_heads`: Number of attention heads
- `num_layers`: Number of Transformer layers
- `attn_dim`: Attention dimension
- `dropout`: Dropout probability
- `max_seq_length`: Maximum sequence length
- `learning_rate`: Learning rate

**Methods:**

#### forward(input_ids, attention_mask=None)

Forward pass computation.

```python
def forward(
    self, 
    input_ids: torch.Tensor, 
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Args:
        input_ids: Input sequence (batch_size, seq_len)
        attention_mask: Attention mask (batch_size, seq_len)
    
    Returns:
        logits: Output logits (batch_size, seq_len, vocab_size)
    """
```

#### generate(input_ids, max_length=50, temperature=1.0, top_k=None, top_p=None)

Generate recommendation sequences.

```python
def generate(
    self,
    input_ids: torch.Tensor,
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> torch.Tensor:
    """
    Args:
        input_ids: Input sequence
        max_length: Maximum generation length
        temperature: Temperature parameter
        top_k: Top-k sampling
        top_p: Top-p sampling
    
    Returns:
        generated: Generated sequence
    """
```

#### generate_with_trie(input_ids, trie, max_length=50)

Generate with Trie constraints.

```python
def generate_with_trie(
    self,
    input_ids: torch.Tensor,
    trie: TrieNode,
    max_length: int = 50
) -> torch.Tensor:
    """
    Args:
        input_ids: Input sequence
        trie: Trie constraint structure
        max_length: Maximum generation length
    
    Returns:
        generated: Constrained generated sequence
    """
```

## Component Classes

### TransformerBlock

Transformer block implementation.

```python
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        attn_dim: int,
        dropout: float = 0.1
    )
```

### MultiHeadAttention

Multi-head attention mechanism.

```python
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1
    )
```

### PositionalEncoding

Positional encoding.

```python
class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_seq_length: int = 5000
    )
```

## Data Structures

### TrieNode

Trie node for constrained generation.

```python
class TrieNode(defaultdict):
    def __init__(self):
        super().__init__(TrieNode)
        self.is_end = False
        
    def add_sequence(self, sequence: List[int]):
        """Add sequence to Trie"""
        node = self
        for token in sequence:
            node = node[token]
        node.is_end = True
        
    def get_valid_tokens(self) -> List[int]:
        """Get valid tokens at current node"""
        return list(self.keys())
```

### Build Trie

```python
def build_trie(valid_sequences: List[List[int]]) -> TrieNode:
    """Build Trie of valid sequences"""
    root = TrieNode()
    for sequence in valid_sequences:
        root.add_sequence(sequence)
    return root
```

## Training Interface

### Training Step

```python
def training_step(self, batch, batch_idx):
    """Training step"""
    input_ids = batch['input_ids']
    labels = batch['labels']
    attention_mask = batch.get('attention_mask', None)
    
    # Forward pass
    logits = self(input_ids, attention_mask)
    
    # Compute loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    # Log metrics
    self.log('train_loss', loss)
    
    return loss
```

### Validation Step

```python
def validation_step(self, batch, batch_idx):
    """Validation step"""
    input_ids = batch['input_ids']
    labels = batch['labels']
    attention_mask = batch.get('attention_mask', None)
    
    # Forward pass
    logits = self(input_ids, attention_mask)
    
    # Compute loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    # Log metrics
    self.log('val_loss', loss)
    
    return loss
```

## Inference Interface

### Batch Generation

```python
def batch_generate(
    model: Tiger,
    input_sequences: List[torch.Tensor],
    max_length: int = 50,
    device: str = 'cuda'
) -> List[torch.Tensor]:
    """Batch generation for recommendations"""
    model.eval()
    model.to(device)
    
    results = []
    
    with torch.no_grad():
        for input_seq in input_sequences:
            input_seq = input_seq.to(device)
            generated = model.generate(input_seq, max_length=max_length)
            results.append(generated.cpu())
    
    return results
```

### Constrained Generation

```python
def constrained_generate(
    model: Tiger,
    input_ids: torch.Tensor,
    valid_item_sequences: List[List[int]],
    max_length: int = 50
) -> torch.Tensor:
    """Constrained generation for recommendations"""
    # Build Trie
    trie = build_trie(valid_item_sequences)
    
    # Constrained generation
    return model.generate_with_trie(input_ids, trie, max_length)
```

## Evaluation Interface

### Top-K Recommendation Evaluation

```python
def evaluate_recommendation(
    model: Tiger,
    test_dataloader: DataLoader,
    k_values: List[int] = [5, 10, 20],
    device: str = 'cuda'
) -> Dict[str, float]:
    """Evaluate recommendation performance"""
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets']
            
            # Generate recommendations
            generated = model.generate(input_ids, max_length=50)
            
            all_predictions.extend(generated.cpu().tolist())
            all_targets.extend(targets.tolist())
    
    # Compute metrics
    metrics = {}
    for k in k_values:
        recall_k = compute_recall_at_k(all_predictions, all_targets, k)
        ndcg_k = compute_ndcg_at_k(all_predictions, all_targets, k)
        
        metrics[f'recall@{k}'] = recall_k
        metrics[f'ndcg@{k}'] = ndcg_k
    
    return metrics
```

### Perplexity Evaluation

```python
def evaluate_perplexity(
    model: Tiger,
    test_dataloader: DataLoader,
    device: str = 'cuda'
) -> float:
    """Evaluate perplexity"""
    model.eval()
    model.to(device)
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch.get('attention_mask', None)
            
            logits = model(input_ids, attention_mask)
            
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Count valid tokens
            valid_tokens = (shift_labels != -100).sum()
            
            total_loss += loss.item()
            total_tokens += valid_tokens.item()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity
```

## Utility Functions

### Sequence Processing

```python
def pad_sequences(
    sequences: List[torch.Tensor],
    pad_token_id: int = 0,
    max_length: Optional[int] = None
) -> torch.Tensor:
    """Pad sequences to same length"""
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded = []
    for seq in sequences:
        if len(seq) < max_length:
            pad_length = max_length - len(seq)
            padded_seq = torch.cat([
                seq, 
                torch.full((pad_length,), pad_token_id, dtype=seq.dtype)
            ])
        else:
            padded_seq = seq[:max_length]
        padded.append(padded_seq)
    
    return torch.stack(padded)
```

### Sampling Strategies

```python
def top_k_top_p_sampling(
    logits: torch.Tensor,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: float = 1.0
) -> torch.Tensor:
    """Top-k and top-p sampling"""
    logits = logits / temperature
    
    # Top-k sampling
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k)
        logits[logits < values[..., [-1]]] = float('-inf')
    
    # Top-p sampling
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Find positions where cumulative probability exceeds top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
    
    # Sampling
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, 1)
    
    return next_token
```

## Usage Examples

### Basic Training

```python
from genrec.models.tiger import Tiger
from genrec.data.p5_amazon import P5AmazonSequenceDataset
import pytorch_lightning as pl

# Create dataset
dataset = P5AmazonSequenceDataset(
    root="dataset/amazon",
    split="beauty",
    train_test_split="train",
    pretrained_rqvae_path="checkpoints/rqvae.ckpt"
)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Create model
model = Tiger(
    vocab_size=1024,
    embedding_dim=512,
    num_heads=8,
    num_layers=6,
    learning_rate=1e-4
)

# Train model
trainer = pl.Trainer(max_epochs=50, gpus=1)
trainer.fit(model, dataloader)
```

### Recommendation Generation

```python
# Load trained model
model = Tiger.load_from_checkpoint("checkpoints/tiger.ckpt")
model.eval()

# User history sequence
user_sequence = torch.tensor([10, 25, 67, 89])  # Semantic ID sequence

# Generate recommendations
with torch.no_grad():
    recommendations = model.generate(
        user_sequence.unsqueeze(0),
        max_length=20,
        temperature=0.8,
        top_k=50
    )
    
print(f"Recommendations: {recommendations.squeeze().tolist()}")
```