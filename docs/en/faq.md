# Frequently Asked Questions (FAQ)

This page contains common questions and answers when using the genrec framework.

## Installation and Environment

### Q: How do I install genrec?

A: Currently supports installation from source:

```bash
git clone https://github.com/phonism/genrec.git
cd genrec
pip install -e .
```

### Q: Which Python versions are supported?

A: We recommend Python 3.8 or higher. The framework has been tested on Python 3.8, 3.9, and 3.10.

### Q: What are the main dependencies?

A: Main dependencies include:
- PyTorch >= 1.11.0
- PyTorch Lightning >= 1.6.0
- sentence-transformers >= 2.2.0
- pandas >= 1.3.0
- numpy >= 1.21.0

### Q: Does it support GPU training?

A: Yes, the framework fully supports GPU training. Make sure you have the correct PyTorch CUDA version installed.

## Data and Datasets

### Q: What dataset formats are supported?

A: The framework supports:
- JSON format recommendation datasets
- CSV format user-item interaction data
- Parquet format large-scale datasets
- Custom formats (by inheriting base classes)

### Q: How do I add a custom dataset?

A: Inherit from the `BaseRecommenderDataset` class and implement necessary methods:

```python
from genrec.data.base_dataset import BaseRecommenderDataset

class MyDataset(BaseRecommenderDataset):
    def download(self):
        # Implement data download logic
        pass
        
    def load_raw_data(self):
        # Implement data loading logic
        return {"items": items_df, "interactions": interactions_df}
        
    def preprocess_data(self, raw_data):
        # Implement data preprocessing logic
        return processed_data
```

### Q: How large is the P5 Amazon dataset?

A: Different categories have different sizes:
- Beauty: ~500MB
- Electronics: ~2GB
- Sports: ~1GB
- Full dataset may require 10GB+ storage

### Q: How are missing item features handled?

A: The framework automatically handles missing features:
- Text fields are filled with "Unknown"
- Numerical fields are filled with mean or 0
- Filling strategies can be customized in configuration

## Model Training

### Q: How long does RQVAE training take?

A: Depends on dataset size and hardware:
- Small datasets (<100k items): 30 minutes - 2 hours
- Medium datasets (100k-1M items): 2-8 hours
- Large datasets (>1M items): 8-24 hours

### Q: What are TIGER training memory requirements?

A: Typical memory usage:
- Minimum: 8GB GPU memory (small batch size)
- Recommended: 16GB GPU memory (medium batch size)
- Large-scale: 32GB+ GPU memory (large batch size)

### Q: How do I choose appropriate embedding dimensions?

A: Rule of thumb:
- Small datasets: 256-512 dimensions
- Medium datasets: 512-768 dimensions
- Large datasets: 768-1024 dimensions
- Specific choice should be based on validation set performance

### Q: What to do when encountering CUDA out of memory during training?

A: Solutions:
1. Reduce batch size
2. Use gradient accumulation
3. Enable mixed precision training
4. Reduce model size

```python
# Reduce batch size
config.batch_size = 16

# Gradient accumulation
config.accumulate_grad_batches = 4

# Mixed precision
config.precision = 16
```

## Model Usage

### Q: How do I generate recommendations?

A: Basic recommendation generation:

```python
# Load models
rqvae = RqVae.load_from_checkpoint("rqvae.ckpt")
tiger = Tiger.load_from_checkpoint("tiger.ckpt")

# User history (item IDs)
user_history = [1, 5, 23, 67]

# Convert to semantic IDs
semantic_ids = rqvae.encode_items(user_history)

# Generate recommendations
recommendations = tiger.generate(semantic_ids, max_length=10)
```

### Q: How do I handle cold start users?

A: For new users:
1. Use popular item recommendations
2. Content-based recommendations using user profile
3. Similarity-based recommendations using item features

```python
def recommend_for_new_user(user_profile, k=10):
    # Find similar items based on user profile
    similar_items = find_similar_items_by_profile(user_profile)
    return similar_items[:k]
```

### Q: How do I ensure recommendation diversity?

A: Methods to improve diversity:
1. Use top-p sampling instead of greedy sampling
2. Post-processing for deduplication and diversification
3. Add diversity loss during training

```python
# Use sampling for generation
recommendations = tiger.generate(
    input_seq, 
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)
```

## Performance and Optimization

### Q: How do I improve inference speed?

A: Optimization methods:
1. Model quantization
2. ONNX export
3. TensorRT optimization
4. Batch inference

```python
# Model quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Batch inference
def batch_recommend(user_histories, batch_size=32):
    results = []
    for i in range(0, len(user_histories), batch_size):
        batch = user_histories[i:i+batch_size]
        batch_results = model.batch_generate(batch)
        results.extend(batch_results)
    return results
```

### Q: Can model size be compressed?

A: Compression techniques:
- Quantization: 50-75% size reduction
- Pruning: Remove unimportant parameters
- Knowledge distillation: Train small model to mimic large model

### Q: How do I monitor model performance?

A: Monitor metrics:
- Inference latency
- Memory usage
- GPU utilization
- Recommendation quality metrics (Recall, NDCG)

## Errors and Debugging

### Q: Getting "RuntimeError: CUDA out of memory" during training?

A: Solution steps:
1. Reduce batch_size
2. Enable gradient checkpointing
3. Clear GPU cache

```python
import torch
torch.cuda.empty_cache()

# Or in training configuration
config.gradient_checkpointing = True
```

### Q: What to do when model loading fails?

A: Check items:
1. Check if checkpoint file is complete
2. PyTorch version compatibility
3. Model architecture match

```python
try:
    model = Model.load_from_checkpoint(checkpoint_path)
except Exception as e:
    print(f"Loading failed: {e}")
    # Try loading state dict
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
```

### Q: What to do when training loss doesn't converge?

A: Debugging methods:
1. Check learning rate (may be too large or too small)
2. Check data preprocessing
3. Increase training data amount
4. Adjust model architecture

```python
# Learning rate scheduling
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=5, factor=0.5
)
```

## Deployment and Production

### Q: How do I deploy to production?

A: Deployment options:
1. REST API service (FastAPI/Flask)
2. Docker containerization
3. Kubernetes cluster
4. Cloud service platforms

### Q: Does it support real-time recommendations?

A: Yes, the framework supports:
- Online inference API
- Batch pre-computation
- Streaming processing integration

### Q: How do I conduct A/B testing?

A: A/B testing framework:

```python
class ABTestFramework:
    def get_variant(self, user_id, experiment_name):
        # Consistent hash-based user grouping
        hash_value = hash(f"{user_id}_{experiment_name}")
        return "A" if hash_value % 2 == 0 else "B"
    
    def recommend_with_test(self, user_id, user_history):
        variant = self.get_variant(user_id, "model_test")
        if variant == "A":
            return model_a.recommend(user_history)
        else:
            return model_b.recommend(user_history)
```

## Advanced Usage

### Q: How do I implement multi-task learning?

A: Extend model to support multiple objectives:

```python
class MultiTaskTiger(Tiger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rating_head = nn.Linear(self.embedding_dim, 1)
        self.category_head = nn.Linear(self.embedding_dim, num_categories)
    
    def forward(self, x):
        hidden = super().forward(x)
        
        # Multiple output heads
        recommendations = self.recommendation_head(hidden)
        ratings = self.rating_head(hidden)
        categories = self.category_head(hidden)
        
        return recommendations, ratings, categories
```

### Q: How do I integrate external features?

A: Feature fusion methods:

```python
class FeatureEnhancedModel(Tiger):
    def __init__(self, user_feature_dim, item_feature_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_feature_proj = nn.Linear(user_feature_dim, self.embedding_dim)
        self.item_feature_proj = nn.Linear(item_feature_dim, self.embedding_dim)
    
    def forward(self, item_seq, user_features=None, item_features=None):
        seq_emb = super().forward(item_seq)
        
        if user_features is not None:
            user_emb = self.user_feature_proj(user_features)
            seq_emb = seq_emb + user_emb.unsqueeze(1)
        
        return seq_emb
```

### Q: How do I handle temporal information in sequences?

A: Time-aware recommendations:

```python
class TimeAwareTiger(Tiger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_emb = nn.Embedding(24 * 7, self.embedding_dim)  # hours*days
    
    def forward(self, item_seq, time_seq=None):
        seq_emb = self.item_embedding(item_seq)
        
        if time_seq is not None:
            time_emb = self.time_emb(time_seq)
            seq_emb = seq_emb + time_emb
        
        return self.transformer(seq_emb)
```

If you have other questions, please check the [API documentation](api/index.md) or submit an issue on GitHub.