# Processors API Reference

Detailed documentation for text and sequence processing utilities.

## Text Processor

### TextProcessor

Core class for text encoding and processing.

```python
class TextProcessor:
    def __init__(self, config: TextEncodingConfig):
        self.config = config
        self.model = None
        self.device = config.device
        self.cache_manager = CacheManager(config.cache_dir)
```

**Parameters:**
- `config`: Text encoding configuration object

**Methods:**

#### load_model()

Load text encoding model.

```python
def load_model(self) -> None:
    """
    Load Sentence Transformer model
    """
    if self.model is None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.config.encoder_model)
        self.model.to(self.device)
        print(f"Loaded text encoder: {self.config.encoder_model}")
```

#### encode_texts(texts, cache_key, force_reload)

Encode text list.

```python
def encode_texts(
    self,
    texts: List[str],
    cache_key: Optional[str] = None,
    force_reload: bool = False
) -> np.ndarray:
    """
    Encode text list to embedding vectors
    
    Args:
        texts: Text list
        cache_key: Cache key, will attempt to use cache if provided
        force_reload: Whether to force recomputation
        
    Returns:
        Embedding matrix (num_texts, embedding_dim)
    """
    # Check cache
    if cache_key and not force_reload and self.cache_manager.exists(cache_key):
        print(f"Loading embeddings from cache: {cache_key}")
        return self.cache_manager.load(cache_key)
    
    # Load model
    self.load_model()
    
    # Batch encoding
    print(f"Encoding {len(texts)} texts with {self.config.encoder_model}")
    embeddings = []
    
    for i in range(0, len(texts), self.config.batch_size):
        batch_texts = texts[i:i + self.config.batch_size]
        batch_embeddings = self.model.encode(
            batch_texts,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=True
        )
        embeddings.append(batch_embeddings)
    
    # Merge results
    embeddings = np.vstack(embeddings)
    
    # Save cache
    if cache_key:
        self.cache_manager.save(cache_key, embeddings)
        print(f"Saved embeddings to cache: {cache_key}")
    
    return embeddings
```

#### encode_item_features(items_df, cache_key, force_reload)

Encode item features.

```python
def encode_item_features(
    self,
    items_df: pd.DataFrame,
    cache_key: Optional[str] = None,
    force_reload: bool = False
) -> np.ndarray:
    """
    Encode item features to embedding vectors
    
    Args:
        items_df: Items DataFrame
        cache_key: Cache key
        force_reload: Whether to force recomputation
        
    Returns:
        Item embedding matrix (num_items, embedding_dim)
    """
    # Format text
    texts = []
    for _, row in items_df.iterrows():
        text = self.config.format_text(row.to_dict())
        texts.append(text)
    
    return self.encode_texts(texts, cache_key, force_reload)
```

#### encode_single_text(text)

Encode single text.

```python
def encode_single_text(self, text: str) -> np.ndarray:
    """
    Encode single text
    
    Args:
        text: Input text
        
    Returns:
        Text embedding vector (embedding_dim,)
    """
    self.load_model()
    
    embedding = self.model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=self.config.normalize_embeddings
    )[0]
    
    return embedding
```

#### compute_similarity(text1, text2)

Compute text similarity.

```python
def compute_similarity(self, text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two texts
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Cosine similarity value [-1, 1]
    """
    embedding1 = self.encode_single_text(text1)
    embedding2 = self.encode_single_text(text2)
    
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
```

#### find_similar_texts(query_text, candidate_texts, top_k)

Find similar texts.

```python
def find_similar_texts(
    self,
    query_text: str,
    candidate_texts: List[str],
    top_k: int = 5
) -> List[Tuple[int, str, float]]:
    """
    Find texts most similar to query text
    
    Args:
        query_text: Query text
        candidate_texts: Candidate text list
        top_k: Return top k most similar
        
    Returns:
        List of (index, text, similarity) tuples, sorted by similarity descending
    """
    query_embedding = self.encode_single_text(query_text)
    candidate_embeddings = self.encode_texts(candidate_texts)
    
    # Compute similarities
    similarities = np.dot(candidate_embeddings, query_embedding)
    
    # Get top-k
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append((idx, candidate_texts[idx], similarities[idx]))
    
    return results
```

## Sequence Processor

### SequenceProcessor

Core class for sequence data processing.

```python
class SequenceProcessor:
    def __init__(self, config: SequenceConfig):
        self.config = config
        
    def build_user_sequences(
        self, 
        interactions_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Build user interaction sequences
        
        Args:
            interactions_df: Interactions DataFrame containing user_id, item_id, timestamp
            
        Returns:
            User sequence list, each sequence contains user ID and item sequence
        """
        sequences = []
        
        # Group by user and sort by time
        for user_id, group in interactions_df.groupby('user_id'):
            user_interactions = group.sort_values('timestamp')
            item_sequence = user_interactions['item_id'].tolist()
            
            # Filter sequences that are too short
            if len(item_sequence) >= self.config.min_seq_length:
                sequences.append({
                    'user_id': user_id,
                    'item_sequence': item_sequence,
                    'timestamps': user_interactions['timestamp'].tolist() if self.config.include_timestamps else None
                })
        
        return sequences
```

#### create_training_samples(sequences)

Create training samples.

```python
def create_training_samples(
    self,
    sequences: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Create training samples from user sequences
    
    Args:
        sequences: User sequence list
        
    Returns:
        Training sample list, each sample contains input and target sequences
    """
    training_samples = []
    
    for seq_data in sequences:
        item_sequence = seq_data['item_sequence']
        
        # Create multiple subsequences
        for i in range(0, len(item_sequence) - self.config.min_seq_length + 1, self.config.sequence_stride):
            # Determine subsequence length
            end_idx = min(i + self.config.max_seq_length, len(item_sequence))
            
            if end_idx - i >= self.config.min_seq_length:
                input_seq = item_sequence[i:end_idx-self.config.target_offset]
                target_seq = item_sequence[i+self.config.target_offset:end_idx]
                
                if len(input_seq) > 0 and len(target_seq) > 0:
                    sample = {
                        'user_id': seq_data['user_id'],
                        'input_sequence': input_seq,
                        'target_sequence': target_seq
                    }
                    
                    # Add timestamp information
                    if self.config.include_timestamps and seq_data['timestamps']:
                        sample['input_timestamps'] = seq_data['timestamps'][i:end_idx-self.config.target_offset]
                        sample['target_timestamps'] = seq_data['timestamps'][i+self.config.target_offset:end_idx]
                    
                    training_samples.append(sample)
    
    return training_samples
```

#### pad_and_truncate_sequence(sequence)

Pad and truncate sequence.

```python
def pad_and_truncate_sequence(self, sequence: List[int]) -> List[int]:
    """
    Pad and truncate sequence to specified length
    
    Args:
        sequence: Input sequence
        
    Returns:
        Processed sequence
    """
    # Truncate
    if len(sequence) > self.config.max_seq_length:
        sequence = self.config.truncate_sequence(sequence)
    
    # Pad
    if len(sequence) < self.config.max_seq_length:
        sequence = self.config.pad_sequence(sequence)
    
    return sequence
```

#### create_attention_mask(sequence)

Create attention mask.

```python
def create_attention_mask(self, sequence: List[int]) -> List[int]:
    """
    Create attention mask for sequence
    
    Args:
        sequence: Input sequence
        
    Returns:
        Attention mask, 1 for valid positions, 0 for padding positions
    """
    mask = []
    for token in sequence:
        if token == self.config.padding_token:
            mask.append(0)
        else:
            mask.append(1)
    
    return mask
```

#### encode_time_features(timestamps)

Encode time features.

```python
def encode_time_features(self, timestamps: List[float]) -> np.ndarray:
    """
    Encode timestamps to feature vectors
    
    Args:
        timestamps: Timestamp list
        
    Returns:
        Time feature matrix (seq_len, time_encoding_dim)
    """
    if not timestamps:
        return np.zeros((0, self.config.time_encoding_dim))
    
    # Normalize timestamps
    timestamps = np.array(timestamps)
    min_time, max_time = timestamps.min(), timestamps.max()
    
    if max_time > min_time:
        normalized_times = (timestamps - min_time) / (max_time - min_time)
    else:
        normalized_times = np.zeros_like(timestamps)
    
    # Create sinusoidal encoding
    time_features = []
    for i in range(self.config.time_encoding_dim // 2):
        freq = 1.0 / (10000 ** (2 * i / self.config.time_encoding_dim))
        sin_features = np.sin(normalized_times * freq)
        cos_features = np.cos(normalized_times * freq)
        time_features.extend([sin_features, cos_features])
    
    # Transpose and truncate to specified dimension
    time_features = np.array(time_features[:self.config.time_encoding_dim]).T
    
    return time_features
```

## Data Augmentation Processor

### DataAugmentor

Data augmentation processor.

```python
class DataAugmentor:
    def __init__(self, augmentation_config: Dict[str, Any]):
        self.config = augmentation_config
        
    def augment_sequence(self, sequence: List[int]) -> List[int]:
        """
        Perform data augmentation on sequence
        
        Args:
            sequence: Original sequence
            
        Returns:
            Augmented sequence
        """
        augmented = sequence.copy()
        
        # Random drop
        if self.config.get('random_drop', False):
            drop_prob = self.config.get('drop_prob', 0.1)
            augmented = [item for item in augmented if random.random() > drop_prob]
        
        # Random shuffle
        if self.config.get('random_shuffle', False):
            shuffle_prob = self.config.get('shuffle_prob', 0.1)
            if random.random() < shuffle_prob:
                # Only shuffle partial subsequence
                start = random.randint(0, max(0, len(augmented) - 3))
                end = min(start + random.randint(2, 4), len(augmented))
                subseq = augmented[start:end]
                random.shuffle(subseq)
                augmented[start:end] = subseq
        
        # Random replace
        if self.config.get('random_replace', False):
            replace_prob = self.config.get('replace_prob', 0.05)
            vocab_size = self.config.get('vocab_size', 1000)
            
            for i in range(len(augmented)):
                if random.random() < replace_prob:
                    augmented[i] = random.randint(1, vocab_size)
        
        return augmented
```

## Preprocessing Pipeline

### PreprocessingPipeline

Data preprocessing pipeline.

```python
class PreprocessingPipeline:
    def __init__(
        self,
        text_processor: TextProcessor,
        sequence_processor: SequenceProcessor,
        augmentor: Optional[DataAugmentor] = None
    ):
        self.text_processor = text_processor
        self.sequence_processor = sequence_processor
        self.augmentor = augmentor
        
    def process_items(
        self,
        items_df: pd.DataFrame,
        cache_key: str = None
    ) -> pd.DataFrame:
        """
        Process item data
        
        Args:
            items_df: Items DataFrame
            cache_key: Cache key
            
        Returns:
            Processed items DataFrame with feature vectors
        """
        print("Processing item features...")
        
        # Encode text features
        embeddings = self.text_processor.encode_item_features(
            items_df, cache_key=cache_key
        )
        
        # Add features to DataFrame
        processed_df = items_df.copy()
        processed_df['features'] = embeddings.tolist()
        
        return processed_df
        
    def process_interactions(
        self,
        interactions_df: pd.DataFrame,
        items_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Process interaction data to generate sequences
        
        Args:
            interactions_df: Interactions DataFrame
            items_df: Items DataFrame
            
        Returns:
            Processed sequence data
        """
        print("Building user sequences...")
        
        # Build user sequences
        sequences = self.sequence_processor.build_user_sequences(interactions_df)
        
        # Create training samples
        training_samples = self.sequence_processor.create_training_samples(sequences)
        
        # Data augmentation
        if self.augmentor:
            augmented_samples = []
            for sample in training_samples:
                # Original sample
                augmented_samples.append(sample)
                
                # Augmented sample
                aug_input = self.augmentor.augment_sequence(sample['input_sequence'])
                aug_target = self.augmentor.augment_sequence(sample['target_sequence'])
                
                augmented_sample = sample.copy()
                augmented_sample['input_sequence'] = aug_input
                augmented_sample['target_sequence'] = aug_target
                augmented_samples.append(augmented_sample)
            
            training_samples = augmented_samples
        
        return training_samples
```

## Utility Functions

### compute_sequence_statistics(sequences)

Compute sequence statistics.

```python
def compute_sequence_statistics(sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute sequence data statistics
    
    Args:
        sequences: Sequence list
        
    Returns:
        Statistics dictionary
    """
    if not sequences:
        return {}
    
    lengths = [len(seq['item_sequence']) for seq in sequences]
    unique_users = len(set(seq['user_id'] for seq in sequences))
    
    # Compute item frequencies
    item_counts = {}
    for seq in sequences:
        for item_id in seq['item_sequence']:
            item_counts[item_id] = item_counts.get(item_id, 0) + 1
    
    stats = {
        'num_sequences': len(sequences),
        'num_unique_users': unique_users,
        'num_unique_items': len(item_counts),
        'avg_sequence_length': np.mean(lengths),
        'min_sequence_length': np.min(lengths),
        'max_sequence_length': np.max(lengths),
        'median_sequence_length': np.median(lengths),
        'total_interactions': sum(lengths),
        'most_popular_items': sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    }
    
    return stats
```

### visualize_embeddings(embeddings, labels, method)

Visualize embedding vectors.

```python
def visualize_embeddings(
    embeddings: np.ndarray,
    labels: List[str] = None,
    method: str = 'tsne',
    save_path: str = None
) -> None:
    """
    Visualize high-dimensional embedding vectors
    
    Args:
        embeddings: Embedding matrix (n_samples, embedding_dim)
        labels: Sample labels
        method: Dimensionality reduction method ('tsne', 'pca', 'umap')
        save_path: Save path
    """
    import matplotlib.pyplot as plt
    
    # Dimensionality reduction
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    elif method == 'umap':
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    if labels:
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                c=[colors[i]],
                label=label,
                alpha=0.7
            )
        plt.legend()
    else:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
    
    plt.title(f'Embedding Visualization ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
```

## Usage Examples

### Text Processing

```python
from genrec.data.processors import TextProcessor
from genrec.data.configs import TextEncodingConfig

# Create configuration
config = TextEncodingConfig(
    encoder_model="sentence-transformers/all-MiniLM-L6-v2",
    template="Title: {title}; Category: {category}",
    batch_size=32
)

# Create processor
processor = TextProcessor(config)

# Encode texts
texts = ["Apple iPhone 13", "Samsung Galaxy S21", "Sony WH-1000XM4"]
embeddings = processor.encode_texts(texts, cache_key="sample_texts")

print(f"Embeddings shape: {embeddings.shape}")
```

### Sequence Processing

```python
from genrec.data.processors import SequenceProcessor
from genrec.data.configs import SequenceConfig

# Create configuration
config = SequenceConfig(
    max_seq_length=50,
    min_seq_length=3,
    target_offset=1
)

# Create processor
processor = SequenceProcessor(config)

# Process interaction data
sequences = processor.build_user_sequences(interactions_df)
training_samples = processor.create_training_samples(sequences)

print(f"Generated {len(training_samples)} training samples")
```

### Complete Preprocessing Pipeline

```python
from genrec.data.processors import PreprocessingPipeline

# Create pipeline
pipeline = PreprocessingPipeline(
    text_processor=text_processor,
    sequence_processor=sequence_processor
)

# Process data
processed_items = pipeline.process_items(items_df, cache_key="items_beauty")
processed_sequences = pipeline.process_interactions(interactions_df, processed_items)

# View statistics
stats = compute_sequence_statistics(processed_sequences)
print(f"Dataset statistics: {stats}")
```