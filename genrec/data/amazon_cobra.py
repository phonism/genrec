"""
Amazon COBRA Dataset for COBRA model training.
Provides semantic IDs and tokenized text inputs for the SentenceT5 encoder.
"""
import os
import gin
import torch

from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any

from genrec.data.amazon import (
    AmazonItemDataset,
    DATASET_CONFIGS,
    parse_gzip_json,
)


@gin.configurable
class AmazonCobraDataset(Dataset):
    """
    Amazon Sequence Dataset for COBRA training.

    Loads user interaction sequences and provides both semantic IDs and
    tokenized text inputs for the SentenceT5 encoder.

    Training uses sliding window strategy:
    - For each user sequence, generate N-1 training samples
    - Each item (except first) serves as a target once

    Valid/Test uses leave-one-out:
    - Valid: history = seq[:-2], target = seq[-2]
    - Test: history = seq[:-1], target = seq[-1]
    """

    def __init__(
        self,
        root: str = "dataset/amazon",
        split: str = "beauty",
        train_test_split: str = "train",
        max_seq_len: int = 20,
        pretrained_rqvae_path: str = "./out/rqvae/amazon/{split}/checkpoint.pt",
        encoder_model_name: str = "./models_hub/sentence-t5-xl",
        max_text_len: int = 128,
        # RQVAE config - should match pretrained model
        rqvae_input_dim: int = 768,
        rqvae_embed_dim: int = 32,
        rqvae_hidden_dims: List[int] = [512, 256, 128, 64],
        rqvae_codebook_size: int = 256,
        rqvae_n_layers: int = 3,
    ) -> None:
        from genrec.models.rqvae import RqVae

        self.root = root
        self.split = split.lower()
        self.train_test_split = train_test_split
        self._max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.n_codebooks = rqvae_n_layers
        self.codebook_size = rqvae_codebook_size

        # Replace {split} placeholder in rqvae path
        pretrained_rqvae_path = pretrained_rqvae_path.format(split=self.split)

        # Load SentenceT5 tokenizer
        st_model = SentenceTransformer(encoder_model_name)
        self.tokenizer = st_model.tokenizer

        # Load item dataset for embeddings
        item_dataset = AmazonItemDataset(
            root=root,
            split=split,
            train_test_split="all",
            encoder_model_name=encoder_model_name,
        )
        self.item_embeddings = torch.tensor(item_dataset.embeddings, dtype=torch.float32)

        # Load pretrained RQVAE and generate semantic IDs
        rqvae = RqVae(
            input_dim=rqvae_input_dim,
            embed_dim=rqvae_embed_dim,
            hidden_dims=rqvae_hidden_dims,
            codebook_size=rqvae_codebook_size,
            codebook_kmeans_init=False,
            codebook_normalize=False,
            codebook_sim_vq=False,
            n_layers=rqvae_n_layers,
            n_cat_features=0,
            commitment_weight=0.25,
        )
        rqvae.load_pretrained(pretrained_rqvae_path)
        rqvae.eval()

        with torch.no_grad():
            self.sem_ids_list = rqvae.get_semantic_ids(self.item_embeddings).sem_ids.tolist()

        # Load item metadata for text
        self._load_item_metadata()

        # Load user sequences and generate samples
        self._load_sequences()
        self._generate_samples()

    def _load_item_metadata(self) -> None:
        """Load item metadata for text generation."""
        config = DATASET_CONFIGS[self.split]
        meta_path = os.path.join(self.root, "raw", self.split, config["meta"])
        reviews_path = os.path.join(self.root, "raw", self.split, config["reviews"])

        # Build item mapping from reviews
        item_id_mapping: Dict[str, int] = {}
        for review in parse_gzip_json(reviews_path):
            asin = review.get('asin')
            if asin and asin not in item_id_mapping:
                item_id_mapping[asin] = len(item_id_mapping)

        # Load metadata
        self.item_texts: Dict[int, str] = {}
        for meta in parse_gzip_json(meta_path):
            asin = meta.get('asin')
            if asin in item_id_mapping:
                item_id = item_id_mapping[asin]
                title = meta.get('title', '')
                brand = meta.get('brand', '')
                text = f"{title} {brand}".strip() or f"item_{item_id}"
                self.item_texts[item_id] = text

        # Fill missing items
        for i in range(len(item_id_mapping)):
            if i not in self.item_texts:
                self.item_texts[i] = f"item_{i}"

    def _load_sequences(self) -> None:
        """Load user interaction sequences from reviews."""
        config = DATASET_CONFIGS[self.split]
        reviews_path = os.path.join(self.root, "raw", self.split, config["reviews"])

        user_sequences: Dict[str, List[tuple]] = {}
        item_id_mapping: Dict[str, int] = {}

        for review in parse_gzip_json(reviews_path):
            asin = review.get('asin')
            user_id = review.get('reviewerID')
            timestamp = review.get('unixReviewTime', 0)

            if asin and user_id:
                if asin not in item_id_mapping:
                    item_id_mapping[asin] = len(item_id_mapping)

                item_id = item_id_mapping[asin]
                if user_id not in user_sequences:
                    user_sequences[user_id] = []
                user_sequences[user_id].append((timestamp, item_id))

        # Sort by timestamp and filter short sequences
        self.sequences = []
        self.user_ids = []
        for uid, seq in user_sequences.items():
            seq.sort(key=lambda x: x[0])
            items = [x[1] for x in seq]
            if len(items) >= 5:
                self.sequences.append(items)
                self.user_ids.append(uid)

        print(f"Loaded {len(self.sequences)} user sequences for COBRA")

    def _generate_samples(self) -> None:
        """Generate training/evaluation samples.

        COBRA uses teacher forcing, so we use full sequence training:
        - Each user = one sample (no sliding window)
        - Train: history = seq[:-2], target = seq[-2] (leave last for test)
        - Valid: history = seq[:-2], target = seq[-2]
        - Test: history = seq[:-1], target = seq[-1]
        """
        self.samples = []

        if self.train_test_split == "train":
            # Full sequence training: one sample per user
            for user_idx, full_seq in enumerate(self.sequences):
                user_id = hash(self.user_ids[user_idx]) % 10000
                # Leave last 2 items for valid/test, use rest for training
                seq = full_seq[:-2]
                if len(seq) >= 2:  # Need at least 2 items (1 history + 1 target)
                    self.samples.append({
                        'user_id': user_id,
                        'history': seq[:-1],  # All but last as history
                        'target': seq[-1],    # Last as target
                    })
        elif self.train_test_split == "valid":
            for user_idx, full_seq in enumerate(self.sequences):
                user_id = hash(self.user_ids[user_idx]) % 10000
                seq = full_seq[:-1]
                self.samples.append({
                    'user_id': user_id,
                    'history': seq[:-1],
                    'target': seq[-1],
                })
        else:  # test
            for user_idx, full_seq in enumerate(self.sequences):
                user_id = hash(self.user_ids[user_idx]) % 10000
                self.samples.append({
                    'user_id': user_id,
                    'history': full_seq[:-1],
                    'target': full_seq[-1],
                })

        print(f"Generated {len(self.samples)} COBRA samples for {self.train_test_split}")

    def _truncate_history(self, history: List[int]) -> List[int]:
        """Truncate history to max_seq_len (take last N items)."""
        if len(history) > self._max_seq_len:
            return history[-self._max_seq_len:]
        return history

    def _tokenize_items(self, item_ids: List[int]) -> torch.Tensor:
        """Tokenize item texts for SentenceT5 encoder."""
        texts = [self.item_texts.get(i, f"item_{i}") for i in item_ids]
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_tensors='pt'
        )
        return encoded['input_ids']  # (T, L)

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        history_items = self._truncate_history(sample['history'])
        target_item = sample['target']

        # Convert to semantic IDs (flattened)
        item_sem_ids = []
        for item_id in history_items:
            if item_id < len(self.sem_ids_list):
                item_sem_ids.extend(self.sem_ids_list[item_id])
            else:
                item_sem_ids.extend([0] * self.n_codebooks)

        # Tokenize texts for history
        encoder_input_ids = self._tokenize_items(history_items)

        # Target semantic IDs
        target_sem_ids = self.sem_ids_list[target_item] if target_item < len(self.sem_ids_list) else [0] * self.n_codebooks

        # Tokenize target item text
        target_encoder_input_ids = self._tokenize_items([target_item])  # (1, L)

        return {
            'input_ids': item_sem_ids,  # List[int] of length T*C
            'encoder_input_ids': encoder_input_ids,  # (T, L)
            'target_sem_ids': target_sem_ids,  # List[int] of length C
            'target_encoder_input_ids': target_encoder_input_ids,  # (1, L)
        }


if __name__ == "__main__":
    # Test dataset creation
    dataset = AmazonCobraDataset(
        root="dataset/amazon",
        split="beauty",
        train_test_split="train",
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"First sample keys: {dataset[0].keys()}")
