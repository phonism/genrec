"""
Amazon Reviews Dataset for RQVAE training.
Supports automatic download and processing of Amazon Review 2014 5-core data.
"""
import gzip
import json
import os
import urllib.request
import numpy as np
import pandas as pd
import torch
import gin

from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import Dict, List, Optional


# Amazon Review 2014 download URLs
AMAZON_REVIEW_BASE_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"

DATASET_CONFIGS = {
    "beauty": {
        "reviews": "reviews_Beauty_5.json.gz",
        "meta": "meta_Beauty.json.gz",
    },
    "sports": {
        "reviews": "reviews_Sports_and_Outdoors_5.json.gz",
        "meta": "meta_Sports_and_Outdoors.json.gz",
    },
    "toys": {
        "reviews": "reviews_Toys_and_Games_5.json.gz",
        "meta": "meta_Toys_and_Games.json.gz",
    },
    "clothing": {
        "reviews": "reviews_Clothing_Shoes_and_Jewelry_5.json.gz",
        "meta": "meta_Clothing_Shoes_and_Jewelry.json.gz",
    },
}


def download_file(url: str, dest_path: str) -> None:
    """Download file with progress bar."""
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {url} -> {dest_path}")

    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get('content-length', 0))
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"Downloaded: {dest_path}")


def parse_gzip_json(path: str):
    """Parse gzipped JSON file line by line."""
    with gzip.open(path, 'rt', encoding='utf-8') as g:
        for line in g:
            try:
                yield json.loads(line.strip())
            except json.JSONDecodeError:
                # Handle malformed lines
                try:
                    yield eval(line.strip())
                except:
                    continue


@gin.configurable
class AmazonItemDataset(Dataset):
    """
    Amazon Item Dataset for RQVAE training.

    Automatically downloads and processes Amazon Review 2014 5-core data.
    Generates item embeddings using sentence transformers.

    Args:
        root: Root directory for data storage
        split: Dataset category ("beauty", "sports", "toys", "clothing")
        train_test_split: "all", "train", or "eval" (95%/5% random split)
        encoder_model_name: Sentence transformer model name or path
        force_regenerate: Regenerate embeddings even if cached
    """

    def __init__(
        self,
        root: str = "dataset/amazon",
        split: str = "beauty",
        train_test_split: str = "all",
        encoder_model_name: str = "sentence-transformers/sentence-t5-base",
        force_regenerate: bool = False,
    ) -> None:
        self.root = root
        self.split = split.lower()
        self.train_test_split = train_test_split
        self.encoder_model_name = encoder_model_name

        if self.split not in DATASET_CONFIGS:
            raise ValueError(f"Unknown split: {split}. Available: {list(DATASET_CONFIGS.keys())}")

        # Paths
        self.raw_dir = os.path.join(root, "raw", self.split)
        self.processed_dir = os.path.join(root, "processed", self.split)
        self.parquet_path = os.path.join(self.processed_dir, "item_emb.parquet")

        # Download if needed
        self._download_if_needed()

        # Load or generate embeddings
        if os.path.exists(self.parquet_path) and not force_regenerate:
            self._load_embeddings()
        else:
            self._generate_embeddings()

        # Apply train/eval split (95%/5% random split)
        self._apply_split()

    def _download_if_needed(self) -> None:
        """Download raw data files if not present."""
        os.makedirs(self.raw_dir, exist_ok=True)
        config = DATASET_CONFIGS[self.split]

        # Download reviews
        reviews_path = os.path.join(self.raw_dir, config["reviews"])
        if not os.path.exists(reviews_path):
            reviews_url = f"{AMAZON_REVIEW_BASE_URL}/{config['reviews']}"
            download_file(reviews_url, reviews_path)

        # Download metadata
        meta_path = os.path.join(self.raw_dir, config["meta"])
        if not os.path.exists(meta_path):
            meta_url = f"{AMAZON_REVIEW_BASE_URL}/{config['meta']}"
            download_file(meta_url, meta_path)

    def _load_embeddings(self) -> None:
        """Load embeddings from cached parquet file."""
        df = pd.read_parquet(self.parquet_path)
        self.embeddings = np.stack(df['embedding'].values, axis=0)
        self.dim = self.embeddings.shape[-1]
        print(f"Loaded {len(self.embeddings)} embeddings from {self.parquet_path}")

    def _generate_embeddings(self) -> None:
        """Generate embeddings from raw data."""
        os.makedirs(self.processed_dir, exist_ok=True)
        config = DATASET_CONFIGS[self.split]

        reviews_path = os.path.join(self.raw_dir, config["reviews"])
        meta_path = os.path.join(self.raw_dir, config["meta"])

        # Build item mapping from reviews (preserves order)
        print("Building item mapping from reviews...")
        item_id_mapping: Dict[str, int] = {}
        for review in tqdm(parse_gzip_json(reviews_path), desc="Processing reviews"):
            asin = review.get('asin')
            if asin and asin not in item_id_mapping:
                item_id_mapping[asin] = len(item_id_mapping) + 1

        print(f"Found {len(item_id_mapping)} unique items")

        # Load metadata
        print("Loading metadata...")
        item_info: Dict[int, dict] = {}
        for meta in tqdm(parse_gzip_json(meta_path), desc="Processing metadata"):
            asin = meta.get('asin')
            if asin in item_id_mapping:
                item_id = item_id_mapping[asin]
                item_info[item_id] = {
                    'title': meta.get('title'),
                    'price': meta.get('price'),
                    'salesRank': meta.get('salesRank'),
                    'brand': meta.get('brand'),
                    'categories': meta.get('categories'),
                }

        print(f"Loaded metadata for {len(item_info)} items")

        # Generate embeddings
        print(f"Generating embeddings with {self.encoder_model_name}...")
        model = SentenceTransformer(self.encoder_model_name)

        item_embeddings: List[dict] = []
        for item_id in tqdm(sorted(item_info.keys()), desc="Generating embeddings"):
            info = item_info[item_id]
            # Text format for embedding
            semantics = (
                f"'title':{info.get('title', '')}\n"
                f" 'price':{info.get('price', '')}\n"
                f" 'salesRank':{info.get('salesRank', '')}\n"
                f" 'brand':{info.get('brand', '')}\n"
                f" 'categories':{info.get('categories', '')}"
            )
            embedding = model.encode(semantics)
            item_embeddings.append({
                'ItemID': item_id,
                'embedding': embedding.tolist()
            })

        # Save to parquet
        df = pd.DataFrame(item_embeddings)
        df.to_parquet(self.parquet_path, index=False)
        print(f"Saved embeddings to {self.parquet_path}")

        # Store embeddings
        self.embeddings = np.stack(df['embedding'].values, axis=0)
        self.dim = self.embeddings.shape[-1]

    def _apply_split(self) -> None:
        """Apply train/eval split."""
        if self.train_test_split == "all":
            return

        gen = torch.Generator()
        gen.manual_seed(42)
        is_train = torch.rand(len(self.embeddings), generator=gen) > 0.05

        if self.train_test_split == "train":
            self.embeddings = self.embeddings[is_train.numpy()]
        elif self.train_test_split == "eval":
            self.embeddings = self.embeddings[~is_train.numpy()]

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> List[float]:
        return self.embeddings[idx].tolist()


@gin.configurable
class AmazonSeqDataset(Dataset):
    """
    Amazon Sequence Dataset for TIGER training.

    Loads user interaction sequences from Amazon Review data and generates
    semantic IDs using a pretrained RQVAE model.

    Training uses sliding window strategy (same as reference):
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
        subsample: bool = True,  # Kept for backward compatibility, ignored
        add_disambiguation: bool = True,  # Add 4th code to disambiguate collisions
        pretrained_rqvae_path: str = "./out/rqvae/amazon/{split}/checkpoint.pt",
        encoder_model_name: str = "sentence-transformers/sentence-t5-base",
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
        self.add_disambiguation = add_disambiguation

        # Replace {split} placeholder in rqvae path
        pretrained_rqvae_path = pretrained_rqvae_path.format(split=self.split)

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

        # Add disambiguation suffix to handle collisions (optional)
        if self.add_disambiguation:
            self.sem_ids_list = self._add_disambiguation_suffix(self.sem_ids_list)

        # Load user sequences and generate samples
        self._load_sequences()
        self._generate_samples()

    def _add_disambiguation_suffix(self, sem_ids_list: List[List[int]]) -> List[List[int]]:
        """Add 4th code to disambiguate items with identical semantic IDs.

        For items with the same 3-code semantic ID, append an incremental suffix:
        - Item A: [23, 45, 67] → [23, 45, 67, 0]
        - Item B: [23, 45, 67] → [23, 45, 67, 1]  # collision, increment
        """
        from collections import defaultdict

        # Group items by their 3-code semantic ID
        code_to_items = defaultdict(list)
        for item_id, codes in enumerate(sem_ids_list):
            key = tuple(codes)
            code_to_items[key].append(item_id)

        # Count collisions for logging
        collision_count = sum(1 for items in code_to_items.values() if len(items) > 1)
        collision_items = sum(len(items) for items in code_to_items.values() if len(items) > 1)
        max_collision = max(len(items) for items in code_to_items.values())

        print(f"Semantic ID collisions: {collision_count} groups, {collision_items} items affected, max group size: {max_collision}")

        # Assign disambiguation suffix
        new_sem_ids = []
        for item_id, codes in enumerate(sem_ids_list):
            key = tuple(codes)
            items_with_same_code = code_to_items[key]
            suffix = items_with_same_code.index(item_id)
            new_sem_ids.append(list(codes) + [suffix])

        return new_sem_ids

    def _load_sequences(self) -> None:
        """Load user interaction sequences from reviews."""
        config = DATASET_CONFIGS[self.split]
        reviews_path = os.path.join(self.root, "raw", self.split, config["reviews"])

        # Build user sequences sorted by timestamp
        user_sequences: Dict[str, List[tuple]] = {}
        item_id_mapping: Dict[str, int] = {}

        print("Loading user sequences...")
        for review in tqdm(parse_gzip_json(reviews_path), desc="Processing reviews"):
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
        # Need at least 5 items for leave-one-out (train needs >= 3 items)
        self.sequences = []
        self.user_ids = []
        for uid, seq in user_sequences.items():
            seq.sort(key=lambda x: x[0])
            items = [x[1] for x in seq]
            if len(items) >= 5:
                self.sequences.append(items)
                self.user_ids.append(uid)

        print(f"Loaded {len(self.sequences)} user sequences")

    def _generate_samples(self) -> None:
        """Generate training/evaluation samples.

        Training: Sliding window strategy (same as reference)
        - For sequence [i1, i2, i3, i4, i5], generate:
          - history=[i1], target=i2
          - history=[i1,i2], target=i3
          - history=[i1,i2,i3], target=i4
          - history=[i1,i2,i3,i4], target=i5
        - Exclude last 2 items for leave-one-out eval

        Valid/Test: Single sample per user
        - Valid: history=seq[:-2], target=seq[-2]
        - Test: history=seq[:-1], target=seq[-1]
        """
        self.samples = []

        if self.train_test_split == "train":
            # Sliding window for training
            for user_idx, full_seq in enumerate(tqdm(self.sequences, desc="Generating train samples")):
                user_id = hash(self.user_ids[user_idx]) % 10000
                # Exclude last 2 items (for valid and test)
                seq = full_seq[:-2]
                # Generate sliding window samples
                for i in range(1, len(seq)):
                    history_items = seq[:i]
                    target_item = seq[i]
                    self.samples.append({
                        'user_id': user_id,
                        'history': history_items,
                        'target': target_item,
                    })
        elif self.train_test_split == "valid":
            # Valid: exclude last 1 item (test target)
            for user_idx, full_seq in enumerate(self.sequences):
                user_id = hash(self.user_ids[user_idx]) % 10000
                seq = full_seq[:-1]
                self.samples.append({
                    'user_id': user_id,
                    'history': seq[:-1],  # All except target
                    'target': seq[-1],    # Second to last of full_seq
                })
        else:  # test
            # Test: use full sequence
            for user_idx, full_seq in enumerate(self.sequences):
                user_id = hash(self.user_ids[user_idx]) % 10000
                self.samples.append({
                    'user_id': user_id,
                    'history': full_seq[:-1],  # All except last
                    'target': full_seq[-1],    # Last item
                })

        print(f"Generated {len(self.samples)} samples for {self.train_test_split}")

    def _truncate_history(self, history: List[int]) -> List[int]:
        """Truncate history to max_seq_len (take last N items)."""
        if len(history) > self._max_seq_len:
            return history[-self._max_seq_len:]
        return history

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from genrec.data.schemas import SeqData

        sample = self.samples[idx]
        user_id = sample['user_id']
        history_items = self._truncate_history(sample['history'])
        target_item = sample['target']

        # Convert to semantic IDs
        item_sem_ids = []
        for item_id in history_items:
            if item_id < len(self.sem_ids_list):
                item_sem_ids.extend(self.sem_ids_list[item_id])

        target_sem_ids = self.sem_ids_list[target_item] if target_item < len(self.sem_ids_list) else [0, 0, 0]

        return SeqData(
            user_id=user_id,
            item_ids=item_sem_ids,
            target_ids=target_sem_ids,
        )


if __name__ == "__main__":
    # Test dataset creation with auto-download
    dataset = AmazonItemDataset(
        root="dataset/amazon",
        split="beauty",
        train_test_split="train",
        encoder_model_name="sentence-transformers/sentence-t5-base",
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Embedding dim: {dataset.dim}")
    print(f"First embedding shape: {len(dataset[0])}")
