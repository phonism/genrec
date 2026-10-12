"""
Amazon Reviews Dataset for RQVAE training.
Supports automatic download and processing of Amazon Review 2014 5-core data.
"""
import gzip
import fcntl
import hashlib
import json
import logging
import os
import urllib.request
import numpy as np
import pandas as pd
import torch
import gin

from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from genrec.data.evaluation_protocol import select_protocol_sequences
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def semantic_id_layer_usage(
    semantic_ids: torch.Tensor,
    codebook_size: int,
) -> List[float]:
    """Return the fraction of used codes in each SID layer."""
    semantic_ids = torch.as_tensor(semantic_ids)
    if semantic_ids.ndim != 2:
        raise ValueError("semantic_ids must have shape [items, layers]")
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    return [
        int(torch.unique(semantic_ids[:, layer]).numel()) / codebook_size
        for layer in range(semantic_ids.shape[1])
    ]


def stable_hash_bucket(value: str, num_buckets: int) -> int:
    """Map a string to a stable feature-hashing bucket.

    Python's built-in ``hash`` is salted per process, which makes a user token
    change between training and a later standalone evaluation process.  TIGER
    only specifies the hashing trick, not a concrete hash function, so use a
    fixed BLAKE2 digest for reproducible bucket assignments.
    """
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")
    digest = hashlib.blake2b(
        value.encode("utf-8"), digest_size=8, person=b"genrec"
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % num_buckets


# Amazon Review 2014 download URLs
AMAZON_REVIEW_BASE_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"

DATASET_CONFIGS = {
    # Amazon 2014 datasets
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
    "home": {
        "reviews": "reviews_Home_and_Kitchen_5.json.gz",
        "meta": "meta_Home_and_Kitchen.json.gz",
    },
    # Amazon 2023 datasets (processed by genrec.data.amazon2023)
    "books": {
        "reviews": "reviews_Books_5.json.gz",
        "meta": "meta_Books.json.gz",
    },
    "arts_crafts_and_sewing": {
        "reviews": "reviews_Arts_Crafts_and_Sewing_5.json.gz",
        "meta": "meta_Arts_Crafts_and_Sewing.json.gz",
    },
}


def download_file(url: str, dest_path: str) -> None:
    """Download file with progress bar."""
    if os.path.exists(dest_path):
        logger.info(f"File already exists: {dest_path}")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info(f"Downloading {url} -> {dest_path}")

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

    logger.info(f"Downloaded: {dest_path}")


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
        item_text_format: str = "legacy",
    ) -> None:
        self.root = root
        self.split = split.lower()
        self.train_test_split = train_test_split
        self.encoder_model_name = encoder_model_name
        self.force_regenerate = force_regenerate
        self.item_text_format = item_text_format

        if item_text_format not in {"legacy", "tiger_paper", "cobra_paper"}:
            raise ValueError(
                "item_text_format must be 'legacy', 'tiger_paper' or "
                "'cobra_paper', got "
                f"{item_text_format!r}"
            )

        if self.split not in DATASET_CONFIGS:
            raise ValueError(f"Unknown split: {split}. Available: {list(DATASET_CONFIGS.keys())}")

        # Paths — include encoder name in parquet filename to avoid cache conflicts
        self.raw_dir = os.path.join(root, "raw", self.split)
        self.processed_dir = os.path.join(root, "processed", self.split)
        encoder_short_name = os.path.basename(encoder_model_name.rstrip("/"))
        format_suffix = "" if item_text_format == "legacy" else f"_{item_text_format}"
        self.parquet_path = os.path.join(
            self.processed_dir,
            f"item_emb_{encoder_short_name}{format_suffix}.parquet",
        )

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
        logger.info(f"Loaded {len(self.embeddings)} embeddings from {self.parquet_path}")

    def _generate_embeddings(self) -> None:
        """Generate embeddings once across concurrent Codex sessions."""
        os.makedirs(self.processed_dir, exist_ok=True)
        lock_path = self.parquet_path + ".lock"
        with open(lock_path, "a+") as lock_file:
            logger.info(f"Waiting for embedding cache lock: {lock_path}")
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            # Another task may have populated the cache while this constructor
            # was waiting. Recheck only after acquiring the OS lock.
            if os.path.exists(self.parquet_path) and not self.force_regenerate:
                self._load_embeddings()
                return
            self._generate_embeddings_unlocked()

    def _generate_embeddings_unlocked(self) -> None:
        """Generate one cache while the caller owns its exclusive lock."""
        config = DATASET_CONFIGS[self.split]

        reviews_path = os.path.join(self.raw_dir, config["reviews"])
        meta_path = os.path.join(self.raw_dir, config["meta"])

        # Build item mapping from reviews (preserves order)
        logger.info("Building item mapping from reviews...")
        item_id_mapping: Dict[str, int] = {}
        for review in tqdm(parse_gzip_json(reviews_path), desc="Processing reviews"):
            asin = review.get('asin')
            if asin and asin not in item_id_mapping:
                item_id_mapping[asin] = len(item_id_mapping) + 1

        logger.info(f"Found {len(item_id_mapping)} unique items")

        # Load metadata
        logger.info("Loading metadata...")
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
                    'description': meta.get('description'),
                }

        logger.info(f"Loaded metadata for {len(item_info)} items")

        # Generate embeddings (batch encode for efficiency with large models)
        logger.info(f"Generating embeddings with {self.encoder_model_name}...")
        model = SentenceTransformer(self.encoder_model_name)

        sorted_ids = sorted(item_info.keys())
        texts = []
        for item_id in sorted_ids:
            info = item_info[item_id]
            if self.item_text_format == "tiger_paper":
                # TIGER section 4.1: title, price, brand and category.  Keep a
                # separate cache from the legacy template, which also includes
                # salesRank, so changing formats can never silently reuse stale
                # embeddings.
                semantics = (
                    f"title: {info.get('title', '')}\n"
                    f"price: {info.get('price', '')}\n"
                    f"brand: {info.get('brand', '')}\n"
                    f"category: {info.get('categories', '')}"
                )
            elif self.item_text_format == "cobra_paper":
                # COBRA section 4.1 lists exactly these four public-dataset
                # attributes.  Keep it separate from the historical template,
                # which used salesRank/brand and omitted description.
                semantics = (
                    f"title: {info.get('title', '')}\n"
                    f"price: {info.get('price', '')}\n"
                    f"category: {info.get('categories', '')}\n"
                    f"description: {info.get('description', '')}"
                )
            else:
                semantics = (
                    f"'title':{info.get('title', '')}\n"
                    f" 'price':{info.get('price', '')}\n"
                    f" 'salesRank':{info.get('salesRank', '')}\n"
                    f" 'brand':{info.get('brand', '')}\n"
                    f" 'categories':{info.get('categories', '')}"
                )
            texts.append(semantics)

        logger.info(f"Batch encoding {len(texts)} items...")
        all_embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

        item_embeddings: List[dict] = []
        for i, item_id in enumerate(sorted_ids):
            item_embeddings.append({
                'ItemID': item_id,
                'embedding': all_embeddings[i].tolist()
            })

        # Save to parquet
        df = pd.DataFrame(item_embeddings)
        temporary_path = f"{self.parquet_path}.part.{os.getpid()}"
        try:
            df.to_parquet(temporary_path, index=False)
            os.replace(temporary_path, self.parquet_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        logger.info(f"Saved embeddings to {self.parquet_path}")

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
        dedup_sinkhorn: bool = False,  # LC-Rec-style last-layer Sinkhorn dedup (keeps n codes)
        user_hash_buckets: int = 10000,
        pretrained_rqvae_path: str = "./out/rqvae/amazon/{split}/checkpoint.pt",
        encoder_model_name: str = "sentence-transformers/sentence-t5-base",
        item_text_format: str = "legacy",
        # RQVAE config - should match pretrained model
        rqvae_input_dim: int = 768,
        rqvae_embed_dim: int = 32,
        rqvae_hidden_dims: List[int] = [512, 256, 128, 64],
        rqvae_codebook_size: int = 256,
        rqvae_n_layers: int = 3,
        rqvae_mlp_activation: str = "silu",
        rqvae_mlp_bias: bool = False,
        min_raw_codebook_usage: float = 0.0,
        evaluation_protocol: str = "leave_one_out",
        temporal_train_ratio: float = 0.8,
        temporal_valid_ratio: float = 0.1,
    ) -> None:
        from genrec.models.rqvae import RqVae

        self.root = root
        self.split = split.lower()
        self.train_test_split = train_test_split
        self._max_seq_len = max_seq_len
        self.add_disambiguation = add_disambiguation
        self.dedup_sinkhorn = dedup_sinkhorn
        self.user_hash_buckets = user_hash_buckets
        self.item_text_format = item_text_format
        self.rqvae_codebook_size = rqvae_codebook_size
        self.rqvae_n_layers = rqvae_n_layers
        self.rqvae_mlp_activation = rqvae_mlp_activation
        self.rqvae_mlp_bias = rqvae_mlp_bias
        if not 0.0 <= min_raw_codebook_usage <= 1.0:
            raise ValueError("min_raw_codebook_usage must be in [0, 1]")
        self.evaluation_protocol = evaluation_protocol
        self.temporal_train_ratio = temporal_train_ratio
        self.temporal_valid_ratio = temporal_valid_ratio

        # Replace {split} placeholder in rqvae path
        pretrained_rqvae_path = pretrained_rqvae_path.format(split=self.split)

        # Load item dataset for embeddings
        item_dataset = AmazonItemDataset(
            root=root,
            split=split,
            train_test_split="all",
            encoder_model_name=encoder_model_name,
            item_text_format=item_text_format,
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
            mlp_activation=rqvae_mlp_activation,
            mlp_bias=rqvae_mlp_bias,
        )
        rqvae.load_pretrained(pretrained_rqvae_path)
        rqvae.eval()

        with torch.no_grad():
            nearest_sem_ids = rqvae.get_semantic_ids(
                self.item_embeddings
            ).sem_ids
            layer_usage = semantic_id_layer_usage(
                nearest_sem_ids, rqvae_codebook_size
            )
            if min(layer_usage, default=0.0) < min_raw_codebook_usage:
                raise RuntimeError(
                    "Refusing to build TIGER data from an unhealthy RQ-VAE: "
                    f"raw layer usage={layer_usage}, required minimum="
                    f"{min_raw_codebook_usage:.2%}."
                )
            if self.dedup_sinkhorn:
                # LC-Rec-style: resolve collisions by re-quantizing the last code
                # with Sinkhorn, keeping the same number of codes (no 4th token).
                final_sem_ids = rqvae.get_semantic_ids_dedup(
                    self.item_embeddings
                ).tolist()
            else:
                final_sem_ids = nearest_sem_ids.tolist()

        self.raw_sem_ids_list = [list(codes) for codes in nearest_sem_ids.tolist()]
        self.sem_ids_list = [list(codes) for codes in final_sem_ids]
        print(
            "TIGER RQ-VAE raw layer usage: "
            f"{[round(value, 4) for value in layer_usage]}"
        )

        # Disambiguation suffix and Sinkhorn dedup are alternative collision
        # strategies: running both would emit n+1-code IDs that downstream
        # sem_id_dim=n consumers silently misalign (the 4th code aliases into
        # codebook-0's token range). dedup_sinkhorn wins when both are set.
        if self.add_disambiguation and self.dedup_sinkhorn:
            logger.warning(
                "dedup_sinkhorn=True already resolves collisions — "
                "skipping the disambiguation suffix (add_disambiguation ignored)."
            )
        elif self.add_disambiguation:
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

        if max_collision > self.rqvae_codebook_size:
            raise ValueError(
                "collision suffix exceeds the configured codebook token range: "
                f"max_group={max_collision}, codebook_size={self.rqvae_codebook_size}"
            )

        logger.info(f"Semantic ID collisions: {collision_count} groups, {collision_items} items affected, max group size: {max_collision}")

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

        logger.info("Loading user sequences...")
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

        selected, self.protocol_info = select_protocol_sequences(
            user_sequences,
            self.train_test_split,
            protocol=self.evaluation_protocol,
            min_seq_len=5,
            temporal_train_ratio=self.temporal_train_ratio,
            temporal_valid_ratio=self.temporal_valid_ratio,
        )
        self.user_ids = [uid for uid, _ in selected]
        self.sequences = [[item_id for _, item_id in seq] for _, seq in selected]

        logger.info(
            f"Loaded {len(self.sequences)} user sequences "
            f"(protocol={self.evaluation_protocol})"
        )

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
                user_id = stable_hash_bucket(
                    self.user_ids[user_idx], self.user_hash_buckets
                )
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
                user_id = stable_hash_bucket(
                    self.user_ids[user_idx], self.user_hash_buckets
                )
                seq = full_seq[:-1]
                self.samples.append({
                    'user_id': user_id,
                    'history': seq[:-1],  # All except target
                    'target': seq[-1],    # Second to last of full_seq
                })
        else:  # test
            # Test: use full sequence
            for user_idx, full_seq in enumerate(self.sequences):
                user_id = stable_hash_bucket(
                    self.user_ids[user_idx], self.user_hash_buckets
                )
                self.samples.append({
                    'user_id': user_id,
                    'history': full_seq[:-1],  # All except last
                    'target': full_seq[-1],    # Last item
                })

        logger.info(f"Generated {len(self.samples)} samples for {self.train_test_split}")

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

        sid_width = (
            len(self.sem_ids_list[0])
            if self.sem_ids_list
            else self.rqvae_n_layers
        )
        target_sem_ids = (
            self.sem_ids_list[target_item]
            if target_item < len(self.sem_ids_list)
            else [0] * sid_width
        )

        return SeqData(
            user_id=user_id,
            item_ids=item_sem_ids,
            target_ids=target_sem_ids,
            target_item=target_item,
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
