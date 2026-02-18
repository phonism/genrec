"""
Amazon Reviews 2023 Dataset.

Downloads and processes Amazon Reviews 2023 dataset from Hugging Face.
Generates 5-core filtered data compatible with GenRec training.

Dataset: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
Paper: "Bridging Language and Items for Retrieval and Recommendation" (arXiv:2403.03952)

Usage:
    # Download and process Books category (5-core)
    python -m genrec.data.amazon2023 --category Books --k_core 5

    # List available categories
    python -m genrec.data.amazon2023 --list_categories
"""

import argparse
import gzip
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import gin
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# Amazon 2023 Category Configurations
# =============================================================================

# Full list: https://amazon-reviews-2023.github.io/
AMAZON2023_CATEGORIES = {
    "All_Beauty": {"reviews": 701, "items": 112},
    "Amazon_Fashion": {"reviews": 883, "items": 186},
    "Appliances": {"reviews": 602, "items": 94},
    "Arts_Crafts_and_Sewing": {"reviews": 2875, "items": 303},
    "Automotive": {"reviews": 7990, "items": 962},
    "Baby_Products": {"reviews": 1752, "items": 69},
    "Beauty_and_Personal_Care": {"reviews": 8673, "items": 659},
    "Books": {"reviews": 29503, "items": 4403},
    "CDs_and_Vinyl": {"reviews": 4543, "items": 527},
    "Cell_Phones_and_Accessories": {"reviews": 6954, "items": 628},
    "Clothing_Shoes_and_Jewelry": {"reviews": 32200, "items": 5813},
    "Digital_Music": {"reviews": 130, "items": 70},
    "Electronics": {"reviews": 20994, "items": 1057},
    "Gift_Cards": {"reviews": 148, "items": 2},
    "Grocery_and_Gourmet_Food": {"reviews": 5074, "items": 334},
    "Handmade_Products": {"reviews": 223, "items": 49},
    "Health_and_Household": {"reviews": 10397, "items": 613},
    "Health_and_Personal_Care": {"reviews": 494, "items": 60},
    "Home_and_Kitchen": {"reviews": 18208, "items": 2050},
    "Industrial_and_Scientific": {"reviews": 1758, "items": 295},
    "Kindle_Store": {"reviews": 5722, "items": 1198},
    "Magazine_Subscriptions": {"reviews": 89, "items": 4},
    "Movies_and_TV": {"reviews": 8259, "items": 575},
    "Musical_Instruments": {"reviews": 1137, "items": 122},
    "Office_Products": {"reviews": 5581, "items": 420},
    "Patio_Lawn_and_Garden": {"reviews": 5236, "items": 513},
    "Pet_Supplies": {"reviews": 6380, "items": 242},
    "Software": {"reviews": 459, "items": 60},
    "Sports_and_Outdoors": {"reviews": 12980, "items": 1388},
    "Subscription_Boxes": {"reviews": 15, "items": 1},
    "Tools_and_Home_Improvement": {"reviews": 10356, "items": 1019},
    "Toys_and_Games": {"reviews": 8201, "items": 835},
    "Unknown": {"reviews": 2981, "items": 2368},
    "Video_Games": {"reviews": 2565, "items": 132},
}

# Data sources
MCAULEY_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
HF_BASE_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"


# =============================================================================
# Data Download and Processing Functions
# =============================================================================

def download_with_progress(url: str, dest_path: str, desc: str = "Downloading") -> None:
    """Download file with progress bar."""
    import requests

    if os.path.exists(dest_path):
        logger.info(f"File already exists: {dest_path}")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info(f"Downloading {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    logger.info(f"Downloaded: {dest_path}")


def download_amazon2023(
    category: str,
    output_dir: str,
    download_meta: bool = True,
    source: str = "huggingface",  # "huggingface" or "mcauley"
) -> Tuple[str, Optional[str]]:
    """Download Amazon 2023 dataset for a specific category."""
    category_normalized = category.replace(" ", "_").replace("&", "and")

    if category_normalized not in AMAZON2023_CATEGORIES:
        raise ValueError(f"Unknown category: {category}. Available: {list(AMAZON2023_CATEGORIES.keys())}")

    raw_dir = os.path.join(output_dir, "raw", category_normalized.lower())
    os.makedirs(raw_dir, exist_ok=True)

    if source == "huggingface":
        # Hugging Face: .jsonl format (uncompressed)
        reviews_filename = f"{category_normalized}.jsonl"
        reviews_url = f"{HF_BASE_URL}/review_categories/{reviews_filename}"
        meta_filename = f"meta_{category_normalized}.jsonl"
        meta_url = f"{HF_BASE_URL}/meta_categories/{meta_filename}"
    else:
        # McAuley Lab: .jsonl.gz format (compressed)
        reviews_filename = f"{category_normalized}.jsonl.gz"
        reviews_url = f"{MCAULEY_BASE_URL}/review_categories/{reviews_filename}"
        meta_filename = f"meta_{category_normalized}.jsonl.gz"
        meta_url = f"{MCAULEY_BASE_URL}/meta_categories/{meta_filename}"

    # Download reviews
    reviews_path = os.path.join(raw_dir, reviews_filename)
    download_with_progress(reviews_url, reviews_path, desc=f"Reviews ({category})")

    # Download metadata
    meta_path = None
    if download_meta:
        meta_path = os.path.join(raw_dir, meta_filename)
        download_with_progress(meta_url, meta_path, desc=f"Metadata ({category})")

    return reviews_path, meta_path


def parse_jsonl(path: str, max_lines: Optional[int] = None):
    """Parse JSONL file (supports both .jsonl and .jsonl.gz)."""
    count = 0
    is_gzipped = path.endswith('.gz')

    open_func = gzip.open if is_gzipped else open
    with open_func(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if max_lines and count >= max_lines:
                break
            try:
                yield json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            count += 1


def apply_k_core_filter(
    reviews_path: str,
    k: int = 5,
    max_reviews: Optional[int] = None,
) -> Tuple[Dict[str, List[dict]], Dict[str, int], Dict[str, int]]:
    """
    Apply k-core filtering to reviews.
    K-core: Keep only users and items with at least k interactions.
    """
    logger.info(f"Loading reviews from {reviews_path}...")

    interactions = []
    for review in tqdm(parse_jsonl(reviews_path, max_reviews), desc="Loading reviews"):
        user_id = review.get('user_id')
        item_id = review.get('parent_asin') or review.get('asin')
        timestamp = review.get('timestamp')
        rating = review.get('rating')

        if user_id and item_id and timestamp:
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'timestamp': timestamp,
                'rating': rating,
            })

    logger.info(f"Loaded {len(interactions)} interactions")
    logger.info(f"Applying {k}-core filtering...")

    iteration = 0
    while True:
        iteration += 1
        user_counts = defaultdict(int)
        item_counts = defaultdict(int)
        for inter in interactions:
            user_counts[inter['user_id']] += 1
            item_counts[inter['item_id']] += 1

        prev_count = len(interactions)
        interactions = [
            inter for inter in interactions
            if user_counts[inter['user_id']] >= k and item_counts[inter['item_id']] >= k
        ]

        logger.info(
            f"  Iteration {iteration}: {prev_count} -> {len(interactions)} "
            f"({len(set(i['user_id'] for i in interactions))} users, "
            f"{len(set(i['item_id'] for i in interactions))} items)"
        )

        if len(interactions) == prev_count:
            break
        if len(interactions) == 0:
            logger.warning(f"No interactions left after {k}-core filtering!")
            return {}, {}, {}

    user_ids = sorted(set(i['user_id'] for i in interactions))
    item_ids = sorted(set(i['item_id'] for i in interactions))
    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_id_map = {iid: idx + 1 for idx, iid in enumerate(item_ids)}

    logger.info(f"Final: {len(interactions)} interactions, {len(user_id_map)} users, {len(item_id_map)} items")

    user_reviews = defaultdict(list)
    for inter in interactions:
        user_reviews[inter['user_id']].append(inter)
    for uid in user_reviews:
        user_reviews[uid].sort(key=lambda x: x['timestamp'])

    return dict(user_reviews), user_id_map, item_id_map


def process_metadata(meta_path: str, item_id_map: Dict[str, int]) -> Dict[int, dict]:
    """Process item metadata."""
    logger.info(f"Loading metadata from {meta_path}...")

    item_meta = {}
    for meta in tqdm(parse_jsonl(meta_path), desc="Loading metadata"):
        item_id = meta.get('parent_asin') or meta.get('asin')
        if item_id in item_id_map:
            internal_id = item_id_map[item_id]
            item_meta[internal_id] = {
                'asin': item_id,
                'title': meta.get('title', ''),
                'description': ' '.join(meta.get('description', [])) if isinstance(meta.get('description'), list) else meta.get('description', ''),
                'price': meta.get('price'),
                'average_rating': meta.get('average_rating'),
                'rating_number': meta.get('rating_number'),
                'features': meta.get('features', []),
                'categories': meta.get('categories', []),
                'store': meta.get('store'),
                'brand': meta.get('details', {}).get('Brand') if meta.get('details') else None,
            }

    logger.info(f"Loaded metadata for {len(item_meta)} items")
    return item_meta


def save_processed_data(
    user_reviews: Dict[str, List[dict]],
    user_id_map: Dict[str, int],
    item_id_map: Dict[str, int],
    item_meta: Dict[int, dict],
    output_dir: str,
    category: str,
    k_core: int,
) -> str:
    """Save processed data in Amazon 2014 compatible format.

    Output structure (compatible with AmazonSASRecDataset):
        dataset/amazon2023/raw/books/reviews_Books_5.json.gz
        dataset/amazon2023/raw/books/meta_Books.json.gz
    """
    category_lower = category.lower().replace(" ", "_").replace("&", "and")
    # Use raw directory structure like Amazon 2014
    raw_dir = os.path.join(output_dir, "raw", category_lower)
    os.makedirs(raw_dir, exist_ok=True)

    # Save reviews in Amazon 2014 format
    reviews_filename = f"reviews_{category}_{k_core}.json.gz"
    reviews_path = os.path.join(raw_dir, reviews_filename)
    logger.info(f"Saving reviews to {reviews_path}...")
    with gzip.open(reviews_path, 'wt', encoding='utf-8') as f:
        for user_id, reviews in tqdm(user_reviews.items(), desc="Saving reviews"):
            for review in reviews:
                record = {
                    'reviewerID': user_id,
                    'asin': review['item_id'],
                    'overall': review.get('rating', 5.0),
                    'unixReviewTime': review['timestamp'],
                }
                f.write(json.dumps(record) + '\n')

    # Save metadata in Amazon 2014 format
    meta_filename = f"meta_{category}.json.gz"
    meta_path = os.path.join(raw_dir, meta_filename)
    logger.info(f"Saving metadata to {meta_path}...")
    with gzip.open(meta_path, 'wt', encoding='utf-8') as f:
        for internal_id, meta in tqdm(item_meta.items(), desc="Saving metadata"):
            record = {
                'asin': meta['asin'],
                'title': meta.get('title', ''),
                'description': meta.get('description', ''),
                'price': meta.get('price'),
                'brand': meta.get('brand'),
                'categories': meta.get('categories', []),
                'salesRank': {},  # Placeholder for Amazon 2014 compatibility
            }
            f.write(json.dumps(record) + '\n')

    # Save statistics
    stats = {
        'category': category,
        'k_core': k_core,
        'num_users': len(user_id_map),
        'num_items': len(item_id_map),
        'num_interactions': sum(len(reviews) for reviews in user_reviews.values()),
        'avg_seq_length': sum(len(reviews) for reviews in user_reviews.values()) / len(user_reviews) if user_reviews else 0,
    }
    with open(os.path.join(raw_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Users: {stats['num_users']:,}")
    logger.info(f"  Items: {stats['num_items']:,}")
    logger.info(f"  Interactions: {stats['num_interactions']:,}")
    logger.info(f"  Avg sequence length: {stats['avg_seq_length']:.2f}")
    logger.info(f"\nTo use with SASRec, add to amazon.py DATASET_CONFIGS:")
    logger.info(f'  "{category_lower}": {{"reviews": "{reviews_filename}", "meta": "{meta_filename}"}},')

    return raw_dir


def prepare_amazon2023(
    category: str = "Books",
    k_core: int = 5,
    output_dir: str = "dataset/amazon2023",
    max_reviews: Optional[int] = None,
    source: str = "huggingface",
) -> str:
    """
    Main function to download and prepare Amazon 2023 dataset.

    Args:
        category: Category name (e.g., "Books")
        k_core: K-core filtering threshold
        output_dir: Output directory
        max_reviews: Maximum reviews to process (for testing)
        source: Data source ("huggingface" or "mcauley")

    Returns:
        Path to processed data directory
    """
    logger.info(f"Processing Amazon 2023 - {category} ({k_core}-core) from {source}")

    reviews_path, meta_path = download_amazon2023(category, output_dir, source=source)
    user_reviews, user_id_map, item_id_map = apply_k_core_filter(reviews_path, k_core, max_reviews)

    if not user_reviews:
        raise ValueError("No data after filtering. Try reducing k_core value.")

    item_meta = process_metadata(meta_path, item_id_map)
    return save_processed_data(user_reviews, user_id_map, item_id_map, item_meta, output_dir, category, k_core)


# =============================================================================
# Dataset Classes (Compatible with GenRec trainers)
# =============================================================================

@gin.configurable
class Amazon2023ItemDataset(Dataset):
    """
    Amazon 2023 Item Dataset for RQVAE training.

    Args:
        root: Root directory for data storage
        split: Dataset split name (e.g., "books_5core")
        train_test_split: "all", "train", or "eval"
        encoder_model_name: Sentence transformer model name
        force_regenerate: Regenerate embeddings even if cached
    """

    def __init__(
        self,
        root: str = "dataset/amazon2023",
        split: str = "books_5core",
        train_test_split: str = "all",
        encoder_model_name: str = "sentence-transformers/sentence-t5-base",
        force_regenerate: bool = False,
    ) -> None:
        self.root = root
        self.split = split.lower()
        self.train_test_split = train_test_split
        self.encoder_model_name = encoder_model_name

        self.processed_dir = os.path.join(root, "processed", self.split)
        self.parquet_path = os.path.join(self.processed_dir, "item_emb.parquet")

        if not os.path.exists(self.processed_dir):
            raise FileNotFoundError(
                f"Processed data not found at {self.processed_dir}. "
                f"Run: python -m genrec.data.amazon2023 --category <category> --k_core <k>"
            )

        if os.path.exists(self.parquet_path) and not force_regenerate:
            self._load_embeddings()
        else:
            self._generate_embeddings()

        self._apply_split()

    def _load_embeddings(self) -> None:
        """Load embeddings from cached parquet file."""
        df = pd.read_parquet(self.parquet_path)
        self.embeddings = np.stack(df['embedding'].values, axis=0)
        self.dim = self.embeddings.shape[-1]
        logger.info(f"Loaded {len(self.embeddings)} embeddings from {self.parquet_path}")

    def _generate_embeddings(self) -> None:
        """Generate embeddings from metadata."""
        # Find metadata file
        meta_files = [f for f in os.listdir(self.processed_dir) if f.startswith('meta_') and f.endswith('.json.gz')]
        if not meta_files:
            raise FileNotFoundError(f"No metadata file found in {self.processed_dir}")

        meta_path = os.path.join(self.processed_dir, meta_files[0])
        logger.info(f"Generating embeddings from {meta_path}...")

        # Load metadata
        item_info = {}
        for meta in parse_jsonl(meta_path):
            asin = meta.get('asin')
            if asin:
                item_info[asin] = meta

        logger.info(f"Loaded {len(item_info)} items")

        # Generate embeddings
        model = SentenceTransformer(self.encoder_model_name)
        item_embeddings = []

        for idx, (asin, info) in enumerate(tqdm(item_info.items(), desc="Generating embeddings")):
            semantics = (
                f"'title':{info.get('title', '')}\n"
                f" 'price':{info.get('price', '')}\n"
                f" 'brand':{info.get('brand', '')}\n"
                f" 'categories':{info.get('categories', '')}\n"
                f" 'description':{info.get('description', '')[:500]}"
            )
            embedding = model.encode(semantics)
            item_embeddings.append({'ItemID': idx + 1, 'asin': asin, 'embedding': embedding.tolist()})

        df = pd.DataFrame(item_embeddings)
        df.to_parquet(self.parquet_path, index=False)
        logger.info(f"Saved embeddings to {self.parquet_path}")

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
class Amazon2023SASRecDataset(Dataset):
    """
    Amazon 2023 Sequence Dataset for SASRec/HSTU training.

    Args:
        root: Root directory
        split: Dataset split (e.g., "books_5core")
        train_test_split: "train", "valid", or "test"
        max_seq_len: Maximum sequence length
        min_seq_len: Minimum sequence length
    """

    def __init__(
        self,
        root: str = "dataset/amazon2023",
        split: str = "books_5core",
        train_test_split: str = "train",
        max_seq_len: int = 50,
        min_seq_len: int = 5,
    ) -> None:
        self.root = root
        self.split = split.lower()
        self.train_test_split = train_test_split
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len

        self.processed_dir = os.path.join(root, "processed", self.split)
        if not os.path.exists(self.processed_dir):
            raise FileNotFoundError(f"Processed data not found at {self.processed_dir}")

        self._load_sequences()
        self._generate_samples()

    def _load_sequences(self) -> None:
        """Load user sequences from reviews."""
        review_files = [f for f in os.listdir(self.processed_dir) if f.startswith('reviews_') and f.endswith('.json.gz')]
        if not review_files:
            raise FileNotFoundError(f"No review file found in {self.processed_dir}")

        reviews_path = os.path.join(self.processed_dir, review_files[0])
        logger.info(f"Loading sequences from {reviews_path}...")

        # Build item mapping and user sequences
        item_id_mapping: Dict[str, int] = {}
        user_sequences: Dict[str, List[tuple]] = {}

        for review in tqdm(parse_jsonl(reviews_path), desc="Processing reviews"):
            asin = review.get('asin')
            user_id = review.get('reviewerID')
            timestamp = review.get('unixReviewTime', 0)

            if asin and user_id:
                if asin not in item_id_mapping:
                    item_id_mapping[asin] = len(item_id_mapping) + 1
                item_id = item_id_mapping[asin]

                if user_id not in user_sequences:
                    user_sequences[user_id] = []
                user_sequences[user_id].append((timestamp, item_id))

        # Sort and filter
        self.sequences = []
        self.timestamps_list = []
        for uid, seq in user_sequences.items():
            seq.sort(key=lambda x: x[0])
            if len(seq) >= self.min_seq_len:
                self.sequences.append([x[1] for x in seq])
                self.timestamps_list.append([x[0] for x in seq])

        self.num_items = len(item_id_mapping)
        logger.info(f"Loaded {len(self.sequences)} sequences, {self.num_items} items")

    def _generate_samples(self) -> None:
        """Generate train/valid/test samples."""
        self.samples = []

        if self.train_test_split == "train":
            for seq, ts in zip(self.sequences, self.timestamps_list):
                train_seq = seq[:-2]
                train_ts = ts[:-2]
                for i in range(1, len(train_seq)):
                    history = train_seq[max(0, i - self.max_seq_len):i]
                    history_ts = train_ts[max(0, i - self.max_seq_len):i]
                    self.samples.append({
                        'history': history,
                        'timestamps': history_ts,
                        'target': train_seq[i],
                    })
        elif self.train_test_split == "valid":
            for seq, ts in zip(self.sequences, self.timestamps_list):
                val_seq = seq[:-1]
                val_ts = ts[:-1]
                history = val_seq[max(0, len(val_seq) - 1 - self.max_seq_len):-1]
                history_ts = val_ts[max(0, len(val_ts) - 1 - self.max_seq_len):-1]
                self.samples.append({
                    'history': history,
                    'timestamps': history_ts,
                    'target': val_seq[-1],
                })
        else:  # test
            for seq, ts in zip(self.sequences, self.timestamps_list):
                history = seq[max(0, len(seq) - 1 - self.max_seq_len):-1]
                history_ts = ts[max(0, len(ts) - 1 - self.max_seq_len):-1]
                self.samples.append({
                    'history': history,
                    'timestamps': history_ts,
                    'target': seq[-1],
                })

        logger.info(f"Generated {len(self.samples)} {self.train_test_split} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# =============================================================================
# CLI
# =============================================================================

def list_categories():
    """Print available categories."""
    print("\nAvailable Amazon 2023 Categories:")
    print("-" * 60)
    print(f"{'Category':<40} {'Reviews (K)':<12} {'Items (K)':<12}")
    print("-" * 60)

    sorted_cats = sorted(AMAZON2023_CATEGORIES.items(), key=lambda x: x[1]['reviews'], reverse=True)
    for cat, info in sorted_cats:
        print(f"{cat:<40} {info['reviews']:>8,}K {info['items']:>8,}K")

    print("-" * 60)
    print("\nRecommended for large-scale experiments:")
    print("  - Books (29.5M reviews)")
    print("  - Clothing_Shoes_and_Jewelry (32.2M reviews)")
    print("  - Electronics (21M reviews)")


def main():
    parser = argparse.ArgumentParser(description="Prepare Amazon Reviews 2023 dataset")
    parser.add_argument('--category', type=str, default='Books', help='Category to process')
    parser.add_argument('--k_core', type=int, default=5, help='K-core filtering threshold')
    parser.add_argument('--output_dir', type=str, default='dataset/amazon2023', help='Output directory')
    parser.add_argument('--max_reviews', type=int, default=None, help='Max reviews (for testing)')
    parser.add_argument('--source', type=str, default='huggingface', choices=['huggingface', 'mcauley'],
                        help='Data source: huggingface (faster) or mcauley (original)')
    parser.add_argument('--list_categories', action='store_true', help='List available categories')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.list_categories:
        list_categories()
        return

    processed_dir = prepare_amazon2023(
        category=args.category,
        k_core=args.k_core,
        output_dir=args.output_dir,
        max_reviews=args.max_reviews,
        source=args.source,
    )
    print(f"\nDone! Data saved to: {processed_dir}")


if __name__ == '__main__':
    main()
