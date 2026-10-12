"""
MovieLens-1M dataset for HSTU training.

Auto-downloads ml-1m.zip from grouplens, parses ratings.dat,
applies item-side 5-core filtering, and emits the same sample schema as
AmazonHSTUDataset so that the existing HSTU collate functions can be reused.
"""
import io
import logging
import os
import urllib.request
import zipfile
from typing import Dict, List

import gin
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from genrec.data.amazon_hstu import hstu_collate_fn, hstu_eval_collate_fn  # noqa: F401

logger = logging.getLogger(__name__)

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def _download_and_extract(root: str) -> str:
    """Download ml-1m.zip if missing and return path to ratings.dat."""
    ratings_path = os.path.join(root, "ml-1m", "ratings.dat")
    if os.path.exists(ratings_path):
        return ratings_path

    os.makedirs(root, exist_ok=True)
    logger.info(f"Downloading {ML1M_URL} -> {root}/ml-1m.zip")
    with urllib.request.urlopen(ML1M_URL) as response:
        total = int(response.headers.get('content-length', 0))
        buf = io.BytesIO()
        with tqdm(total=total, unit='B', unit_scale=True, desc="Downloading ml-1m") as pbar:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                buf.write(chunk)
                pbar.update(len(chunk))
        buf.seek(0)
        with zipfile.ZipFile(buf) as z:
            z.extractall(root)

    if not os.path.exists(ratings_path):
        raise RuntimeError(f"Expected {ratings_path} after extraction")
    return ratings_path


def _parse_ratings(path: str):
    """Yield (user_id, item_id, timestamp) for every rating in ratings.dat."""
    with open(path, encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 4:
                continue
            uid, mid, _rating, ts = parts
            yield uid, mid, int(ts)


def _five_core_filter(user_to_seq: Dict, min_count: int = 5):
    """Iteratively drop items and users with fewer than min_count interactions."""
    while True:
        item_counts: Dict = {}
        for seq in user_to_seq.values():
            for _, item in seq:
                item_counts[item] = item_counts.get(item, 0) + 1

        before = (len(user_to_seq), len(item_counts))

        kept_items = {item for item, c in item_counts.items() if c >= min_count}
        new_user_to_seq = {}
        for u, seq in user_to_seq.items():
            filtered = [(t, i) for t, i in seq if i in kept_items]
            if len(filtered) >= min_count:
                new_user_to_seq[u] = filtered

        after_items = {i for seq in new_user_to_seq.values() for _, i in seq}
        after = (len(new_user_to_seq), len(after_items))

        user_to_seq = new_user_to_seq
        if after == before:
            return user_to_seq


@gin.configurable
class MovieLens1MHSTUDataset(Dataset):
    """MovieLens-1M sequence dataset for HSTU. Schema matches AmazonHSTUDataset."""

    def __init__(
        self,
        root: str = "dataset/ml1m",
        split: str = "ml1m",  # unused, kept for API parity with AmazonHSTUDataset
        train_test_split: str = "train",
        max_seq_len: int = 200,
        min_seq_len: int = 5,
    ) -> None:
        self.root = root
        self.train_test_split = train_test_split
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len

        self._load_sequences()
        self._generate_samples()

    def _load_sequences(self) -> None:
        ratings_path = _download_and_extract(self.root)

        user_to_seq: Dict[str, List] = {}
        for uid, mid, ts in tqdm(_parse_ratings(ratings_path), desc="Reading ratings"):
            user_to_seq.setdefault(uid, []).append((ts, mid))

        user_to_seq = _five_core_filter(user_to_seq, min_count=self.min_seq_len)

        # Stable item id remap (1-indexed; 0 reserved for padding).
        self.item_id_mapping: Dict[str, int] = {}
        for seq in user_to_seq.values():
            for _, item in seq:
                if item not in self.item_id_mapping:
                    self.item_id_mapping[item] = len(self.item_id_mapping) + 1

        self.sequences = []
        for seq in user_to_seq.values():
            seq.sort(key=lambda x: x[0])
            timestamps = [t for t, _ in seq]
            items = [self.item_id_mapping[i] for _, i in seq]
            self.sequences.append((items, timestamps))

        self.num_items = len(self.item_id_mapping)
        logger.info(f"Loaded {len(self.sequences)} sequences, {self.num_items} items")

    def _generate_samples(self) -> None:
        self.samples = []

        if self.train_test_split == "train":
            for items, timestamps in tqdm(self.sequences, desc="Generating train samples"):
                items = items[:-2]
                timestamps = timestamps[:-2]
                if len(items) < 2:
                    continue
                history = items[:-1]
                history_ts = timestamps[:-1]
                target = items[-1]
                self.samples.append({
                    'history': history,
                    'timestamps': history_ts,
                    'target': target,
                })

        elif self.train_test_split == "valid":
            for items, timestamps in self.sequences:
                items = items[:-1]
                timestamps = timestamps[:-1]
                if len(items) < 2:
                    continue
                start = max(0, len(items) - 1 - self.max_seq_len)
                self.samples.append({
                    'history': items[start:-1],
                    'timestamps': timestamps[start:-1],
                    'target': items[-1],
                })

        else:  # test
            for items, timestamps in self.sequences:
                if len(items) < 2:
                    continue
                start = max(0, len(items) - 1 - self.max_seq_len)
                self.samples.append({
                    'history': items[start:-1],
                    'timestamps': timestamps[start:-1],
                    'target': items[-1],
                })

        logger.info(f"Generated {len(self.samples)} {self.train_test_split} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


if __name__ == "__main__":
    ds = MovieLens1MHSTUDataset(train_test_split="train")
    print(f"Dataset size: {len(ds)}, num_items: {ds.num_items}")
    sample = ds[0]
    print(f"Sample: history[:5]={sample['history'][:5]}, target={sample['target']}")
