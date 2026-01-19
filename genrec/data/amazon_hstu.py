"""
Amazon Dataset for HSTU training.
Extends SASRec dataset with timestamp support for temporal attention bias.
"""
import os
import gin
import torch
from torch.utils.data import Dataset
from typing import Dict, List
from tqdm import tqdm

from genrec.data.amazon import DATASET_CONFIGS, parse_gzip_json


@gin.configurable
class AmazonHSTUDataset(Dataset):
    """
    Amazon Sequence Dataset for HSTU with timestamp support.

    Returns item ID sequences along with their timestamps for temporal bias.
    """

    def __init__(
        self,
        root: str = "dataset/amazon",
        split: str = "beauty",
        train_test_split: str = "train",
        max_seq_len: int = 50,
        min_seq_len: int = 5,
    ) -> None:
        self.root = root
        self.split = split.lower()
        self.train_test_split = train_test_split
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len

        self._load_sequences()
        self._generate_samples()

    def _load_sequences(self) -> None:
        """Load user interaction sequences with timestamps."""
        config = DATASET_CONFIGS[self.split]
        reviews_path = os.path.join(self.root, "raw", self.split, config["reviews"])

        user_sequences: Dict[str, List[tuple]] = {}
        self.item_id_mapping: Dict[str, int] = {}

        print(f"Loading sequences from {reviews_path}...")
        for review in tqdm(parse_gzip_json(reviews_path), desc="Processing reviews"):
            asin = review.get('asin')
            user_id = review.get('reviewerID')
            timestamp = review.get('unixReviewTime', 0)

            if asin and user_id:
                if asin not in self.item_id_mapping:
                    self.item_id_mapping[asin] = len(self.item_id_mapping) + 1

                item_id = self.item_id_mapping[asin]
                if user_id not in user_sequences:
                    user_sequences[user_id] = []
                user_sequences[user_id].append((timestamp, item_id))

        # Sort by timestamp and filter short sequences
        self.sequences = []  # List of (items, timestamps)
        for uid, seq in user_sequences.items():
            seq.sort(key=lambda x: x[0])
            timestamps = [x[0] for x in seq]
            items = [x[1] for x in seq]
            if len(items) >= self.min_seq_len:
                self.sequences.append((items, timestamps))

        self.num_items = len(self.item_id_mapping)
        print(f"Loaded {len(self.sequences)} sequences, {self.num_items} items")

    def _generate_samples(self) -> None:
        """Generate training/evaluation samples."""
        self.samples = []

        if self.train_test_split == "train":
            for items, timestamps in tqdm(self.sequences, desc="Generating train samples"):
                # Leave last 2 for valid/test
                items = items[:-2]
                timestamps = timestamps[:-2]
                if len(items) < 2:
                    continue
                # Sliding window
                for i in range(1, len(items)):
                    start = max(0, i - self.max_seq_len)
                    history = items[start:i]
                    history_ts = timestamps[start:i]
                    target = items[i]
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
                history = items[start:-1]
                history_ts = timestamps[start:-1]
                target = items[-1]
                self.samples.append({
                    'history': history,
                    'timestamps': history_ts,
                    'target': target,
                })

        else:  # test
            for items, timestamps in self.sequences:
                if len(items) < 2:
                    continue
                start = max(0, len(items) - 1 - self.max_seq_len)
                history = items[start:-1]
                history_ts = timestamps[start:-1]
                target = items[-1]
                self.samples.append({
                    'history': history,
                    'timestamps': history_ts,
                    'target': target,
                })

        print(f"Generated {len(self.samples)} {self.train_test_split} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def hstu_collate_fn(batch: List[Dict], max_seq_len: int = 50):
    """
    Collate function for HSTU training.

    Pads sequences and timestamps, creates input/target tensors.
    """
    histories = [b['history'] for b in batch]
    timestamps_list = [b['timestamps'] for b in batch]
    targets = [b['target'] for b in batch]

    max_len = min(max(len(h) for h in histories), max_seq_len)

    input_ids = []
    target_ids = []
    timestamps = []

    for history, ts, target in zip(histories, timestamps_list, targets):
        if len(history) > max_len:
            history = history[-max_len:]
            ts = ts[-max_len:]

        seq = history + [target]
        ts_seq = ts + [ts[-1] if ts else 0]  # Use last timestamp for target

        pad_len = max_len + 1 - len(seq)
        padded_seq = [0] * pad_len + seq
        padded_ts = [0] * pad_len + ts_seq

        input_ids.append(padded_seq[:-1])
        target_ids.append(padded_seq[1:])
        timestamps.append(padded_ts[:-1])

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'targets': torch.tensor(target_ids, dtype=torch.long),
        'timestamps': torch.tensor(timestamps, dtype=torch.long),
    }


def hstu_eval_collate_fn(batch: List[Dict], max_seq_len: int = 50):
    """Collate function for HSTU evaluation."""
    histories = [b['history'] for b in batch]
    timestamps_list = [b['timestamps'] for b in batch]
    targets = [b['target'] for b in batch]

    max_len = min(max(len(h) for h in histories), max_seq_len)

    input_ids = []
    timestamps = []

    for history, ts in zip(histories, timestamps_list):
        if len(history) > max_len:
            history = history[-max_len:]
            ts = ts[-max_len:]

        pad_len = max_len - len(history)
        input_ids.append([0] * pad_len + history)
        timestamps.append([0] * pad_len + ts)

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'targets': torch.tensor(targets, dtype=torch.long),
        'timestamps': torch.tensor(timestamps, dtype=torch.long),
    }


if __name__ == "__main__":
    dataset = AmazonHSTUDataset(
        root="dataset/amazon",
        split="beauty",
        train_test_split="train",
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Num items: {dataset.num_items}")
    sample = dataset[0]
    print(f"Sample: history={sample['history'][:5]}..., timestamps={sample['timestamps'][:5]}..., target={sample['target']}")
