"""
Amazon Dataset for SASRec training.
Simple sequence dataset returning item IDs directly (no semantic IDs).
"""
import os
import gin
import torch
from torch.utils.data import Dataset
from typing import Dict, List
from tqdm import tqdm

from genrec.data.amazon import DATASET_CONFIGS, parse_gzip_json


@gin.configurable
class AmazonSASRecDataset(Dataset):
    """
    Amazon Sequence Dataset for SASRec.

    Returns raw item ID sequences for next-item prediction.
    Uses leave-one-out evaluation strategy.

    Training: sliding window on seq[:-2]
    Valid: history=seq[:-2], target=seq[-2]
    Test: history=seq[:-1], target=seq[-1]
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
        """Load user interaction sequences from reviews."""
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
                    # Item IDs start from 1 (0 is padding)
                    self.item_id_mapping[asin] = len(self.item_id_mapping) + 1

                item_id = self.item_id_mapping[asin]
                if user_id not in user_sequences:
                    user_sequences[user_id] = []
                user_sequences[user_id].append((timestamp, item_id))

        # Sort by timestamp and filter short sequences
        self.sequences = []
        for uid, seq in user_sequences.items():
            seq.sort(key=lambda x: x[0])
            items = [x[1] for x in seq]
            if len(items) >= self.min_seq_len:
                self.sequences.append(items)

        self.num_items = len(self.item_id_mapping)
        print(f"Loaded {len(self.sequences)} sequences, {self.num_items} items")

    def _generate_samples(self) -> None:
        """Generate training/evaluation samples."""
        self.samples = []

        if self.train_test_split == "train":
            for full_seq in tqdm(self.sequences, desc="Generating train samples"):
                seq = full_seq[:-2]  # Leave last 2 for valid/test
                if len(seq) < 2:
                    continue
                # Sliding window: each position is a training sample
                for i in range(1, len(seq)):
                    history = seq[max(0, i - self.max_seq_len):i]
                    target = seq[i]
                    self.samples.append({'history': history, 'target': target})

        elif self.train_test_split == "valid":
            for full_seq in self.sequences:
                seq = full_seq[:-1]  # Exclude test target
                if len(seq) < 2:
                    continue
                history = seq[max(0, len(seq) - 1 - self.max_seq_len):-1]
                target = seq[-1]
                self.samples.append({'history': history, 'target': target})

        else:  # test
            for full_seq in self.sequences:
                if len(full_seq) < 2:
                    continue
                history = full_seq[max(0, len(full_seq) - 1 - self.max_seq_len):-1]
                target = full_seq[-1]
                self.samples.append({'history': history, 'target': target})

        print(f"Generated {len(self.samples)} {self.train_test_split} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        sample = self.samples[idx]
        return {
            'history': sample['history'],
            'target': sample['target'],
        }


def sasrec_collate_fn(batch: List[Dict], max_seq_len: int = 50):
    """
    Collate function for SASRec.

    Pads sequences to same length and creates input/target tensors.
    For training: input = [i1, i2, ..., in], target = [i2, i3, ..., in+1]
    """
    histories = [b['history'] for b in batch]
    targets = [b['target'] for b in batch]

    # Pad to max length in batch
    max_len = min(max(len(h) for h in histories), max_seq_len)

    input_ids = []
    target_ids = []

    for history, target in zip(histories, targets):
        # Truncate if needed
        if len(history) > max_len:
            history = history[-max_len:]

        # Create input and target sequences
        # Input: history padded to max_len
        # Target: history[1:] + [target] padded to max_len
        seq = history + [target]  # Full sequence including target

        # Pad from left (SASRec style)
        pad_len = max_len + 1 - len(seq)
        padded_seq = [0] * pad_len + seq

        input_ids.append(padded_seq[:-1])  # [pad, i1, i2, ..., in]
        target_ids.append(padded_seq[1:])   # [i1, i2, ..., in, target]

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'targets': torch.tensor(target_ids, dtype=torch.long),
    }


def sasrec_eval_collate_fn(batch: List[Dict], max_seq_len: int = 50):
    """Collate function for SASRec evaluation."""
    histories = [b['history'] for b in batch]
    targets = [b['target'] for b in batch]

    max_len = min(max(len(h) for h in histories), max_seq_len)

    input_ids = []
    for history in histories:
        if len(history) > max_len:
            history = history[-max_len:]
        pad_len = max_len - len(history)
        input_ids.append([0] * pad_len + history)

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'targets': torch.tensor(targets, dtype=torch.long),
    }


if __name__ == "__main__":
    dataset = AmazonSASRecDataset(
        root="dataset/amazon",
        split="beauty",
        train_test_split="train",
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Num items: {dataset.num_items}")
    print(f"Sample: {dataset[0]}")
