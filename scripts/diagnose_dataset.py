"""
Diagnostic script to analyze valid vs test dataset differences.
"""
import sys
sys.path.insert(0, '.')

import torch
from collections import Counter
from genrec.data.amazon import AmazonSeqDataset

def diagnose():
    print("=" * 60)
    print("Loading datasets...")
    print("=" * 60)

    # Load all three splits
    train_dataset = AmazonSeqDataset(
        root="dataset/amazon",
        split="beauty",
        train_test_split="train",
        max_seq_len=20,
        pretrained_rqvae_path="./out/rqvae/amazon/beauty/checkpoint_epoch_999.pt",
    )

    valid_dataset = AmazonSeqDataset(
        root="dataset/amazon",
        split="beauty",
        train_test_split="valid",
        max_seq_len=20,
        pretrained_rqvae_path="./out/rqvae/amazon/beauty/checkpoint_epoch_999.pt",
    )

    test_dataset = AmazonSeqDataset(
        root="dataset/amazon",
        split="beauty",
        train_test_split="test",
        max_seq_len=20,
        pretrained_rqvae_path="./out/rqvae/amazon/beauty/checkpoint_epoch_999.pt",
    )

    print(f"\nDataset sizes (samples):")
    print(f"  Train: {len(train_dataset)} (sliding window)")
    print(f"  Valid: {len(valid_dataset)} (1 per user)")
    print(f"  Test:  {len(test_dataset)} (1 per user)")
    print(f"\nNumber of users: {len(train_dataset.sequences)}")

    # Collect target items from samples
    print("\n" + "=" * 60)
    print("Analyzing target items...")
    print("=" * 60)

    train_targets = [s['target'] for s in train_dataset.samples]
    valid_targets = [s['target'] for s in valid_dataset.samples]
    test_targets = [s['target'] for s in test_dataset.samples]

    train_target_set = set(train_targets)
    valid_target_set = set(valid_targets)
    test_target_set = set(test_targets)

    print(f"\nUnique target items:")
    print(f"  Train targets: {len(train_target_set)}")
    print(f"  Valid targets: {len(valid_target_set)}")
    print(f"  Test targets:  {len(test_target_set)}")

    # Check overlap
    valid_in_train = len(valid_target_set & train_target_set)
    test_in_train = len(test_target_set & train_target_set)
    test_in_valid = len(test_target_set & valid_target_set)

    print(f"\nTarget item overlap:")
    print(f"  Valid targets in train: {valid_in_train}/{len(valid_target_set)} ({100*valid_in_train/len(valid_target_set):.1f}%)")
    print(f"  Test targets in train:  {test_in_train}/{len(test_target_set)} ({100*test_in_train/len(test_target_set):.1f}%)")
    print(f"  Test targets in valid:  {test_in_valid}/{len(test_target_set)} ({100*test_in_valid/len(test_target_set):.1f}%)")

    # Check item frequency in training
    print("\n" + "=" * 60)
    print("Analyzing item frequency in training...")
    print("=" * 60)

    # Count all items that appear in training (history + target)
    all_train_items = []
    for sample in train_dataset.samples:
        all_train_items.extend(sample['history'])
        all_train_items.append(sample['target'])

    item_freq = Counter(all_train_items)

    # Frequency of valid targets
    valid_target_freqs = [item_freq.get(t, 0) for t in valid_targets]
    test_target_freqs = [item_freq.get(t, 0) for t in test_targets]

    print(f"\nTarget item frequency in training data:")
    print(f"  Valid targets - mean freq: {sum(valid_target_freqs)/len(valid_target_freqs):.2f}, "
          f"zero freq: {sum(1 for f in valid_target_freqs if f == 0)}/{len(valid_target_freqs)}")
    print(f"  Test targets  - mean freq: {sum(test_target_freqs)/len(test_target_freqs):.2f}, "
          f"zero freq: {sum(1 for f in test_target_freqs if f == 0)}/{len(test_target_freqs)}")

    # Check semantic ID collision
    print("\n" + "=" * 60)
    print("Analyzing semantic ID collision...")
    print("=" * 60)

    sem_ids_list = train_dataset.sem_ids_list
    sem_id_to_items = {}
    for item_id, sem_id in enumerate(sem_ids_list):
        key = tuple(sem_id)
        if key not in sem_id_to_items:
            sem_id_to_items[key] = []
        sem_id_to_items[key].append(item_id)

    collisions = {k: v for k, v in sem_id_to_items.items() if len(v) > 1}
    total_items = len(sem_ids_list)
    unique_sem_ids = len(sem_id_to_items)
    collision_items = sum(len(v) for v in collisions.values())

    print(f"\nSemantic ID statistics:")
    print(f"  Total items: {total_items}")
    print(f"  Unique semantic IDs: {unique_sem_ids}")
    print(f"  Collision rate: {100*(total_items - unique_sem_ids)/total_items:.2f}%")
    print(f"  Items with collision: {collision_items}")

    # Check if valid/test targets have collisions
    valid_collision_count = 0
    test_collision_count = 0

    for t in valid_targets:
        if t < len(sem_ids_list):
            key = tuple(sem_ids_list[t])
            if len(sem_id_to_items.get(key, [])) > 1:
                valid_collision_count += 1

    for t in test_targets:
        if t < len(sem_ids_list):
            key = tuple(sem_ids_list[t])
            if len(sem_id_to_items.get(key, [])) > 1:
                test_collision_count += 1

    print(f"\nTargets with semantic ID collision:")
    print(f"  Valid: {valid_collision_count}/{len(valid_targets)} ({100*valid_collision_count/len(valid_targets):.1f}%)")
    print(f"  Test:  {test_collision_count}/{len(test_targets)} ({100*test_collision_count/len(test_targets):.1f}%)")

    # Check history length
    print("\n" + "=" * 60)
    print("Analyzing history length...")
    print("=" * 60)

    train_hist_lens = [len(s['history']) for s in train_dataset.samples]
    valid_hist_lens = [len(s['history']) for s in valid_dataset.samples]
    test_hist_lens = [len(s['history']) for s in test_dataset.samples]

    print(f"\nHistory length (in items, before truncation):")
    print(f"  Train - mean: {sum(train_hist_lens)/len(train_hist_lens):.2f}, "
          f"min: {min(train_hist_lens)}, max: {max(train_hist_lens)}")
    print(f"  Valid - mean: {sum(valid_hist_lens)/len(valid_hist_lens):.2f}, "
          f"min: {min(valid_hist_lens)}, max: {max(valid_hist_lens)}")
    print(f"  Test  - mean: {sum(test_hist_lens)/len(test_hist_lens):.2f}, "
          f"min: {min(test_hist_lens)}, max: {max(test_hist_lens)}")

    print("\n" + "=" * 60)
    print("Diagnosis complete!")
    print("=" * 60)


if __name__ == "__main__":
    diagnose()
