"""
Analyze the sliding window training data distribution.
"""
import sys
sys.path.insert(0, '.')

from collections import Counter
from genrec.data.amazon import AmazonSeqDataset

def analyze():
    print("=" * 60)
    print("Loading train dataset...")
    print("=" * 60)

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

    print(f"\nTotal train samples: {len(train_dataset.samples)}")
    print(f"Total valid samples: {len(valid_dataset.samples)}")
    print(f"Total users: {len(train_dataset.sequences)}")

    # Analyze history length distribution
    print("\n" + "=" * 60)
    print("History length distribution in TRAINING samples:")
    print("=" * 60)

    train_hist_lens = [len(s['history']) for s in train_dataset.samples]
    hist_len_counts = Counter(train_hist_lens)

    print(f"Min history length: {min(train_hist_lens)}")
    print(f"Max history length: {max(train_hist_lens)}")
    print(f"Mean history length: {sum(train_hist_lens)/len(train_hist_lens):.2f}")

    print("\nHistory length distribution:")
    for length in sorted(hist_len_counts.keys())[:15]:
        count = hist_len_counts[length]
        pct = 100 * count / len(train_hist_lens)
        print(f"  Length {length}: {count:6d} samples ({pct:5.2f}%)")

    # What percentage has very short history?
    short_history = sum(1 for l in train_hist_lens if l <= 3)
    print(f"\nSamples with history <= 3: {short_history}/{len(train_hist_lens)} ({100*short_history/len(train_hist_lens):.1f}%)")

    # Analyze valid history length
    print("\n" + "=" * 60)
    print("History length distribution in VALIDATION samples:")
    print("=" * 60)

    valid_hist_lens = [len(s['history']) for s in valid_dataset.samples]
    valid_hist_len_counts = Counter(valid_hist_lens)

    print(f"Min history length: {min(valid_hist_lens)}")
    print(f"Max history length: {max(valid_hist_lens)}")
    print(f"Mean history length: {sum(valid_hist_lens)/len(valid_hist_lens):.2f}")

    print("\nHistory length distribution:")
    for length in sorted(valid_hist_len_counts.keys())[:15]:
        count = valid_hist_len_counts[length]
        pct = 100 * count / len(valid_hist_lens)
        print(f"  Length {length}: {count:6d} samples ({pct:5.2f}%)")

    # Key difference
    print("\n" + "=" * 60)
    print("KEY OBSERVATION:")
    print("=" * 60)

    train_short = sum(1 for l in train_hist_lens if l <= 3) / len(train_hist_lens)
    valid_short = sum(1 for l in valid_hist_lens if l <= 3) / len(valid_hist_lens)

    print(f"Training has {100*train_short:.1f}% samples with short history (<=3)")
    print(f"Validation has {100*valid_short:.1f}% samples with short history (<=3)")

    if train_short > valid_short + 0.1:
        print("\n⚠️  WARNING: Training data is dominated by SHORT histories!")
        print("The model may overfit to short-history patterns and struggle with longer histories in validation.")

    # Check if early samples in sliding window are too easy/trivial
    print("\n" + "=" * 60)
    print("Analyzing per-user sample contribution:")
    print("=" * 60)

    samples_per_user = Counter(s['user_id'] for s in train_dataset.samples)
    sample_counts = list(samples_per_user.values())

    print(f"Min samples per user: {min(sample_counts)}")
    print(f"Max samples per user: {max(sample_counts)}")
    print(f"Mean samples per user: {sum(sample_counts)/len(sample_counts):.2f}")

    # Users with many samples dominate training
    heavy_users = sum(1 for c in sample_counts if c > 10)
    total_samples_from_heavy = sum(c for c in sample_counts if c > 10)
    print(f"\nUsers with >10 samples: {heavy_users}")
    print(f"Total samples from these users: {total_samples_from_heavy} ({100*total_samples_from_heavy/len(train_dataset.samples):.1f}%)")

if __name__ == "__main__":
    analyze()
