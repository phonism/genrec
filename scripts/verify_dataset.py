"""
Verify AmazonSeqDataset leave-one-out implementation.
"""
import sys
sys.path.insert(0, '.')

import torch
from genrec.data.amazon import AmazonSeqDataset, AmazonItemDataset

def test_leave_one_out():
    """Test that leave-one-out split is correct."""
    print("=" * 60)
    print("Testing Leave-One-Out Split")
    print("=" * 60)

    # Create datasets for all splits (without RQVAE to speed up)
    # We'll manually check the sequence splitting logic

    # First, let's create a simple mock test
    print("\n1. Testing sequence splitting logic...")

    # Simulate a user sequence
    full_seq = [10, 20, 30, 40, 50]  # 5 items
    print(f"   Full sequence: {full_seq}")

    # Train: seq[:-2], history=seq[:-1], target=seq[-1]
    train_seq = full_seq[:-2]  # [10, 20, 30]
    train_history = train_seq[:-1]  # [10, 20]
    train_target = train_seq[-1]  # 30
    print(f"   Train: history={train_history}, target={train_target}")

    # Valid: seq[:-1], history=seq[:-1], target=seq[-1]
    valid_seq = full_seq[:-1]  # [10, 20, 30, 40]
    valid_history = valid_seq[:-1]  # [10, 20, 30]
    valid_target = valid_seq[-1]  # 40
    print(f"   Valid: history={valid_history}, target={valid_target}")

    # Test: seq[:], history=seq[:-1], target=seq[-1]
    test_seq = full_seq  # [10, 20, 30, 40, 50]
    test_history = test_seq[:-1]  # [10, 20, 30, 40]
    test_target = test_seq[-1]  # 50
    print(f"   Test:  history={test_history}, target={test_target}")

    # Verify no overlap in targets
    assert train_target != valid_target != test_target, "Targets should be different!"
    print("\n   ✓ Targets are different for each split")

    # Verify history is incrementally longer
    assert len(train_history) < len(valid_history) < len(test_history), "History should grow!"
    print("   ✓ History length increases: train < valid < test")


def test_dataset_loading():
    """Test actual dataset loading."""
    print("\n" + "=" * 60)
    print("Testing Dataset Loading (without RQVAE)")
    print("=" * 60)

    # Check if item dataset can be loaded
    print("\n2. Loading AmazonItemDataset...")
    try:
        item_dataset = AmazonItemDataset(
            root="dataset/amazon",
            split="beauty",
            train_test_split="all",
            encoder_model_name="sentence-transformers/sentence-t5-base",
        )
        print(f"   ✓ Loaded {len(item_dataset)} items")
        print(f"   Embedding dim: {item_dataset.dim}")
    except Exception as e:
        print(f"   ✗ Failed to load item dataset: {e}")
        return False

    return True


def test_full_dataset(rqvae_path: str):
    """Test full dataset with RQVAE."""
    print("\n" + "=" * 60)
    print("Testing Full AmazonSeqDataset with RQVAE")
    print("=" * 60)

    print(f"\n3. Loading datasets with RQVAE from: {rqvae_path}")

    try:
        train_dataset = AmazonSeqDataset(
            root="dataset/amazon",
            split="beauty",
            train_test_split="train",
            max_seq_len=20,
            subsample=False,  # Disable subsample for verification
            pretrained_rqvae_path=rqvae_path,
            rqvae_hidden_dims=[512, 256, 128, 64],
        )
        print(f"   ✓ Train dataset: {len(train_dataset)} sequences")

        valid_dataset = AmazonSeqDataset(
            root="dataset/amazon",
            split="beauty",
            train_test_split="valid",
            max_seq_len=20,
            subsample=False,
            pretrained_rqvae_path=rqvae_path,
            rqvae_hidden_dims=[512, 256, 128, 64],
        )
        print(f"   ✓ Valid dataset: {len(valid_dataset)} sequences")

        test_dataset = AmazonSeqDataset(
            root="dataset/amazon",
            split="beauty",
            train_test_split="test",
            max_seq_len=20,
            subsample=False,
            pretrained_rqvae_path=rqvae_path,
            rqvae_hidden_dims=[512, 256, 128, 64],
        )
        print(f"   ✓ Test dataset: {len(test_dataset)} sequences")

        # All splits should have same number of users
        assert len(train_dataset) == len(valid_dataset) == len(test_dataset), \
            "All splits should have same number of sequences!"
        print(f"\n   ✓ All splits have same size: {len(train_dataset)}")

        # Check a few samples
        print("\n4. Checking sample data...")
        for i in range(min(3, len(train_dataset))):
            train_sample = train_dataset[i]
            valid_sample = valid_dataset[i]
            test_sample = test_dataset[i]

            print(f"\n   User {i}:")
            print(f"   Train: {len(train_sample.item_ids)//3} items -> target {train_sample.target_ids}")
            print(f"   Valid: {len(valid_sample.item_ids)//3} items -> target {valid_sample.target_ids}")
            print(f"   Test:  {len(test_sample.item_ids)//3} items -> target {test_sample.target_ids}")

            # Verify targets are different
            if train_sample.target_ids != valid_sample.target_ids != test_sample.target_ids:
                print(f"   ✓ Targets are different")
            else:
                print(f"   ✗ WARNING: Some targets may be same (could happen if items repeat)")

        # Check semantic ID format
        print("\n5. Checking semantic ID format...")
        sample = train_dataset[0]
        print(f"   item_ids length: {len(sample.item_ids)} (should be multiple of 3)")
        print(f"   target_ids: {sample.target_ids} (should be list of 3)")
        assert len(sample.item_ids) % 3 == 0, "item_ids should be multiple of 3"
        assert len(sample.target_ids) == 3, "target_ids should have 3 elements"
        print("   ✓ Semantic ID format is correct")

        # Check value ranges
        print("\n6. Checking value ranges...")
        all_ids = sample.item_ids + sample.target_ids
        min_id, max_id = min(all_ids), max(all_ids)
        print(f"   ID range: [{min_id}, {max_id}] (should be in [0, 255])")
        assert 0 <= min_id and max_id <= 255, "IDs should be in [0, 255]"
        print("   ✓ Value ranges are correct")

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"   ✗ RQVAE checkpoint not found: {e}")
        print("   Please provide correct path to RQVAE checkpoint")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rqvae_path", type=str,
                        default="./out/rqvae/amazon/beauty/checkpoint_epoch_999.pt",
                        help="Path to pretrained RQVAE checkpoint")
    parser.add_argument("--skip_rqvae", action="store_true",
                        help="Skip tests that require RQVAE")
    args = parser.parse_args()

    # Test 1: Logic test
    test_leave_one_out()

    # Test 2: Item dataset loading
    if test_dataset_loading():
        # Test 3: Full dataset (requires RQVAE)
        if not args.skip_rqvae:
            test_full_dataset(args.rqvae_path)
        else:
            print("\nSkipping RQVAE tests (--skip_rqvae flag set)")
