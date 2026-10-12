"""
Compare our data generation with reference implementation.
Check for potential data leakage or implementation bugs.
"""
import sys
sys.path.insert(0, '.')

from genrec.data.amazon import AmazonSeqDataset


def simulate_reference_data_generation(sequences, max_len=20):
    """
    Simulate what reference implementation would generate.

    Reference code (from reference/TIGER/model/dataset.py):
    ```
    if mode == 'train':
        for i in range(1, len(sequence)):
            processed_data.append({
                'history': sequence[:i],
                'target': sequence[i]
            })
    elif mode == 'evaluation':
        processed_data.append({
            'history': sequence[:-1],
            'target': sequence[-1]
        })
    ```

    Key point: Reference uses SEPARATE parquet files:
    - train.parquet: sequence = original_seq (excluding last 2 for leave-one-out)
    - valid.parquet: sequence = original_seq[:-1] (excluding test target)
    - test.parquet:  sequence = original_seq (full)
    """
    train_samples = []
    valid_samples = []
    test_samples = []

    for user_idx, full_seq in enumerate(sequences):
        # Train: use full_seq[:-2] (exclude valid and test targets)
        train_seq = full_seq[:-2]
        for i in range(1, len(train_seq)):
            history = train_seq[:i]
            # Truncate to max_len (take last max_len items)
            if len(history) > max_len:
                history = history[-max_len:]
            train_samples.append({
                'user_idx': user_idx,
                'history': history,
                'target': train_seq[i]
            })

        # Valid: use full_seq[:-1], target = last item = full_seq[-2]
        valid_seq = full_seq[:-1]
        valid_history = valid_seq[:-1]  # = full_seq[:-2]
        if len(valid_history) > max_len:
            valid_history = valid_history[-max_len:]
        valid_samples.append({
            'user_idx': user_idx,
            'history': valid_history,
            'target': valid_seq[-1]  # = full_seq[-2]
        })

        # Test: use full_seq, target = last item = full_seq[-1]
        test_history = full_seq[:-1]
        if len(test_history) > max_len:
            test_history = test_history[-max_len:]
        test_samples.append({
            'user_idx': user_idx,
            'history': test_history,
            'target': full_seq[-1]
        })

    return train_samples, valid_samples, test_samples

def compare():
    print("=" * 70)
    print("Loading dataset...")
    print("=" * 70)

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

    print("\n" + "=" * 70)
    print("Checking a few user sequences in detail...")
    print("=" * 70)

    # Check first 5 users - use direct index instead of hash
    for user_idx in range(5):
        full_seq = train_dataset.sequences[user_idx]
        user_id_str = train_dataset.user_ids[user_idx]

        print(f"\n--- User {user_idx} (ID: {user_id_str[:20]}...) ---")
        print(f"Full sequence length: {len(full_seq)}")
        print(f"Full sequence: {full_seq[:10]}{'...' if len(full_seq) > 10 else ''}")

        # What reference would do:
        ref_train_seq = full_seq[:-2]
        ref_valid_target = full_seq[-2]
        ref_test_target = full_seq[-1]

        print(f"\nExpected (reference-style):")
        print(f"  Train seq: {ref_train_seq} (length={len(ref_train_seq)})")
        print(f"  Train targets: {ref_train_seq[1:]} (sliding window)")
        print(f"  Valid: history={full_seq[:-2]}, target={ref_valid_target}")
        print(f"  Test:  history={full_seq[:-1]}, target={ref_test_target}")

        # Our actual samples for this user (by index)
        # Count samples that came from this user_idx
        train_start_idx = sum(len(train_dataset.sequences[i][:-2]) - 1 for i in range(user_idx) if len(train_dataset.sequences[i][:-2]) > 1)
        train_count = max(0, len(full_seq[:-2]) - 1)

        print(f"\nOur actual samples:")
        print(f"  Train samples count: {train_count}")
        if train_count > 0:
            actual_train_targets = full_seq[1:-2]
            print(f"  Train targets: {list(actual_train_targets)}")

        # Valid/test are 1 per user, so use user_idx directly
        valid_sample = valid_dataset.samples[user_idx]
        test_sample = test_dataset.samples[user_idx]

        print(f"  Valid: history_len={len(valid_sample['history'])}, target={valid_sample['target']}")
        print(f"  Test:  history_len={len(test_sample['history'])}, target={test_sample['target']}")

        # Verify correctness
        expected_valid_history = full_seq[:-2]
        expected_valid_target = full_seq[-2]
        expected_test_history = full_seq[:-1]
        expected_test_target = full_seq[-1]

        valid_correct = (valid_sample['history'] == expected_valid_history and
                        valid_sample['target'] == expected_valid_target)
        test_correct = (test_sample['history'] == expected_test_history and
                       test_sample['target'] == expected_test_target)

        print(f"\n  VERIFICATION:")
        print(f"    Valid correct? {valid_correct}")
        print(f"    Test correct? {test_correct}")

        # Check leakage for this specific user
        train_targets_set = set(full_seq[1:-2])
        print(f"    Valid target {valid_sample['target']} in this user's train targets? {valid_sample['target'] in train_targets_set}")
        print(f"    Test target {test_sample['target']} in this user's train targets? {test_sample['target'] in train_targets_set}")

    print("\n" + "=" * 70)
    print("Comparing with simulated reference implementation...")
    print("=" * 70)

    # Generate what reference would produce
    ref_train, ref_valid, ref_test = simulate_reference_data_generation(
        train_dataset.sequences, max_len=20
    )

    print(f"\nSample counts:")
    print(f"  Reference train: {len(ref_train)}")
    print(f"  Our train:       {len(train_dataset.samples)}")
    print(f"  Match: {len(ref_train) == len(train_dataset.samples)}")

    print(f"\n  Reference valid: {len(ref_valid)}")
    print(f"  Our valid:       {len(valid_dataset.samples)}")
    print(f"  Match: {len(ref_valid) == len(valid_dataset.samples)}")

    print(f"\n  Reference test:  {len(ref_test)}")
    print(f"  Our test:        {len(test_dataset.samples)}")
    print(f"  Match: {len(ref_test) == len(test_dataset.samples)}")

    # Compare samples for first few users
    print("\nComparing samples for first 3 users:")
    for user_idx in range(3):
        print(f"\n  User {user_idx}:")

        # Reference valid
        rv = ref_valid[user_idx]
        # Our valid
        ov = valid_dataset.samples[user_idx]

        print(f"    Valid - Reference: history_len={len(rv['history'])}, target={rv['target']}")
        print(f"    Valid - Ours:      history_len={len(ov['history'])}, target={ov['target']}")
        print(f"    Valid match: {rv['history'] == ov['history'] and rv['target'] == ov['target']}")

        # Reference test
        rt = ref_test[user_idx]
        # Our test
        ot = test_dataset.samples[user_idx]

        print(f"    Test - Reference:  history_len={len(rt['history'])}, target={rt['target']}")
        print(f"    Test - Ours:       history_len={len(ot['history'])}, target={ot['target']}")
        print(f"    Test match: {rt['history'] == ot['history'] and rt['target'] == ot['target']}")

    # Check all valid/test samples match
    valid_mismatches = 0
    test_mismatches = 0

    for i in range(len(ref_valid)):
        rv = ref_valid[i]
        ov = valid_dataset.samples[i]
        if rv['history'] != ov['history'] or rv['target'] != ov['target']:
            valid_mismatches += 1

    for i in range(len(ref_test)):
        rt = ref_test[i]
        ot = test_dataset.samples[i]
        if rt['history'] != ot['history'] or rt['target'] != ot['target']:
            test_mismatches += 1

    print(f"\nOverall comparison:")
    print(f"  Valid mismatches: {valid_mismatches}/{len(ref_valid)}")
    print(f"  Test mismatches:  {test_mismatches}/{len(ref_test)}")

    if valid_mismatches == 0 and test_mismatches == 0:
        print("  ✅ All valid/test samples match reference!")
    else:
        print("  ⚠️ Some samples don't match reference!")

    print("\n" + "=" * 70)
    print("Global leakage check...")
    print("=" * 70)

    # Check for REAL leakage using actual user index (not hashed user_id)
    # For each user, check if their valid/test target appears in their train targets
    print("\nChecking per-user leakage (using actual user index, not hash)...")

    valid_leaks = 0
    test_leaks = 0

    for user_idx in range(len(train_dataset.sequences)):
        full_seq = train_dataset.sequences[user_idx]

        # Train targets for this user: full_seq[1:-2]
        train_targets = set(full_seq[1:-2])

        # Valid target: full_seq[-2]
        valid_target = full_seq[-2]

        # Test target: full_seq[-1]
        test_target = full_seq[-1]

        if valid_target in train_targets:
            valid_leaks += 1
            if valid_leaks <= 5:
                print(f"  Valid leak user {user_idx}: target {valid_target} in train_targets {list(train_targets)[:10]}")

        if test_target in train_targets:
            test_leaks += 1
            if test_leaks <= 5:
                print(f"  Test leak user {user_idx}: target {test_target} in train_targets {list(train_targets)[:10]}")

    print(f"\nPer-user valid leakage: {valid_leaks}/{len(train_dataset.sequences)} ({100*valid_leaks/len(train_dataset.sequences):.2f}%)")
    print(f"Per-user test leakage: {test_leaks}/{len(test_dataset.sequences)} ({100*test_leaks/len(test_dataset.sequences):.2f}%)")

    if valid_leaks > 0 or test_leaks > 0:
        print("\n⚠️  WARNING: REAL DATA LEAKAGE DETECTED!")
        print("This means some users have their valid/test target appearing earlier in their sequence.")
    else:
        print("\n✅ No per-user data leakage detected")

    print("\n" + "=" * 70)
    print("Checking valid/test history construction...")
    print("=" * 70)

    # The key issue might be: does valid/test history include items that are used as training targets?
    # This is EXPECTED and OK - we just can't have the SAME user's valid/test target in their train targets

    # But let's check something else: is the valid history correct?
    # Valid: history should be full_seq[:-2], target should be full_seq[-2]

    errors = 0
    for user_idx in range(min(100, len(valid_dataset.sequences))):
        full_seq = valid_dataset.sequences[user_idx]
        user_id_str = valid_dataset.user_ids[user_idx]
        user_hash = hash(user_id_str) % 10000

        # Expected valid
        expected_history = full_seq[:-2]
        expected_target = full_seq[-2]

        # Actual valid
        user_valid_samples = [s for s in valid_dataset.samples if s['user_id'] == user_hash]
        if user_valid_samples:
            actual_history = user_valid_samples[0]['history']
            actual_target = user_valid_samples[0]['target']

            if actual_target != expected_target:
                print(f"ERROR User {user_idx}: expected target {expected_target}, got {actual_target}")
                errors += 1

            if actual_history != expected_history:
                print(f"ERROR User {user_idx}: history mismatch")
                print(f"  Expected: {expected_history[-5:]}")
                print(f"  Actual:   {actual_history[-5:]}")
                errors += 1

    if errors == 0:
        print("✅ Valid history/target construction looks correct (checked 100 users)")
    else:
        print(f"⚠️  Found {errors} errors in valid construction")

    # Same check for test
    errors = 0
    for user_idx in range(min(100, len(test_dataset.sequences))):
        full_seq = test_dataset.sequences[user_idx]
        user_id_str = test_dataset.user_ids[user_idx]
        user_hash = hash(user_id_str) % 10000

        # Expected test
        expected_history = full_seq[:-1]
        expected_target = full_seq[-1]

        # Actual test
        user_test_samples = [s for s in test_dataset.samples if s['user_id'] == user_hash]
        if user_test_samples:
            actual_history = user_test_samples[0]['history']
            actual_target = user_test_samples[0]['target']

            if actual_target != expected_target:
                print(f"ERROR User {user_idx}: expected target {expected_target}, got {actual_target}")
                errors += 1

            if actual_history != expected_history:
                print(f"ERROR User {user_idx}: history mismatch")
                errors += 1

    if errors == 0:
        print("✅ Test history/target construction looks correct (checked 100 users)")
    else:
        print(f"⚠️  Found {errors} errors in test construction")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"""
Our implementation:
  - Train: sliding window on full_seq[:-2], generates {len(train_dataset.samples)} samples
  - Valid: history=full_seq[:-2], target=full_seq[-2]
  - Test:  history=full_seq[:-1], target=full_seq[-1]

Reference implementation:
  - Train: sliding window on (history + target from train.parquet)
  - Valid: history=seq[:-1], target=seq[-1] from valid.parquet
  - Test:  history=seq[:-1], target=seq[-1] from test.parquet

Key difference:
  - Reference uses SEPARATE parquet files for train/valid/test
  - Each parquet file has its own 'history' and 'target' columns
  - The parquet files are pre-split, we don't know exactly how

Potential issue:
  - If reference's valid.parquet contains sequence = full_seq[:-1],
    then valid history = full_seq[:-2], target = full_seq[-2] (SAME as ours)
  - If reference's test.parquet contains sequence = full_seq,
    then test history = full_seq[:-1], target = full_seq[-1] (SAME as ours)

  This should be CORRECT. But let's verify the actual numbers match paper.
""")


if __name__ == "__main__":
    compare()
