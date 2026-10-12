import pytest

from genrec.data.evaluation_protocol import select_protocol_sequences


USERS = {
    "u1": [(1, 10), (2, 11), (3, 12), (8, 13), (10, 14)],
    "u2": [(4, 20), (5, 21), (6, 22), (9, 23), (11, 24)],
}


def _items(sequence):
    return [item_id for _, item_id in sequence]


def test_leave_one_out_keeps_sorted_sequences():
    selected, info = select_protocol_sequences(USERS, "train")
    assert info.protocol == "leave_one_out"
    assert _items(selected[0][1]) == [10, 11, 12, 13, 14]


def test_global_temporal_targets_are_after_global_cutoffs():
    train, train_info = select_protocol_sequences(
        USERS, "train", protocol="global_temporal", temporal_train_ratio=0.5,
        temporal_valid_ratio=0.2,
    )
    valid, valid_info = select_protocol_sequences(
        USERS, "valid", protocol="global_temporal", temporal_train_ratio=0.5,
        temporal_valid_ratio=0.2,
    )
    test, test_info = select_protocol_sequences(
        USERS, "test", protocol="global_temporal", temporal_train_ratio=0.5,
        temporal_valid_ratio=0.2,
    )

    assert train_info.train_cutoff == valid_info.train_cutoff == test_info.train_cutoff
    assert train_info.valid_cutoff == valid_info.valid_cutoff == test_info.valid_cutoff

    # Remove the two placeholders consumed by existing train builders.
    for _, sequence in train:
        assert all(ts <= train_info.train_cutoff for ts, _ in sequence[:-2])
    # Existing valid/test builders see the penultimate/final interaction as target.
    for _, sequence in valid:
        assert train_info.train_cutoff < sequence[-2][0] <= train_info.valid_cutoff
    for _, sequence in test:
        assert sequence[-1][0] > train_info.valid_cutoff


def test_invalid_temporal_ratios_are_rejected():
    with pytest.raises(ValueError):
        select_protocol_sequences(
            USERS, "train", protocol="global_temporal",
            temporal_train_ratio=0.9, temporal_valid_ratio=0.1,
        )
