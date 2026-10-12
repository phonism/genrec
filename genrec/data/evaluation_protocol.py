"""Shared sequence-splitting protocols for recommendation evaluation."""

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple


TimedInteraction = Tuple[int, int]  # (timestamp, item_id)
TimedUserSequence = Tuple[str, List[TimedInteraction]]


@dataclass(frozen=True)
class ProtocolInfo:
    protocol: str
    train_cutoff: int = 0
    valid_cutoff: int = 0
    users_before: int = 0
    users_after: int = 0


def _quantile_cutoff(sorted_values: Sequence[int], fraction: float) -> int:
    if not sorted_values:
        raise ValueError("cannot compute a temporal cutoff from no interactions")
    index = max(0, min(len(sorted_values) - 1, math.ceil(len(sorted_values) * fraction) - 1))
    return int(sorted_values[index])


def select_protocol_sequences(
    user_sequences: Dict[str, List[TimedInteraction]],
    split: str,
    protocol: str = "leave_one_out",
    min_seq_len: int = 5,
    temporal_train_ratio: float = 0.8,
    temporal_valid_ratio: float = 0.1,
) -> Tuple[List[TimedUserSequence], ProtocolInfo]:
    """Return sequences shaped for the repository's existing sample builders.

    ``leave_one_out`` returns the sorted sequences unchanged.  For
    ``global_temporal``, global timestamp cutoffs are computed after the same
    user-length filtering used by the model datasets:

    - train: all interactions through the train cutoff;
    - valid: the first interaction in the validation window;
    - test: the first interaction after the validation cutoff.

    The returned sequences contain small trailing placeholders so existing LOO
    builders (which remove the last two/one elements) consume the desired
    prefix and target without duplicating protocol logic across every model.
    Placeholders are copies of an existing interaction and are always removed
    before a sample is emitted.
    """
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"unknown data split: {split}")
    if protocol not in {"leave_one_out", "global_temporal"}:
        raise ValueError(
            f"unknown evaluation protocol: {protocol}; use leave_one_out or global_temporal"
        )
    if not 0 < temporal_train_ratio < 1:
        raise ValueError("temporal_train_ratio must be between 0 and 1")
    if not 0 < temporal_valid_ratio < 1:
        raise ValueError("temporal_valid_ratio must be between 0 and 1")
    if temporal_train_ratio + temporal_valid_ratio >= 1:
        raise ValueError("train + valid temporal ratios must leave a non-empty test fraction")

    filtered: List[TimedUserSequence] = []
    for user_id, interactions in user_sequences.items():
        ordered = sorted(interactions, key=lambda x: x[0])
        if len(ordered) >= min_seq_len:
            filtered.append((user_id, ordered))

    if protocol == "leave_one_out":
        return filtered, ProtocolInfo(
            protocol=protocol,
            users_before=len(filtered),
            users_after=len(filtered),
        )

    timestamps = sorted(ts for _, seq in filtered for ts, _ in seq)
    train_cutoff = _quantile_cutoff(timestamps, temporal_train_ratio)
    valid_cutoff = _quantile_cutoff(
        timestamps, temporal_train_ratio + temporal_valid_ratio
    )
    if valid_cutoff <= train_cutoff:
        raise ValueError(
            "global_temporal cutoffs collapse to the same timestamp; use data "
            "with finer timestamps or different split ratios"
        )
    if timestamps[-1] <= valid_cutoff:
        raise ValueError("global_temporal split produced no test-time interactions")

    selected: List[TimedUserSequence] = []
    for user_id, interactions in filtered:
        if split == "train":
            prefix = [event for event in interactions if event[0] <= train_cutoff]
            if prefix:
                # Existing train builders strip two held-out positions.
                selected.append((user_id, prefix + [prefix[-1], prefix[-1]]))
            continue

        if split == "valid":
            target_index = next(
                (
                    idx for idx, (ts, _) in enumerate(interactions)
                    if train_cutoff < ts <= valid_cutoff
                ),
                None,
            )
            if target_index is not None and target_index > 0:
                target = interactions[target_index]
                # Existing valid builders strip one test placeholder.
                selected.append((user_id, interactions[:target_index] + [target, target]))
            continue

        target_index = next(
            (idx for idx, (ts, _) in enumerate(interactions) if ts > valid_cutoff),
            None,
        )
        if target_index is not None and target_index > 0:
            selected.append((user_id, interactions[:target_index + 1]))

    if not selected:
        raise ValueError(
            f"global_temporal produced no usable {split} user sequences"
        )

    return selected, ProtocolInfo(
        protocol=protocol,
        train_cutoff=train_cutoff,
        valid_cutoff=valid_cutoff,
        users_before=len(filtered),
        users_after=len(selected),
    )
