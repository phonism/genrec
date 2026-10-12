import math

import pytest
import torch

from genrec.data.amazon import (
    AmazonSeqDataset,
    semantic_id_layer_usage,
    stable_hash_bucket,
)
from genrec.data.schemas import SeqData
from genrec.trainers.rqvae_trainer import (
    health_selection_key,
    summarize_semantic_ids,
)
from genrec.trainers.tiger_trainer import inverse_sqrt_lr_lambda, t5_collate


def test_tiger_collate_prepends_disjoint_user_token_and_right_pads():
    sample = SeqData(
        user_id=7,
        item_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        target_ids=[9, 10, 11, 12],
        target_item=99,
    )
    batch = t5_collate(
        [sample],
        codebook_size=256,
        sem_id_dim=4,
        max_history_tokens=12,
        include_user_token=True,
        num_user_tokens=2000,
        padding_side="right",
    )

    assert batch["input_ids"][0].tolist() == [
        1032,  # 4 * 256 + 1 + user bucket 7
        2, 259, 516, 773,
        6, 263, 520, 777,
        0, 0, 0, 0,
    ]
    assert batch["attention_mask"][0].tolist() == [1] * 9 + [0] * 4
    assert batch["labels"][0].tolist() == [10, 267, 524, 781]
    assert batch["target_items"].tolist() == [99]


def test_tiger_collate_rejects_sid_width_misalignment():
    sample = SeqData(user_id=0, item_ids=[1, 2, 3], target_ids=[1, 2, 3, 4])
    with pytest.raises(ValueError, match="divisible by sem_id_dim"):
        t5_collate([sample], sem_id_dim=4)


def test_inverse_sqrt_schedule_matches_published_shape():
    assert inverse_sqrt_lr_lambda(1, 10000) == 1.0
    assert inverse_sqrt_lr_lambda(10000, 10000) == 1.0
    assert inverse_sqrt_lr_lambda(40000, 10000) == 0.5
    assert math.isclose(
        inverse_sqrt_lr_lambda(200000, 10000), math.sqrt(0.05)
    )


def test_user_hash_is_stable_and_bounded():
    first = stable_hash_bucket("A2EXAMPLE", 2000)
    assert first == stable_hash_bucket("A2EXAMPLE", 2000)
    assert 0 <= first < 2000
    with pytest.raises(ValueError, match="positive"):
        stable_hash_bucket("A2EXAMPLE", 0)


def test_semantic_id_layer_usage_detects_collapsed_upstream_tokenizer():
    usage = semantic_id_layer_usage(
        torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3]]),
        codebook_size=4,
    )
    assert usage == [0.25, 1.0]


def test_collision_suffix_makes_catalog_ids_unique():
    dataset = AmazonSeqDataset.__new__(AmazonSeqDataset)
    dataset.rqvae_codebook_size = 256
    resolved = dataset._add_disambiguation_suffix(
        [[12, 24, 52], [12, 24, 52], [7, 8, 9]]
    )

    assert resolved == [[12, 24, 52, 0], [12, 24, 52, 1], [7, 8, 9, 0]]
    assert len({tuple(sid) for sid in resolved}) == 3


def test_rqvae_health_reports_layer_usage_and_prefers_paper_healthy_state():
    healthy = summarize_semantic_ids(
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 3]], codebook_size=5
    )
    collapsed = summarize_semantic_ids(
        [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]], codebook_size=5
    )

    assert healthy["layer_usage"] == [1.0, 0.8]
    assert collapsed["layer_usage"] == [0.2, 1.0]
    # A checkpoint satisfying the paper's usage gate must beat a lower-
    # collision checkpoint that has a collapsed layer.
    assert health_selection_key(healthy, 0.8) > health_selection_key(
        collapsed, 0.8
    )
