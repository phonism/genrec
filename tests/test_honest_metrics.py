import torch

from genrec.modules.metrics import SemanticIdItemMapper, TopKAccumulator


def test_mapper_uses_training_popularity_and_deterministic_ties():
    semantic_ids = [
        [1, 2],  # collision bucket A
        [1, 2],
        [3, 4],  # collision bucket B, tied popularity
        [3, 4],
    ]
    mapper = SemanticIdItemMapper(
        semantic_ids,
        item_popularity={0: 2, 1: 5, 2: 3, 3: 3},
    )

    assert mapper.sid_to_item[(1, 2)] == 1
    assert mapper.sid_to_item[(3, 4)] == 2
    assert mapper.collision_rate == 0.5
    assert mapper.num_collision_items == 4


def test_from_sequences_excludes_validation_and_test_targets():
    semantic_ids = [[7], [7]]
    # Item 0 is more popular in the training prefixes.  Item 1 appears only in
    # the two held-out positions and must not influence disambiguation.
    sequences = [[0, 0, 1, 1], [0, 1, 1]]

    mapper = SemanticIdItemMapper.from_sequences(
        semantic_ids, sequences, train_holdout=2
    )

    assert mapper.sid_to_item[(7,)] == 0


def test_map_predictions_deduplicates_items_and_pads():
    mapper = SemanticIdItemMapper(
        [[1, 1], [1, 1], [2, 2]],
        item_popularity={0: 1, 1: 3, 2: 1},
    )
    predictions = torch.tensor(
        [
            [[1, 1], [1, 1], [2, 2], [9, 9]],
            [[9, 9], [2, 2], [1, 1], [1, 1]],
        ]
    )

    mapped = mapper.map_predictions(predictions)

    assert mapped.tolist() == [[1, 2, -1, -1], [2, 1, -1, -1]]


def test_item_level_metric_does_not_credit_non_representative_collision_item():
    mapper = SemanticIdItemMapper(
        [[5, 6], [5, 6]],
        item_popularity={0: 10, 1: 1},
    )
    predicted_sids = torch.tensor([[[5, 6]]])
    predicted_items = mapper.map_predictions(predicted_sids)

    sid_metric = TopKAccumulator(ks=[1])
    sid_metric.accumulate(torch.tensor([[5, 6]]), predicted_sids)
    honest_metric = TopKAccumulator(ks=[1])
    honest_metric.accumulate(torch.tensor([[1]]), predicted_items.unsqueeze(-1))

    assert sid_metric.reduce()["Recall@1"] == 1.0
    assert honest_metric.reduce()["Recall@1"] == 0.0


def test_item_dedup_can_pull_a_deeper_beam_into_top_k():
    mapper = SemanticIdItemMapper(
        [[1], [2], [3]], item_popularity={0: 1, 1: 1, 2: 1}
    )
    # The target SID is third in the raw list, but a duplicate item occupies
    # raw rank two. After item dedup, the target becomes the second deployable
    # recommendation. Hence SID@2 and item@2 are intentionally not comparable.
    predicted_sids = torch.tensor([[[1], [1], [3]]])

    sid_metric = TopKAccumulator(ks=[2])
    sid_metric.accumulate(torch.tensor([[3]]), predicted_sids)
    item_metric = TopKAccumulator(ks=[2])
    item_metric.accumulate(
        torch.tensor([[2]]), mapper.map_predictions(predicted_sids).unsqueeze(-1)
    )

    assert sid_metric.reduce()["Recall@2"] == 0.0
    assert item_metric.reduce()["Recall@2"] == 1.0


def test_requested_k_larger_than_candidates_does_not_turn_a_miss_into_a_hit():
    accumulator = TopKAccumulator(ks=[5, 10])
    actual = torch.tensor([[99]])
    candidates = torch.tensor([[[1], [2], [3], [4], [5]]])

    accumulator.accumulate(actual, candidates)
    metrics = accumulator.reduce()

    assert metrics["Recall@5"] == 0.0
    assert metrics["Recall@10"] == 0.0
