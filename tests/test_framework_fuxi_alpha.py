"""Causality, target/candidate contracts for the padded FuXi-alpha port."""
import csv
from types import SimpleNamespace

import pytest
import torch

from genrec.data.sequential_csv import SequentialCSVDataset
from genrec.models.fuxi_alpha import FuXiAlpha
from genrec.trainers.fuxi_alpha_trainer import evaluate


def _csv(path):
    with path.open("w") as f:
        writer = csv.DictWriter(f, ["user_id", "sequence_item_ids", "sequence_timestamps"])
        writer.writeheader()
        writer.writerow({"user_id": 1, "sequence_item_ids": "1,2,3,4,5,6",
                         "sequence_timestamps": "10,20,30,40,50,60"})


def test_csv_native_train_omits_test_and_keeps_shifted_targets(tmp_path):
    path = tmp_path / "users.csv"; _csv(path)
    ds = SequentialCSVDataset(path, "train", 4, 2)
    row = ds[0]
    assert row["input_ids"].tolist() == [1, 2, 3, 4, 0, 0]
    assert row["targets"].tolist() == [2, 3, 4, 5, 0, 0]
    assert row["timestamps"].tolist() == [10, 20, 30, 40, 50, 0]
    assert row["lengths"].item() == 4
    test = SequentialCSVDataset(path, "test", 4, 2)[0]
    assert test["input_ids"].tolist() == [2, 3, 4, 5, 0, 0]
    assert test["target_items"].item() == 6
    with pytest.raises(ValueError, match="no validation"):
        SequentialCSVDataset(path, "valid", 4, 2)


def test_leave_one_out_keeps_short_user_only_in_evaluable_splits(tmp_path):
    path = tmp_path / "users.csv"
    with path.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sequence_item_ids", "sequence_timestamps"])
        writer.writerows([["1,2,3", "10,20,30"], ["4,5,6,7", "40,50,60,70"]])

    options = {"max_seq_len": 3, "max_output_len": 1, "evaluation_protocol": "leave_one_out"}
    train = SequentialCSVDataset(path, "train", **options)
    assert len(train) == 1
    assert train[0]["targets"].tolist() == [5, 0, 0, 0]
    for phase, expected_target in [("valid", 2), ("test", 3)]:
        dataset = SequentialCSVDataset(path, phase, **options)
        assert len(dataset) == 2
        assert dataset[0]["target_items"].item() == expected_target


def test_history_filter_excludes_repeated_target_when_catalogue_is_small():
    model = FuXiAlpha(3, max_seq_len=2, max_output_len=1, embed_dim=4, num_blocks=1)
    batch = {
        "input_ids": torch.tensor([[1, 2, 0], [1, 2, 0]]),
        "timestamps": torch.tensor([[10, 20, 30], [10, 20, 30]]),
        "target_items": torch.tensor([1, 3]),
    }
    cpu = SimpleNamespace(device=torch.device("cpu"), gather_for_metrics=lambda values: values)
    metrics = evaluate(model, [batch], cpu, torch.tensor([1, 2, 3]))
    assert metrics["Recall@1"] == metrics["Recall@10"] == metrics["NDCG@10"] == 0.5
    assert metrics["MRR"] == pytest.approx((1 + 1 / 2501) / 2)


def test_fuxi_preserves_causality_padding_and_batch_independence():
    torch.manual_seed(11)
    model = FuXiAlpha(8, max_seq_len=4, max_output_len=2, embed_dim=4,
                      num_blocks=2, num_heads=2, dqk=2, dv=3, dropout=0.0).eval()
    ids = torch.tensor([[1, 2, 3, 4, 0, 0]])
    stamps = torch.tensor([[10, 20, 30, 40, 50, 0]])
    original = model.encode(ids, stamps)
    changed = ids.clone(); changed[0, 3] = 5
    torch.testing.assert_close(original[:, :3], model.encode(changed, stamps)[:, :3])
    together = model.encode(torch.cat([ids, changed]), stamps.repeat(2, 1))
    torch.testing.assert_close(original, together[:1])
    assert original[:, 4:].count_nonzero() == 0
    with pytest.raises(ValueError, match="right padded"):
        model.encode(torch.tensor([[0, 1, 2, 3, 0, 0]]), stamps)


def test_fuxi_masks_positive_negative_collisions_and_backpropagates():
    model = FuXiAlpha(8, max_seq_len=3, max_output_len=1, embed_dim=4,
                      num_blocks=1, dropout=0.0)
    ids = torch.tensor([[1, 2, 3, 0]])
    stamps = torch.tensor([[10, 20, 30, 40]])
    targets = torch.tensor([[2, 3, 4, 0]])
    _, collision_loss = model(ids, stamps, targets, negative_ids=torch.tensor([[2, 2], [3, 3], [4, 4]]))
    torch.testing.assert_close(collision_loss, torch.tensor(0.0))
    _, loss = model(ids, stamps, targets, negative_ids=torch.tensor([[5, 6], [5, 6], [5, 6]]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.layers[0]._uvqk.grad.abs().sum() > 0
    assert model.item_embedding.weight.grad.abs().sum() > 0


def test_native_default_constructor_and_features_reserve_target_position(tmp_path):
    path = tmp_path / "users.csv"; _csv(path)
    row = SequentialCSVDataset(path, "train")[0]
    model = FuXiAlpha(3952)
    assert row["input_ids"].shape == row["timestamps"].shape == (211,)
    assert model.position_embedding.weight.shape == (211, 50)
    assert all(layer._rel_attn_bias._pos_w.shape == (421,) for layer in model.layers)
    # From pinned upstream ML-1M model-info.json, including all padding weights.
    assert sum(p.numel() for p in model.parameters()) == 269400
