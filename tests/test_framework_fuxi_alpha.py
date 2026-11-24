"""Causality, target/candidate contracts for the padded FuXi-alpha port."""
import csv

import pytest
import torch

from genrec.data.sequential_csv import SequentialCSVDataset
from genrec.models.fuxi_alpha import FuXiAlpha


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
