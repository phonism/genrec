import pytest
import torch

from genrec.models.hstu import HSTU


def _tiny_hstu():
    return HSTU(
        num_items=20,
        max_seq_len=4,
        embed_dim=8,
        num_heads=2,
        num_blocks=1,
        dropout=0.0,
        use_temporal_bias=False,
        score_l2_norm=True,
        score_temperature=0.05,
    )


def test_meta_local_sampler_draws_negatives_per_positive(monkeypatch):
    model = _tiny_hstu()
    input_ids = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]])
    targets = torch.tensor([[0, 2, 3, 4], [0, 5, 6, 7]])
    randint_shapes = []
    original_randint = torch.randint

    def recording_randint(*args, **kwargs):
        size = kwargs.get("size")
        if size is None and len(args) >= 3:
            size = args[2]
        randint_shapes.append(tuple(size))
        return original_randint(*args, **kwargs)

    monkeypatch.setattr(torch, "randint", recording_randint)
    _, loss = model.forward_sampled_softmax(
        input_ids,
        targets=targets,
        num_negatives=4,
        sampling_strategy="local",
    )

    # Six non-padding positive positions, each with its own four samples.
    assert (6, 4) in randint_shapes
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_sampled_softmax_rejects_unknown_sampler():
    model = _tiny_hstu()
    with pytest.raises(ValueError, match="sampling_strategy"):
        model.forward_sampled_softmax(
            torch.tensor([[1, 2, 3, 4]]),
            targets=torch.tensor([[2, 3, 4, 5]]),
            sampling_strategy="batchish",
        )
