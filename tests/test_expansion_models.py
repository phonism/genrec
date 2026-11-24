"""Algebraic, causality and checkpoint contracts for additional models."""
import pytest
import torch

from genrec.models.fuxi_beta import FuXiBeta, BetaRelativeBias
from genrec.models.fuxi_gamma import FuXiGamma, GammaRelativeBias
from genrec.models.wukong import LowRankFactorization, WukongInteraction, Wukong


@pytest.mark.parametrize("cls", [FuXiBeta, FuXiGamma])
def test_fuxi_extension_causality_padding_and_finite_gradients(cls, tmp_path):
    torch.manual_seed(4)
    kwargs = dict(num_items=10, max_seq_len=4, max_output_len=2, embed_dim=8,
                  num_heads=2, dv=4, num_blocks=2, dropout=0.0)
    model = cls(**kwargs)
    ids = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 0, 0, 0, 0]])
    times = torch.tensor([[10, 10, 30, 40, 50, 0], [10, 20, 30, 0, 0, 0]])
    original = model.encode(ids, times)
    changed = ids.clone(); changed[0, 3] = 9
    torch.testing.assert_close(original[0, :3], model.encode(changed, times)[0, :3])
    assert original[0, 4:].count_nonzero() == 0 and original[1, 2:].count_nonzero() == 0
    targets = torch.tensor([[2, 3, 4, 7, 0, 0], [6, 8, 0, 0, 0, 0]])
    _, loss = model(ids, times, targets, negative_ids=torch.full((6, 3), 10))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    path = tmp_path / "checkpoint.pt"; torch.save(model.state_dict(), path)
    other = cls(**kwargs); other.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(other(ids, times, last_only=True)[0], model(ids, times, last_only=True)[0])


def test_temporal_functions_preserve_source_signed_and_absolute_semantics():
    t = torch.tensor([[10, 100, 20]])
    beta = BetaRelativeBias(3)
    with torch.no_grad(): beta._pow_a.fill_(0.5); beta._pow_b.fill_(0.5)
    _, b = beta(t)
    assert b[0, 1, 1].item() == 0.5  # relu(-80)+1, not abs(-80)+1
    gamma = GammaRelativeBias(3)
    with torch.no_grad(): gamma._alpha.fill_(0.5); gamma._beta.fill_(0.5)
    g, _ = gamma(t)
    torch.testing.assert_close(g[0, 1, 1], 0.5 * torch.tensor(0.8).pow(torch.tensor(80.).sqrt()))


@pytest.mark.parametrize("projection", ["static", "attentive"])
def test_optimized_fm_matches_naive_pairwise_product_and_all_gradients(projection):
    torch.manual_seed(8)
    layer = LowRankFactorization(5, 4, rank=3, projection=projection)
    x = torch.randn(2, 5, 4, requires_grad=True)
    y = layer.projection_matrix(x)
    naive = (x @ x.transpose(1, 2)) @ y
    optimized = layer(x)
    torch.testing.assert_close(optimized, naive, rtol=2e-5, atol=2e-5)
    params = (x,) + tuple(layer.parameters())
    left = torch.autograd.grad(naive.square().mean(), params, retain_graph=True)
    right = torch.autograd.grad(optimized.square().mean(), params)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=5e-5, atol=5e-5)


def test_wukong_residual_compression_and_end_to_end_loss():
    block = WukongInteraction(3, 4, fmb_tokens=2, lcb_tokens=2, rank=2, hidden=(8,))
    assert block(torch.randn(2, 3, 4)).shape == (2, 4, 4)
    features = {"x": {"type": "numeric"}, "id": {"type": "categorical", "vocab_size": 8, "padding_idx": 0}}
    model = Wukong(features, embedding_dim=4, num_blocks=2, fmb_tokens=2,
                   lcb_tokens=2, rank=2, fmb_hidden=(8,), head_hidden=(8,))
    x = torch.tensor([[0.1, 1], [0.9, 2], [0.2, 3]])
    logits, loss = model(x, torch.tensor([0., 1., 0.]))
    assert logits.shape == (3, 1) and torch.isfinite(loss)
    loss.backward()
    assert model.blocks[0].fm.project[0].weight.grad.abs().sum() > 0
