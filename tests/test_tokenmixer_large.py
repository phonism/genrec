"""Mathematical contracts for a paper reconstruction without an author oracle."""
import torch
import pytest
from torch.nn import functional as F

from genrec.models.tokenmixer_large import (
    GroupwiseTokenizer, PerTokenSwiGLU, SparsePerTokenMoE,
    TokenMixerLarge, TokenMixerLargeBlock, mix_tokens, revert_tokens,
)


FEATURES = {"I1": {"type": "numeric"}, "C1": {"type": "categorical", "vocab_size": 8, "padding_idx": 0},
            "C2": {"type": "categorical", "vocab_size": 6, "padding_idx": 0}}


def test_mixing_preserves_every_token_head_chunk_and_has_exact_inverse():
    x = torch.arange(24.0).reshape(1, 3, 8).requires_grad_()
    mixed = mix_tokens(x, 2)
    assert mixed.shape == (1, 2, 12)
    torch.testing.assert_close(mixed[0, 0], torch.tensor([0., 1., 2., 3., 8., 9., 10., 11., 16., 17., 18., 19.]))
    restored = revert_tokens(mixed, 3)
    torch.testing.assert_close(restored, x, rtol=0, atol=0)
    restored.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    with pytest.raises(ValueError, match="divisible"):
        mix_tokens(x, 3)


def test_batched_swiglu_matches_independent_tokenwise_matrix_formula():
    torch.manual_seed(7)
    bank = PerTokenSwiGLU(3, 4, 8)
    x = torch.randn(5, 3, 4, requires_grad=True)
    expected = torch.stack([
        (F.silu(x[:, i] @ bank.gate_weight[i]) * (x[:, i] @ bank.up_weight[i])) @ bank.down_weight[i]
        for i in range(3)
    ], 1)
    actual = bank(x)
    torch.testing.assert_close(actual, expected)
    parameters = (x,) + tuple(bank.parameters())
    left = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
    right = torch.autograd.grad(expected.square().sum(), parameters)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b)


def test_small_initialization_scales_standard_deviation_not_variance():
    torch.manual_seed(23); original = PerTokenSwiGLU(2, 8, 16, down_init_gain=1.0)
    torch.manual_seed(23); small = PerTokenSwiGLU(2, 8, 16, down_init_gain=0.01)
    torch.testing.assert_close(small.up_weight, original.up_weight)
    torch.testing.assert_close(small.gate_weight, original.gate_weight)
    torch.testing.assert_close(small.down_weight, original.down_weight * 0.01)


def test_zero_down_projections_leave_both_block_residual_paths_intact():
    block = TokenMixerLargeBlock(3, 8, 2, hidden_multiplier=2)
    with torch.no_grad():
        block.mix_ffn.down_weight.zero_(); block.token_ffn.down_weight.zero_()
    x = torch.randn(2, 3, 8, requires_grad=True)
    y = block(x)
    torch.testing.assert_close(y, x, rtol=0, atol=0)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_dense_hidden_partition_recovers_dense_function_and_input_gradient():
    torch.manual_seed(12)
    dense = PerTokenSwiGLU(2, 4, 16)
    sparse = SparsePerTokenMoE(2, 4, 16, num_experts=4, top_k=4, gate_scale=3.0)
    sparse.partition_from_dense(dense)
    with torch.no_grad(): sparse.router_weight.zero_()
    x = torch.randn(3, 2, 4, requires_grad=True)
    original, reconstructed = dense(x), sparse(x)
    torch.testing.assert_close(reconstructed, original, rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(torch.autograd.grad(original.sum(), x)[0],
                               torch.autograd.grad(reconstructed.sum(), x)[0], rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("normalization", ["selected", "all"])
def test_sparse_dispatch_computes_only_chosen_rows_and_exposes_top1_gradient(normalization):
    torch.manual_seed(5)
    moe = SparsePerTokenMoE(1, 4, 16, 4, 2, router_normalization=normalization)
    with torch.no_grad(): moe.router_weight[0].copy_(torch.tensor([[1., 0., -1.]]).expand(4, -1))
    calls = [0, 0, 0]
    hooks = []
    for index, expert in enumerate(moe.routed[0]):
        def count(module, inputs, output, i=index): calls[i] += len(inputs[0])
        hooks.append(expert.register_forward_hook(count))
    x = torch.ones(5, 1, 4, requires_grad=True)
    y = moe(x)
    y.sum().backward()
    for hook in hooks: hook.remove()
    assert calls == [5, 0, 0]
    assert moe.routed[0][0].down.weight.grad is not None
    assert moe.routed[0][1].down.weight.grad is None
    assert moe.routed[0][2].down.weight.grad is None
    gradient = moe.router_weight.grad.abs().sum().item()
    assert gradient == 0 if normalization == "selected" else gradient > 0
    summary = moe.routing_summary()
    assert summary["routed_fractions_per_slot"] == [[1., 0., 0.]]
    assert summary["activated_experts_including_shared"] == 2


def test_selected_top2_gates_normalize_and_train_router():
    torch.manual_seed(19)
    moe = SparsePerTokenMoE(2, 4, 16, 4, 3)
    x = torch.randn(5, 2, 4)
    _, weights = moe.routes(x)
    torch.testing.assert_close(weights.sum(-1), torch.ones(5, 2))
    moe(x).square().sum().backward()
    assert moe.router_weight.grad.abs().sum() > 0
    assert moe.route_counts.sum().item() == 5 * 2 * 2


def test_global_and_group_tokens_and_explicit_group_partition():
    tokenizer = GroupwiseTokenizer(FEATURES, 2, 3, 8, hidden=4, feature_groups=[["I1"], ["C1", "C2"]])
    embedded = torch.randn(2, 3, 2)
    changed = embedded.clone(); changed[:, 0] += 4
    original, updated = tokenizer(embedded), tokenizer(changed)
    assert not torch.allclose(original[:, 0], updated[:, 0])
    assert not torch.allclose(original[:, 1], updated[:, 1])
    torch.testing.assert_close(original[:, 2], updated[:, 2])
    with pytest.raises(ValueError, match="partition"):
        GroupwiseTokenizer(FEATURES, 2, 3, 8, feature_groups=[["I1"], ["I1", "C1"]])


def test_joint_auxiliary_loss_inference_and_serialized_model_are_consistent(tmp_path):
    torch.manual_seed(15)
    kwargs = dict(feature_map=FEATURES, embedding_dim=4, token_num=4, token_dim=8,
                  num_blocks=6, num_heads=2, hidden_multiplier=2, tokenizer_hidden=8,
                  inter_residual_gap=2, aux_loss_weight=0.2, embedding_regularizer=0)
    model = TokenMixerLarge(**kwargs)
    assert model.aux_depths == [2, 4]  # never add the interval residual to final block 6
    x, labels = torch.tensor([[.2, 1, 2], [.6, 3, 4], [.8, 2, 1]]), torch.tensor([0., 1., 0.])
    logits, intermediate = model.encode(x, return_auxiliary=True)
    target = labels[:, None]
    expected = F.binary_cross_entropy_with_logits(logits, target) + 0.2 * (
        F.binary_cross_entropy_with_logits(logits + intermediate[4], target) +
        F.binary_cross_entropy_with_logits(logits + intermediate[4] + intermediate[2], target)) / 2
    actual_logits, loss = model(x, labels)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert model.aux_heads["2"].weight.grad.abs().sum() > 0
    assert model.blocks[0].mix_ffn.up_weight.grad.abs().sum() > 0
    torch.testing.assert_close(model(x)[0], actual_logits)
    path = tmp_path / "weights.pt"; torch.save({"model": model.state_dict(), "kwargs": kwargs}, path)
    state = torch.load(path, weights_only=False)
    restored = TokenMixerLarge(**state["kwargs"]); restored.load_state_dict(state["model"])
    torch.testing.assert_close(restored(x)[0], actual_logits)
