import torch

from genrec.models.rqvae import Quantize, QuantizeForwardMode, RqVae


def test_quantize_random_init_matches_reference_scale():
    torch.manual_seed(0)
    quantize = Quantize(
        embed_dim=8,
        n_embed=256,
        do_kmeans_init=False,
        forward_mode=QuantizeForwardMode.STE,
    )

    bound = 1.0 / 256
    assert quantize.weight.min().item() >= -bound
    assert quantize.weight.max().item() <= bound
    assert abs(quantize.weight.mean().item()) < bound / 10


def test_model_config_contains_arguments_not_model_instance():
    model = RqVae(**_config())

    assert "self" not in model.config
    assert model.config["input_dim"] == 4


def _config():
    return {
        "input_dim": 4,
        "embed_dim": 2,
        "hidden_dims": [3],
        "codebook_size": 2,
        "codebook_kmeans_init": True,
        "codebook_normalize": False,
        "codebook_sim_vq": False,
        "codebook_mode": QuantizeForwardMode.STE,
        "codebook_last_layer_mode": QuantizeForwardMode.STE,
        "n_layers": 1,
        "commitment_weight": 0.25,
        "n_cat_features": 0,
    }


def test_from_pretrained_never_reinitializes_trained_codebook(tmp_path):
    model = RqVae(**_config())
    checkpoint = tmp_path / "rqvae.pt"
    torch.save(
        {"model": model.state_dict(), "model_config": _config(), "epoch": 7},
        checkpoint,
    )

    loaded = RqVae.from_pretrained(str(checkpoint))
    loaded.eval()
    before = loaded.layers[0].weight.detach().clone()
    loaded.get_semantic_ids(torch.randn(8, 4))

    assert loaded.layers[0].do_kmeans_init is False
    assert loaded.layers[0].kmeans_initted is True
    assert torch.equal(before, loaded.layers[0].weight)


def test_legacy_load_pretrained_disables_reinitialization(tmp_path):
    source = RqVae(**_config())
    checkpoint = tmp_path / "rqvae.pt"
    torch.save(
        {"model": source.state_dict(), "model_config": _config(), "epoch": 7},
        checkpoint,
    )

    loaded = RqVae(**_config())
    loaded.load_pretrained(str(checkpoint))

    assert all(not layer.do_kmeans_init for layer in loaded.layers)
    assert all(layer.kmeans_initted for layer in loaded.layers)


def test_paper_decoder_can_reconstruct_without_forced_unit_norm():
    model = RqVae(**_config(), decoder_normalize=False)
    model.eval()
    decoded = model.decode(torch.randn(8, 2))

    assert model.decoder_normalize is False
    assert not torch.allclose(decoded.norm(dim=-1), torch.ones(8))


def test_reference_mean_loss_normalizes_dimensions_and_rq_layers():
    config = {**_config(), "codebook_kmeans_init": False}
    sum_model = RqVae(**config)
    mean_model = RqVae(**config, loss_reduction="mean")
    mean_model.load_state_dict(sum_model.state_dict())
    batch = torch.randn(8, 4)

    sum_output = sum_model(batch, gumbel_t=0.2)
    mean_output = mean_model(batch, gumbel_t=0.2)

    assert torch.allclose(
        mean_output.reconstruction_loss,
        sum_output.reconstruction_loss / sum_model.input_dim,
    )
    assert torch.allclose(
        mean_output.rqvae_loss,
        sum_output.rqvae_loss / (sum_model.embed_dim * sum_model.n_layers),
    )


def test_reference_xavier_init_zeros_mlp_biases():
    config = {**_config(), "codebook_kmeans_init": False}
    model = RqVae(
        **config,
        mlp_bias=True,
        mlp_weight_init="xavier_normal",
    )
    linear_layers = [
        module for module in model.modules() if isinstance(module, torch.nn.Linear)
    ]

    assert linear_layers
    assert all(
        layer.bias is None or torch.count_nonzero(layer.bias) == 0
        for layer in linear_layers
    )


def test_from_pretrained_preserves_reference_mlp_and_loss_options(tmp_path):
    config = {
        **_config(),
        "codebook_kmeans_init": False,
        "mlp_activation": "relu",
        "mlp_bias": True,
        "mlp_weight_init": "xavier_normal",
        "decoder_normalize": False,
        "loss_reduction": "mean",
    }
    source = RqVae(**config)
    checkpoint = tmp_path / "rqvae_reference.pt"
    torch.save(
        {"model": source.state_dict(), "model_config": config, "epoch": 3},
        checkpoint,
    )

    loaded = RqVae.from_pretrained(str(checkpoint))

    assert loaded.encoder.activation == "relu"
    assert loaded.encoder.weight_init == "xavier_normal"
    assert loaded.decoder_normalize is False
    assert loaded.loss_reduction == "mean"
    assert any(
        layer.bias is not None
        for layer in loaded.encoder.modules()
        if isinstance(layer, torch.nn.Linear)
    )
