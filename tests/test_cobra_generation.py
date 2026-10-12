import types

import pytest
import torch
from torch import nn

from genrec.models.cobra import Cobra, CobraGenerationOutput
from genrec.modules.encoder import LightT5Encoder
from genrec.data.amazon_cobra import AmazonCobraDataset


def _tiny_cobra():
    return Cobra(
        encoder_n_layers=1,
        encoder_hidden_dim=4,
        encoder_num_heads=1,
        encoder_vocab_size=16,
        id_vocab_size=4,
        n_codebooks=3,
        d_model=4,
        max_len=64,
        decoder_n_layers=1,
        decoder_num_heads=1,
        encoder_type="light",
        use_dense=False,
    )


def test_cobra_legacy_text_flattens_historical_rqvae_attributes():
    dataset = AmazonCobraDataset.__new__(AmazonCobraDataset)
    dataset.item_text_format = "flattened_attributes"
    text = dataset._format_item_text({
        "title": "Face Cream",
        "price": 12.5,
        "salesRank": {"Beauty": 7},
        "brand": "Example",
        "categories": [["Beauty", "Skin Care"]],
    }, item_id=3)

    for field in ("title", "price", "salesRank", "brand", "categories"):
        assert f"'{field}':" in text


def test_cobra_paper_text_uses_exact_published_fields():
    dataset = AmazonCobraDataset.__new__(AmazonCobraDataset)
    dataset.item_text_format = "paper_attributes"
    text = dataset._format_item_text({
        "title": "Face Cream",
        "price": 12.5,
        "salesRank": {"Beauty": 7},
        "brand": "Example",
        "categories": [["Beauty", "Skin Care"]],
        "description": ["A hydrating cream"],
    }, item_id=3)

    for field in ("title", "price", "category", "description"):
        assert f"{field}:" in text
    assert "salesRank" not in text
    assert "brand:" not in text


def test_light_encoder_cls_pooling_is_finite_padding_invariant_and_trainable():
    encoder = LightT5Encoder(
        n_layers=1,
        hidden_dim=4,
        output_dim=4,
        num_heads=1,
        ff_dim=8,
        vocab_size=16,
        max_seq_len=4,
        dropout=0.0,
        pooling="cls",
    )
    encoder.eval()

    short = encoder(torch.tensor([[1, 2]]))
    padded = encoder(torch.tensor([[1, 2, 0, 0]]))

    assert short.shape == (1, 4)
    assert torch.isfinite(padded).all()
    assert torch.allclose(short, padded, atol=1e-6)
    padded.sum().backward()
    assert encoder.cls_embedding.grad is not None


def test_light_encoder_mean_pooling_keeps_historical_state_shape():
    encoder = LightT5Encoder(
        n_layers=1,
        hidden_dim=4,
        output_dim=4,
        num_heads=1,
        ff_dim=8,
        vocab_size=16,
        max_seq_len=4,
        dropout=0.0,
        pooling="mean",
    )

    assert "cls_embedding" not in encoder.state_dict()
    assert encoder.pos_embedding.weight.shape == (4, 4)


def test_cobra_rejects_unknown_dense_negative_scope():
    try:
        Cobra(
            encoder_hidden_dim=4,
            encoder_num_heads=1,
            encoder_vocab_size=16,
            id_vocab_size=4,
            d_model=4,
            decoder_n_layers=1,
            decoder_num_heads=1,
            dense_negative_scope="within_user",
        )
    except ValueError as exc:
        assert "dense_negative_scope" in str(exc)
    else:
        raise AssertionError("invalid dense negative scope was accepted")


def test_cobra_rejects_unknown_encoder_type():
    with pytest.raises(ValueError, match="encoder_type"):
        Cobra(
            encoder_n_layers=1,
            encoder_hidden_dim=16,
            encoder_num_heads=2,
            encoder_vocab_size=32,
            id_vocab_size=8,
            n_codebooks=2,
            d_model=16,
            max_len=16,
            decoder_n_layers=1,
            decoder_num_heads=2,
            encoder_type="typo",
        )
def test_cobra_masks_duplicate_catalog_items_as_false_negatives():
    similarities = torch.tensor([
        [3.0, 3.0, 1.0],
        [3.0, 3.0, 1.0],
        [1.0, 1.0, 3.0],
    ])
    masked = Cobra._mask_duplicate_item_negatives(
        similarities, torch.tensor([7, 7, 9])
    )

    assert masked[0, 0] == 3.0
    assert masked[1, 1] == 3.0
    assert masked[0, 1] < -1e8
    assert masked[1, 0] < -1e8
    assert masked[0, 2] == 1.0


def test_dense_memory_queue_runs_end_to_end_and_tracks_item_ids():
    model = Cobra(
        encoder_n_layers=1,
        encoder_hidden_dim=4,
        encoder_num_heads=1,
        encoder_vocab_size=16,
        id_vocab_size=6,
        n_codebooks=3,
        d_model=4,
        max_len=64,
        queue_size=3,
        decoder_n_layers=1,
        decoder_num_heads=1,
        decoder_dropout=0.0,
        encoder_type="light",
        dense_negative_scope="all_batch",
        use_dense_memory_queue=True,
    )
    model.train()

    output = model(
        input_ids=torch.tensor([
            [0, 1, 2, 1, 2, 3],
            [2, 3, 4, 3, 4, 5],
        ]),
        encoder_input_ids=torch.tensor([
            [[1, 2, 0], [3, 4, 0]],
            [[5, 6, 0], [7, 8, 0]],
        ]),
        item_ids=torch.tensor([[10, 11], [20, 21]]),
    )

    assert torch.isfinite(output.loss_dense)
    assert model.queue_filled.item() == 2
    assert set(model.item_queue[:2].tolist()) == {11, 21}

    # The second step consumes queued features and updates the circular buffer
    # before backward.  This catches accidental use of a detached *view* of the
    # queue, whose in-place update invalidates autograd's version counter.
    output.loss.backward()
    model.zero_grad(set_to_none=True)
    output = model(
        input_ids=torch.tensor([
            [0, 1, 2, 1, 2, 3],
            [2, 3, 4, 3, 4, 5],
        ]),
        encoder_input_ids=torch.tensor([
            [[1, 2, 0], [3, 4, 0]],
            [[5, 6, 0], [7, 8, 0]],
        ]),
        item_ids=torch.tensor([[10, 11], [20, 21]]),
    )
    assert torch.isfinite(output.loss_dense)
    output.loss.backward()

    model._dequeue_and_enqueue(
        torch.nn.functional.normalize(torch.randn(2, 4), dim=-1),
        torch.tensor([31, 32]),
    )
    assert model.queue_filled.item() == 3
    assert set(model.item_queue.tolist()) == {21, 31, 32}


def test_fuse_generated_supports_multiple_weights_without_regeneration():
    model = _tiny_cobra()
    generated = CobraGenerationOutput(
        sem_ids=torch.tensor([[[0, 0, 0], [1, 1, 1]]]),
        dense_vecs=torch.tensor([[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]),
        scores=torch.tensor([[-0.1, -1.0]]),
    )
    item_sem_ids = torch.tensor([
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1],
    ])
    item_dense_vecs = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    buckets = {(0, 0, 0): [0, 1], (1, 1, 1): [2]}

    dense_heavy = model.fuse_generated(
        generated, item_dense_vecs, item_sem_ids, buckets,
        n_candidates=3, tau=0.1, psi=16.0,
    )
    sparse_heavy = model.fuse_generated(
        generated, item_dense_vecs, item_sem_ids, buckets,
        n_candidates=3, tau=8.0, psi=0.1,
    )

    assert dense_heavy.item_ids.shape == (1, 3)
    assert dense_heavy.item_ids[0, 0].item() in {1, 2}
    assert sparse_heavy.item_ids[0, 0].item() in {0, 1}
    assert len(set(dense_heavy.item_ids[0].tolist())) == 3


class _RecordingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_tokens = None

    def forward(self, tokens):
        self.last_tokens = tokens.detach().clone()
        values = tokens.float().sum(dim=-1, keepdim=True)
        return values.repeat_interleave(4, dim=-1)


class _RecordingEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.last_input_ids = None
        self.last_mask = None

    def forward(self, input_ids, input_vecs, mask, n_complete_items=None):
        self.last_input_ids = input_ids.detach().clone()
        self.last_mask = mask.detach().clone()
        return torch.zeros(
            input_ids.size(0), mask.size(1), self.d_model,
            device=input_ids.device,
        )


class _IdentityDecoder(nn.Module):
    def forward(self, tgt, **kwargs):
        return tgt


def test_encode_item_tokens_skips_padded_rows():
    model = _tiny_cobra()
    encoder = _RecordingEncoder()
    model.encoder = encoder

    tokens = torch.tensor([
        [[1, 2], [3, 4], [0, 0]],
        [[5, 6], [0, 0], [0, 0]],
    ])
    item_mask = torch.tensor([
        [True, True, False],
        [True, False, False],
    ])

    vecs = model._encode_item_tokens(tokens, item_mask=item_mask)

    assert encoder.last_tokens.shape[0] == 3
    assert vecs.shape == (2, 3, 4)
    assert torch.equal(vecs[~item_mask], torch.zeros(3, 4))
    assert torch.count_nonzero(vecs[item_mask]) > 0


def test_generated_prefix_is_inserted_before_each_samples_padding_gap():
    model = _tiny_cobra()
    recorder = _RecordingEmbedding(model.d_model)
    model.cobra_emb = recorder
    model.decoder = _IdentityDecoder()

    pad = model.pad_id
    input_ids = torch.tensor([
        [0, 1, 2, pad, pad, pad],
        [1, 2, 3, 2, 3, 0],
    ])
    vecs = torch.zeros(2, 2, model.d_model)
    prefix = torch.tensor([
        [[3, 1]],
        [[0, 2]],
    ])

    hidden = model._decode_generated_prefix(
        input_ids=input_ids,
        vecs=vecs,
        generated_prefix=prefix,
        history_lengths=torch.tensor([1, 2]),
    )

    # Sample 0's target begins at raw sparse position 3, not after the batch's
    # padded two-item width.  Sample 1 uses the extra item slot at position 6.
    assert recorder.last_input_ids[0, 3:5].tolist() == [3, 1]
    assert recorder.last_input_ids[1, 6:8].tolist() == [0, 2]
    assert hidden.shape == (2, 1, model.d_model)


def test_catalog_constrained_generation_emits_only_real_sids_and_refines_after_full_sid():
    model = _tiny_cobra()
    model.cobra_emb = _RecordingEmbedding(model.d_model)
    model.decoder = _IdentityDecoder()
    for head in model.sparse_head:
        nn.init.zeros_(head.weight)
        head.bias.data.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))

    decoded_prefix_lengths = []

    def fake_decode(self, input_ids, vecs, generated_prefix, history_lengths):
        decoded_prefix_lengths.append(generated_prefix.size(-1))
        return torch.zeros(
            generated_prefix.size(0), generated_prefix.size(1), self.d_model
        )

    model._decode_generated_prefix = types.MethodType(fake_decode, model)
    valid_sem_ids = torch.tensor([
        [0, 1, 2],
        [3, 0, 1],
    ])

    output = model.generate(
        input_ids=torch.tensor([[0, 0, 0]]),
        encoder_input_ids=torch.zeros(1, 1, 2, dtype=torch.long),
        n_candidates=2,
        valid_sem_ids=valid_sem_ids,
    )

    valid = {tuple(row) for row in valid_sem_ids.tolist()}
    assert {tuple(row) for row in output.sem_ids[0].tolist()} == valid
    # c1 and c2 predictions consume prefixes of length 1/2; dense refinement
    # must then re-run the decoder with all three generated codes.
    assert decoded_prefix_lengths == [1, 2, 3]
