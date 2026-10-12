import pytest
import torch
from torch.utils.data import DataLoader

from genrec.data.amazon_notellm import (
    AmazonNoteLLMDataset,
    build_item_text,
    format_item_prompt,
    notellm_collate_fn,
)
from genrec.models.notellm import Query2Embedding


def test_notellm_full_catalog_metrics_rank_exact_item_ids():
    catalog = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32
    )
    queries = torch.tensor([[0.9, 0.1], [-0.8, 0.0]], dtype=torch.float32)
    metrics = Query2Embedding.compute_full_catalog_metrics(
        query_embeddings=queries,
        catalog_embeddings=catalog,
        target_item_ids=torch.tensor([10, 30]),
        catalog_item_ids=torch.tensor([10, 20, 30]),
        topks=(1, 2),
    )
    assert metrics["recall@1"] == pytest.approx(1.0)
    assert metrics["ndcg@1"] == pytest.approx(1.0)


def test_notellm_full_catalog_rejects_duplicate_catalog_ids():
    with pytest.raises(ValueError, match="unique"):
        Query2Embedding.compute_full_catalog_metrics(
            query_embeddings=torch.eye(2),
            catalog_embeddings=torch.eye(2),
            target_item_ids=torch.tensor([1, 1]),
            catalog_item_ids=torch.tensor([1, 1]),
        )


def test_notellm_full_catalog_excludes_i2i_query_item():
    metrics = Query2Embedding.compute_full_catalog_metrics(
        query_embeddings=torch.tensor([[1.0, 0.0]]),
        catalog_embeddings=torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]
        ),
        target_item_ids=torch.tensor([2]),
        catalog_item_ids=torch.tensor([1, 2, 3]),
        excluded_item_ids=torch.tensor([1]),
        topks=(1,),
    )
    assert metrics["recall@1"] == pytest.approx(1.0)


def test_notellm_full_catalog_rejects_excluding_target():
    with pytest.raises(ValueError, match="target item cannot also be excluded"):
        Query2Embedding.compute_full_catalog_metrics(
            query_embeddings=torch.eye(2),
            catalog_embeddings=torch.eye(2),
            target_item_ids=torch.tensor([1, 2]),
            catalog_item_ids=torch.tensor([1, 2]),
            excluded_item_ids=torch.tensor([1, 2]),
        )


def test_notellm_prompt_sanitizes_marker_and_keeps_it_at_end():
    assert format_item_prompt("a [EMB] product").endswith("[EMB]")
    assert format_item_prompt("a [EMB] product").count("[EMB]") == 1


def test_notellm_item_text_is_deterministic_and_bounded():
    text = build_item_text(
        {
            "title": "Example",
            "brand": "Brand",
            "categories": [["Root", "Leaf"]],
        },
        fallback="asin",
    )
    assert text == "title: Example | brand: Brand | category: Root; Leaf"
    assert build_item_text({}, fallback="asin") == "item: asin"


def test_notellm_adapter_builds_i2i_transitions_for_each_split():
    dataset = object.__new__(AmazonNoteLLMDataset)
    dataset.sequences = [[1, 2, 3, 4, 5]]
    dataset.max_seq_len = 50

    dataset.train_test_split = "train"
    dataset._generate_samples()
    assert dataset.samples == [
        {"query": 1, "target": 2},
        {"query": 2, "target": 3},
    ]

    dataset.train_test_split = "valid"
    dataset._generate_samples()
    assert dataset.samples == [{"query": 3, "target": 4}]

    dataset.train_test_split = "test"
    dataset._generate_samples()
    assert dataset.samples == [{"query": 4, "target": 5}]
    assert dataset[0] == {"query": 4, "target": 5}
    assert notellm_collate_fn(dataset.samples) == {
        "queries": [4],
        "targets": [5],
    }

    batch = next(
        iter(DataLoader(dataset, batch_size=1, collate_fn=notellm_collate_fn))
    )
    assert batch == {"queries": [4], "targets": [5]}


def test_notellm_paired_contrastive_loss_is_symmetric():
    model = object.__new__(Query2Embedding)
    torch.nn.Module.__init__(model)
    model.tau = torch.nn.Parameter(torch.tensor(0.0))

    aligned = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        requires_grad=True,
    )
    misaligned = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    )
    aligned_loss = model.paired_contrastive_loss(aligned)
    misaligned_loss = model.paired_contrastive_loss(misaligned)

    assert aligned_loss < misaligned_loss
    aligned_loss.backward()
    assert aligned.grad is not None
    assert torch.isfinite(aligned.grad).all()


def test_notellm_gcl_masks_repeated_catalog_items_as_false_negatives():
    model = object.__new__(Query2Embedding)
    torch.nn.Module.__init__(model)
    model.tau = torch.nn.Parameter(torch.tensor(0.0))
    embeddings = torch.tensor([[1.0, 0.0]] * 4)

    unmasked = model.paired_contrastive_loss(embeddings)
    masked = model.paired_contrastive_loss(
        embeddings, item_ids=torch.tensor([1, 2, 1, 3])
    )

    assert masked < unmasked


def test_notellm_csft_mask_keeps_real_eos_when_pad_equals_eos():
    class Encoded:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    class FakeTokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=True):
            mapping = {
                "short": [7, 99],
                "long": [5, 6, 99],
                "one": [8],
                "two": [8, 9],
            }
            return Encoded(mapping[text])

    model = object.__new__(Query2Embedding)
    torch.nn.Module.__init__(model)
    model.tokenizer = FakeTokenizer()
    model.emb_id = 99

    tokens = model.tokenize_with_targets(
        ["short", "long"], ["one", "two"], max_length=8
    )

    # First row is [7, 99, 8, EOS, PAD, PAD].  The genuine EOS remains visible.
    assert tokens["attention_mask"][0].tolist() == [1, 1, 1, 1, 0, 0]
    assert tokens["labels"][0].tolist() == [-100, -100, 8, 0, -100, -100]
    assert tokens["emb_token_idx"].squeeze(1).tolist() == [0, 1]
