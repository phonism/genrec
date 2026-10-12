from types import SimpleNamespace

import pytest
import torch

from genrec.data.amazon_onerec import add_disambiguation_suffix
from genrec.modules.prefix_trie import PrefixTrie


class FakeTokenizer:
    eos_token_id = 7

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "### Response:":
            return SimpleNamespace(input_ids=[3, 4])
        codebook, code = text.removeprefix("<C").removesuffix(">").split("_")
        return SimpleNamespace(input_ids=[10 + int(codebook) * 16 + int(code)])


def test_onerec_suffix_makes_every_catalog_sid_unique():
    raw = [[1, 2, 3], [1, 2, 3], [1, 2, 4], [1, 2, 3]]
    final, stats = add_disambiguation_suffix(raw, codebook_size=8)

    assert final == [
        [1, 2, 3, 0],
        [1, 2, 3, 1],
        [1, 2, 4, 0],
        [1, 2, 3, 2],
    ]
    assert len({tuple(sem_ids) for sem_ids in final}) == len(final)
    assert stats == {
        "raw_unique": 2,
        "collision_groups": 1,
        "collision_items": 3,
        "max_group": 3,
    }


def test_onerec_suffix_rejects_unrepresentable_collision_bucket():
    with pytest.raises(ValueError, match="exceeds codebook token range"):
        add_disambiguation_suffix([[0], [0], [0]], codebook_size=2)


def test_prefix_trie_allows_only_catalog_paths():
    tokenizer = FakeTokenizer()
    trie = PrefixTrie(
        [[1, 2, 3], [1, 4, 5], [2, 6, 7]],
        num_codebooks=3,
        codebook_size=8,
        tokenizer=tokenizer,
    )

    c0_1 = tokenizer("<C0_1>").input_ids[0]
    c0_2 = tokenizer("<C0_2>").input_ids[0]
    c1_2 = tokenizer("<C1_2>").input_ids[0]
    c1_4 = tokenizer("<C1_4>").input_ids[0]
    c2_3 = tokenizer("<C2_3>").input_ids[0]

    assert set(trie.allowed_token_ids(())) == {c0_1, c0_2}
    assert set(trie.allowed_token_ids((c0_1,))) == {c1_2, c1_4}
    assert trie.allowed_token_ids((c0_1, c1_2)) == [c2_3]
    assert trie.allowed_token_ids((c0_1, c1_2, c2_3)) == [tokenizer.eos_token_id]
    with pytest.raises(ValueError, match="absent from catalog trie"):
        trie.allowed_token_ids((c0_2, c1_2))


def test_hf_prefix_callback_follows_response_marker():
    tokenizer = FakeTokenizer()
    trie = PrefixTrie(
        [[1, 2]], num_codebooks=2, codebook_size=8, tokenizer=tokenizer
    )
    callback = trie.get_prefix_allowed_tokens_fn()
    c0 = tokenizer("<C0_1>").input_ids[0]
    c1 = tokenizer("<C1_2>").input_ids[0]

    assert callback(0, torch.tensor([99, 3, 4])) == [c0]
    assert callback(0, torch.tensor([99, 3, 4, c0])) == [c1]
    assert callback(0, torch.tensor([99, 3, 4, c0, c1])) == [tokenizer.eos_token_id]
