from types import SimpleNamespace

import torch
from torch import nn

from genrec.models.lcrec import LCRec
from genrec.modules.prefix_trie import PrefixTrie
from tests.test_onerec_prefix import FakeTokenizer

VOCAB = 48
NUM_CODEBOOKS = 2
CODEBOOK_SIZE = 4


def token(codebook, code):
    return FakeTokenizer()(f"<C{codebook}_{code}>").input_ids[0]


class FixedPreferenceLM:
    """Every position prefers code 3, then code 0, regardless of context."""

    def __call__(self, input_ids, attention_mask, past_key_values=None, use_cache=True):
        del attention_mask, past_key_values, use_cache
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, VOCAB)
        for codebook in range(NUM_CODEBOOKS):
            logits[:, :, token(codebook, 3)] = 3.0
            logits[:, :, token(codebook, 0)] = 2.0
        kv = torch.zeros(batch, 1, seq, 1)
        return SimpleNamespace(logits=logits, past_key_values=((kv, kv.clone()),))


def make_model():
    model = LCRec.__new__(LCRec)
    nn.Module.__init__(model)
    model.model = FixedPreferenceLM()
    return model


def decoding_tables():
    eos = FakeTokenizer.eos_token_id
    position_mask = torch.zeros(NUM_CODEBOOKS + 1, VOCAB, dtype=torch.bool)
    code_map = torch.zeros(NUM_CODEBOOKS, VOCAB, dtype=torch.long)
    for codebook in range(NUM_CODEBOOKS):
        for code in range(CODEBOOK_SIZE):
            position_mask[codebook, token(codebook, code)] = True
            code_map[codebook, token(codebook, code)] = code
    position_mask[NUM_CODEBOOKS, eos] = True
    return position_mask, code_map


def run_beam(prefix_trie):
    position_mask, code_map = decoding_tables()
    input_ids = torch.ones(2, 3, dtype=torch.long)
    sem_ids, _ = make_model().generate_constrained_beam(
        input_ids, torch.ones_like(input_ids), position_mask, code_map,
        NUM_CODEBOOKS, beam_width=2, topk=2, prefix_trie=prefix_trie,
    )
    return sem_ids


def test_position_mask_only_matches_reference_decoder():
    # The trainer always passes prefix_trie (None by default); this used to
    # raise TypeError at the first evaluation of every LCRec run.
    assert run_beam(None)[0, 0].tolist() == [3, 3]


def test_catalog_trie_keeps_every_beam_on_catalog_paths():
    catalog = [[0, 1], [3, 2]]
    trie = PrefixTrie(catalog, NUM_CODEBOOKS, CODEBOOK_SIZE, FakeTokenizer())
    sem_ids = run_beam(trie)
    for row in sem_ids.tolist():
        assert row == [[3, 2], [0, 1]]
