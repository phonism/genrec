"""Catalog-prefix constraints for semantic-ID generation."""

from collections import defaultdict
import logging
from typing import Callable, Dict, List, Set, Tuple

import torch


class PrefixTrie:
    """Trie containing only semantic-ID sequences present in the catalog."""

    def __init__(
        self,
        sem_ids_list: List[List[int]],
        num_codebooks: int,
        codebook_size: int,
        tokenizer,
    ) -> None:
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.tokenizer = tokenizer

        self.token_id_map: Dict[Tuple[int, int], int] = {}
        for codebook in range(num_codebooks):
            for code in range(codebook_size):
                ids = tokenizer(
                    f"<C{codebook}_{code}>", add_special_tokens=False
                ).input_ids
                if len(ids) == 1:
                    self.token_id_map[(codebook, code)] = ids[0]

        self.trie: Dict[tuple, Set[int]] = defaultdict(set)
        self._build_trie(sem_ids_list)
        self.response_marker_ids = tokenizer(
            "### Response:", add_special_tokens=False
        ).input_ids
        self.all_codebook_tokens: Set[int] = set(self.token_id_map.values())

    def sem_ids_to_token_ids(self, sem_ids: List[int]) -> List[int]:
        """Map one semantic ID to position-specific codebook token IDs."""
        return [
            self.token_id_map[(codebook, code)]
            for codebook, code in enumerate(sem_ids)
            if codebook < self.num_codebooks
            and (codebook, code) in self.token_id_map
        ]

    def _build_trie(self, sem_ids_list: List[List[int]]) -> None:
        eos_id = self.tokenizer.eos_token_id
        skipped = 0
        for sem_ids in sem_ids_list:
            token_ids = self.sem_ids_to_token_ids(sem_ids)
            if len(token_ids) != self.num_codebooks:
                skipped += 1
                continue
            for depth in range(len(token_ids)):
                self.trie[tuple(token_ids[:depth])].add(token_ids[depth])
            self.trie[tuple(token_ids)].add(eos_id)

        if skipped:
            logging.getLogger(__name__).warning(
                "PrefixTrie: skipped %d/%d items whose semantic IDs have no "
                "codebook token",
                skipped,
                len(sem_ids_list),
            )
        if not self.trie.get(()) or skipped == len(sem_ids_list):
            raise ValueError("PrefixTrie contains no valid catalog semantic IDs")

    def allowed_token_ids(self, generated_prefix) -> List[int]:
        """Return valid next tokens for a generated codebook-token prefix."""
        prefix = tuple(int(token) for token in generated_prefix)
        allowed = self.trie.get(prefix)
        if not allowed:
            raise ValueError(f"semantic-ID prefix is absent from catalog trie: {prefix}")
        return list(allowed)

    def get_prefix_allowed_tokens_fn(self) -> Callable:
        """Return a callback compatible with Hugging Face ``generate``."""

        def fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
            del batch_id
            input_list = input_ids.tolist()
            marker_len = len(self.response_marker_ids)
            response_start = -1
            for index in range(len(input_list) - marker_len + 1):
                if input_list[index:index + marker_len] == self.response_marker_ids:
                    response_start = index + marker_len
                    break

            if response_start == -1:
                generated = []
                for token in reversed(input_list):
                    if token in self.all_codebook_tokens:
                        generated.insert(0, token)
                    else:
                        break
            else:
                generated = input_list[response_start:]

            allowed = self.trie.get(tuple(generated))
            if allowed:
                return list(allowed)
            return [self.tokenizer.eos_token_id]

        return fn
