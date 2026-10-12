"""
Evaluation metrics for recommendation.
"""
import torch
import math
from collections import Counter, defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union


class SemanticIdItemMapper:
    """Resolve generated semantic IDs to deployable item IDs.

    A semantic-ID tuple is not necessarily unique.  Treating a generated SID
    as a hit for every item in its collision bucket inflates recommendation
    metrics.  This mapper assigns exactly one catalog item to each SID using a
    deterministic, training-only popularity policy and removes duplicate items
    from a ranked list before item-level evaluation.

    ``item_popularity`` must be computed from training interactions only.  Ties
    are resolved by the smaller item id so all ranks build the same mapping.
    """

    def __init__(
        self,
        semantic_ids: Sequence[Sequence[int]],
        item_popularity: Optional[Union[Mapping[int, int], Sequence[int]]] = None,
        padding_item_id: int = -1,
    ) -> None:
        self.padding_item_id = padding_item_id
        self.num_items = len(semantic_ids)

        if item_popularity is None:
            popularity = defaultdict(int)
        elif isinstance(item_popularity, Mapping):
            popularity = defaultdict(int, item_popularity)
        else:
            popularity = defaultdict(
                int, {item_id: int(count) for item_id, count in enumerate(item_popularity)}
            )

        buckets: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        for item_id, sid in enumerate(semantic_ids):
            buckets[tuple(int(code) for code in sid)].append(item_id)

        self.sid_to_item: Dict[Tuple[int, ...], int] = {}
        for sid, item_ids in buckets.items():
            self.sid_to_item[sid] = min(
                item_ids, key=lambda item_id: (-popularity[item_id], item_id)
            )

        self.num_unique_sids = len(buckets)
        self.num_collision_items = sum(
            len(item_ids) for item_ids in buckets.values() if len(item_ids) > 1
        )
        self.max_collision_group = max((len(v) for v in buckets.values()), default=0)

    @classmethod
    def from_sequences(
        cls,
        semantic_ids: Sequence[Sequence[int]],
        sequences: Sequence[Sequence[int]],
        train_holdout: int = 2,
        padding_item_id: int = -1,
    ) -> "SemanticIdItemMapper":
        """Build a mapper using only each user's training prefix.

        The repository's leave-one-out protocol reserves the final two items
        for validation and test.  Excluding them here prevents the SID-to-item
        resolver from leaking evaluation targets through popularity counts.
        """
        popularity: Counter[int] = Counter()
        for sequence in sequences:
            end = max(len(sequence) - train_holdout, 0)
            popularity.update(int(item_id) for item_id in sequence[:end])
        return cls(semantic_ids, popularity, padding_item_id=padding_item_id)

    @property
    def collision_rate(self) -> float:
        if self.num_items == 0:
            return 0.0
        return (self.num_items - self.num_unique_sids) / self.num_items

    def map_predictions(
        self,
        predicted_semantic_ids: torch.Tensor,
        topk: Optional[int] = None,
        deduplicate: bool = True,
    ) -> torch.Tensor:
        """Map ``[B, K, C]`` SIDs to ranked ``[B, topk]`` item IDs."""
        if predicted_semantic_ids.ndim != 3:
            raise ValueError(
                "predicted_semantic_ids must have shape [batch, candidates, codes]"
            )

        batch_size, n_candidates, _ = predicted_semantic_ids.shape
        output_k = n_candidates if topk is None else topk
        if output_k < 1:
            raise ValueError("topk must be positive")

        rows = predicted_semantic_ids.detach().cpu().tolist()
        mapped_rows: List[List[int]] = []
        for row in rows:
            ranked_items: List[int] = []
            seen = set()
            for sid_values in row:
                item_id = self.sid_to_item.get(tuple(int(v) for v in sid_values))
                if item_id is None or (deduplicate and item_id in seen):
                    continue
                ranked_items.append(item_id)
                seen.add(item_id)
                if len(ranked_items) == output_k:
                    break
            mapped_rows.append(
                ranked_items + [self.padding_item_id] * (output_k - len(ranked_items))
            )
        return torch.tensor(
            mapped_rows, dtype=torch.long, device=predicted_semantic_ids.device
        ).reshape(batch_size, output_k)


class TopKAccumulator:
    """
    Accumulator for Recall@K and NDCG@K metrics.

    Supports either semantic-ID vectors or scalar item IDs represented as a
    length-one vector.  Headline recommender metrics should use item IDs.
    """
    def __init__(self, ks: List[int] = [1, 5, 10]):
        self.ks = ks
        self.reset()

    def reset(self) -> None:
        """Reset the accumulator."""
        self.total = 0
        self.recalls = defaultdict(float)
        self.ndcgs = defaultdict(float)

    def accumulate(self, actual: torch.Tensor, top_k: torch.Tensor) -> None:
        """
        Accumulate metrics for a batch.

        Args:
            actual: (B, id_dim) ground-truth IDs
            top_k: (B, K, id_dim) top-K predicted IDs
        """
        B = actual.size(0)
        K = top_k.size(1)

        # Check if actual matches any of top_k predictions (exact match on all sem_id positions)
        # actual: (B, D) -> (B, 1, D)
        # top_k: (B, K, D)
        matches = (actual.unsqueeze(1) == top_k).all(dim=-1)  # (B, K) - True if all positions match

        # Find rank of first match (0-indexed), or K if no match
        # matches: (B, K)
        match_found = matches.any(dim=1)  # (B,)
        first_match_rank = matches.float().argmax(dim=1)  # (B,) - rank of first True, or 0 if no True
        # Fix: argmax returns 0 if no match, so we need to handle this
        first_match_rank = torch.where(match_found, first_match_rank, torch.tensor(K, device=matches.device))

        for k in self.ks:
            # Recall@K: 1 if target in top-K, else 0
            # ``match_found`` is essential when callers request a K larger than
            # the available candidate list. A miss uses sentinel rank ``K``;
            # without this guard, five candidates evaluated at @10 would turn
            # every miss into a hit.
            effective_k = min(k, K)
            in_top_k = match_found & (first_match_rank < effective_k)
            recall = in_top_k.float().sum().item()
            self.recalls[k] += recall

            # NDCG@K: 1/log2(rank+2) if target in top-K, else 0
            # rank is 0-indexed, so DCG = 1/log2(rank+2)
            dcg = torch.where(
                in_top_k,
                1.0 / torch.log2(first_match_rank.float() + 2),
                torch.tensor(0.0, device=matches.device)
            )
            # IDCG = 1/log2(2) = 1 (best case: target at rank 0)
            ndcg = dcg.sum().item()
            self.ndcgs[k] += ndcg

        self.total += B

    def sync_across_processes(self, accelerator) -> None:
        """All-reduce the raw counts so reduce() returns global metrics.

        Without this, reduce() on a sharded eval dataloader yields this rank's
        shard only — and any is_best/early-stop decision derived from it can
        diverge across ranks. Collective op: must be called by every rank.
        """
        counts = [float(self.total)]
        for k in self.ks:
            counts.extend([self.recalls[k], self.ndcgs[k]])
        t = torch.tensor(counts, device=accelerator.device, dtype=torch.float32)
        vals = accelerator.reduce(t, reduction="sum").tolist()
        self.total = int(vals[0])
        for i, k in enumerate(self.ks):
            self.recalls[k] = vals[1 + 2 * i]
            self.ndcgs[k] = vals[2 + 2 * i]

    def reduce(self) -> Dict[str, float]:
        """Compute final metrics."""
        result = {}
        for k in self.ks:
            result[f"Recall@{k}"] = self.recalls[k] / self.total if self.total > 0 else 0
            result[f"NDCG@{k}"] = self.ndcgs[k] / self.total if self.total > 0 else 0
        return result
