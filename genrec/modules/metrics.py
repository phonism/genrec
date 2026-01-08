"""
Evaluation metrics for recommendation.
"""
import torch
import math
from collections import defaultdict
from typing import Dict, List


class TopKAccumulator:
    """
    Accumulator for Recall@K and NDCG@K metrics.

    Matches the TIGER paper evaluation: Recall@5, Recall@10, NDCG@5, NDCG@10
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
            actual: (B, sem_id_dim) ground truth semantic IDs
            top_k: (B, K, sem_id_dim) top-K predicted semantic IDs
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
            recall = (first_match_rank < k).float().sum().item()
            self.recalls[k] += recall

            # NDCG@K: 1/log2(rank+2) if target in top-K, else 0
            # rank is 0-indexed, so DCG = 1/log2(rank+2)
            in_top_k = first_match_rank < k
            dcg = torch.where(
                in_top_k,
                1.0 / torch.log2(first_match_rank.float() + 2),
                torch.tensor(0.0, device=matches.device)
            )
            # IDCG = 1/log2(2) = 1 (best case: target at rank 0)
            ndcg = dcg.sum().item()
            self.ndcgs[k] += ndcg

        self.total += B

    def reduce(self) -> Dict[str, float]:
        """Compute final metrics."""
        result = {}
        for k in self.ks:
            result[f"Recall@{k}"] = self.recalls[k] / self.total if self.total > 0 else 0
            result[f"NDCG@{k}"] = self.ndcgs[k] / self.total if self.total > 0 else 0
        return result
