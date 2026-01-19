"""
Training scripts for GenRec models.

This module provides trainers for various recommendation models:

Trainers:
    - rqvae_trainer: RQVAE training with collision rate metrics
    - tiger_trainer: TIGER training with constrained decoding
    - lcrec_trainer: LCRec LLM fine-tuning with beam search
    - cobra_trainer: COBRA training with sparse+dense loss balancing
    - sasrec_trainer: SASRec training with Recall@K, NDCG@K
    - hstu_trainer: HSTU training with temporal bias support

Usage:
    python genrec/trainers/<trainer>.py config/<model>/amazon.gin
"""

__all__ = [
    "rqvae_trainer",
    "tiger_trainer",
    "lcrec_trainer",
    "cobra_trainer",
    "sasrec_trainer",
    "hstu_trainer",
]
