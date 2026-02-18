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

Utilities:
    - trainer_utils: Common functions (setup_accelerator, setup_wandb, etc.)

Usage:
    python genrec/trainers/<trainer>.py config/<model>/amazon.gin
"""

from genrec.trainers.trainer_utils import (
    TrainerConfig,
    setup_accelerator,
    setup_wandb,
    save_checkpoint,
    load_checkpoint,
    get_parameter_count,
    log_training_info,
)

__all__ = [
    "rqvae_trainer",
    "tiger_trainer",
    "lcrec_trainer",
    "cobra_trainer",
    "sasrec_trainer",
    "hstu_trainer",
    # Utilities
    "TrainerConfig",
    "setup_accelerator",
    "setup_wandb",
    "save_checkpoint",
    "load_checkpoint",
    "get_parameter_count",
    "log_training_info",
]
