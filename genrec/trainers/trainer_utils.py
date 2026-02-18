"""
Common utilities for trainers.
Provides shared functionality to reduce code duplication across trainers.
"""
import os
import torch
import wandb
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from accelerate import Accelerator


@dataclass
class TrainerConfig:
    """Common configuration for all trainers."""
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulate_every: int = 1
    num_warmup_steps: int = 500

    # Paths
    dataset_folder: str = "dataset/amazon"
    save_dir_root: str = "out/"

    # Accelerator
    split_batches: bool = True
    amp: bool = False
    mixed_precision_type: str = "fp16"

    # Wandb
    wandb_logging: bool = False
    wandb_project: str = "training"
    wandb_run_name: Optional[str] = None
    wandb_log_interval: int = 10

    # Checkpointing
    save_every_epoch: int = 10
    eval_valid_every_epoch: int = 5
    eval_test_every_epoch: int = 10
    do_eval: bool = True
    resume_from_checkpoint: Optional[str] = None


def setup_accelerator(
    split_batches: bool = True,
    gradient_accumulate_every: int = 1,
    amp: bool = False,
    mixed_precision_type: str = "fp16",
) -> Accelerator:
    """
    Setup and return an Accelerator instance.

    Args:
        split_batches: Whether to split batches across devices
        gradient_accumulate_every: Gradient accumulation steps
        amp: Whether to use automatic mixed precision
        mixed_precision_type: Type of mixed precision ("fp16" or "bf16")

    Returns:
        Configured Accelerator instance
    """
    return Accelerator(
        split_batches=split_batches,
        gradient_accumulation_steps=gradient_accumulate_every,
        mixed_precision=mixed_precision_type if amp else "no",
    )


def setup_wandb(
    project: str,
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    step_metrics: Optional[Dict[str, str]] = None,
) -> None:
    """
    Setup Weights & Biases logging.

    Args:
        project: WandB project name
        run_name: Run name (auto-generated if None)
        config: Configuration dict to log
        step_metrics: Dict mapping metric prefix to step metric name
                     e.g., {"train/*": "global_step", "eval/*": "epoch"}
    """
    wandb.login()
    wandb.init(
        project=project,
        name=run_name,
        config=config,
    )

    # Define step metrics for better visualization
    if step_metrics:
        for metric_pattern, step_metric in step_metrics.items():
            wandb.define_metric(metric_pattern, step_metric=step_metric)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    logger=None,
) -> None:
    """
    Save a training checkpoint.

    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Optional learning rate scheduler
        epoch: Current epoch number
        step: Current step number
        extra: Extra data to save in checkpoint
        logger: Optional logger for info message
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if step is not None:
        state["step"] = step
    if extra is not None:
        state.update(extra)

    torch.save(state, path)

    if logger:
        logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device("cpu"),
    logger=None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint to
        logger: Optional logger for info message

    Returns:
        Dict containing checkpoint data (epoch, step, etc.)
    """
    if logger:
        logger.info(f"Loading checkpoint from {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


def get_parameter_count(model: torch.nn.Module) -> int:
    """Get total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_training_info(
    logger,
    model: torch.nn.Module,
    train_dataloader,
    valid_dataloader=None,
    test_dataloader=None,
    device=None,
) -> None:
    """Log common training information."""
    num_params = get_parameter_count(model)

    logger.info(f"Train dataloader: {len(train_dataloader)} batches")
    if valid_dataloader:
        logger.info(f"Valid dataloader: {len(valid_dataloader)} batches")
    if test_dataloader:
        logger.info(f"Test dataloader: {len(test_dataloader)} batches")
    if device:
        logger.info(f"Device: {device}")
    logger.info(f"Trainable parameters: {num_params:,}")
