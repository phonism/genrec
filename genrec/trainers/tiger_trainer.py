"""
Trainer for the TIGER model - Generative Retrieval for Sequential Recommendation.
"""
import os
import gin
import torch
import wandb

from genrec.data.utils import cycle
from genrec.models.tiger import Tiger
from genrec.models.rqvae import RqVae
from genrec.modules.utils import parse_config, setup_logger
from genrec.data.schemas import SeqData
from genrec.modules.metrics import TopKAccumulator
from genrec.trainers.trainer_utils import setup_accelerator, setup_wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any
from transformers.optimization import get_cosine_schedule_with_warmup
from genrec.modules.scheduler import InverseSquareRootScheduler

def pad_collate(
    batch: List[SeqData],
    pad_id: int = 0,
    device: torch.device = torch.device("cpu"),
    padding_side: str = "left"
) -> Dict[str, torch.Tensor]:
    """
    Every sample has the following fields:
        "ids": List[List[int]]  (L_i, 3)                # sparse ids
        "vec_inputs": Dict[str, List[int]]  (L_i, V_ij) # dense ids
        "lengths": List[int]  (L_i)                     # original length (optional)

    Args:
        batch: List[Dict[str, Any]]
        pad_id: int
        device: torch.device
    Returns:
        Dict[str, torch.Tensor]
    """
    B = len(batch)
    max_item_length = max(len(x.item_ids) for x in batch)
    target_len = len(batch[0].target_ids)

    user_ids = torch.full((B, 1), pad_id, dtype=torch.long, device=device)
    ids = torch.full((B, max_item_length), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((B, max_item_length), dtype=torch.long, device=device)
    token_type_ids = torch.zeros((B, max_item_length), dtype=torch.long, device=device)

    # Build target tensors batch-wise (avoid per-sample torch.arange)
    target_input_ids = torch.tensor(
        [x.target_ids for x in batch], dtype=torch.long, device=device
    )
    target_token_type_ids = torch.arange(
        target_len, dtype=torch.long, device=device
    ).unsqueeze(0).expand(B, -1).clone()

    for i in range(B):
        uid = batch[i].user_id
        item_ids = batch[i].item_ids
        L = len(item_ids)
        user_ids[i, 0] = uid
        item_t = torch.tensor(item_ids)
        type_t = torch.arange(L) % 3
        if padding_side == "left":
            ids[i, :L] = item_t
            token_type_ids[i, :L] = type_t
            mask[i, :L] = 1
        else:
            ids[i, max_item_length - L:] = item_t
            token_type_ids[i, max_item_length - L:] = type_t
            mask[i, max_item_length - L:] = 1

    return {
        "user_input_ids": user_ids,
        "item_input_ids": ids,
        "token_type_ids": token_type_ids,
        "target_input_ids": target_input_ids,
        "target_token_type_ids": target_token_type_ids,
        "seq_mask": mask,
    }


@gin.configurable
def train(
    epochs=1,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.01,
    dataset_folder="dataset/query",
    save_dir_root="out/",
    dataset=None,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    wandb_project="Training",
    wandb_run_name=None,  # Run name, auto-generated if None
    wandb_log_interval=10,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    save_every_epoch=100,
    eval_valid_every_epoch=10,
    eval_test_every_epoch=50,
    do_eval=True,
    embedding_dim=128,
    attn_dim=256,
    dropout=0.1,
    num_heads=8,
    n_layers=2,
    num_item_embeddings=256,
    num_user_embeddings=10000,
    num_warmup_steps=1000,
    sem_id_dim=3,
    max_seq_len=2048,
    pretrained_rqvae_path="./out/rqvae/p5_amazon/beauty/checkpoint_299999.pt",
    resume_from_checkpoint=None,
    lr_schedule="cosine",
    beam_size=10,
    early_stop_patience=0,  # 0 = disabled
    d_kv=0,  # T5-style d_kv (0 = legacy mode)
    dim_feedforward=1024,
    share_position_bias=False,
    scale_attn=True,
    add_final_norm=False,
    decoder_bidirectional=True,
):
    """
    Trains a Tiger model.
    """
    # Setup logger
    logger = setup_logger(save_dir_root, name="tiger")

    accelerator = setup_accelerator(
        split_batches=split_batches,
        gradient_accumulate_every=gradient_accumulate_every,
        amp=amp,
        mixed_precision_type=mixed_precision_type,
    )
    device = accelerator.device

    if wandb_logging and accelerator.is_main_process:
        setup_wandb(
            project=wandb_project,
            run_name=wandb_run_name,
            config=locals(),
            step_metrics={"train/*": "global_step", "eval/*": "epoch"}
        )

    train_dataset = dataset(
        root=dataset_folder,
        train_test_split="train",
        max_seq_len=max_seq_len,
        subsample=True,
        pretrained_rqvae_path=pretrained_rqvae_path
    )

    valid_dataset = dataset(
        root=dataset_folder,
        train_test_split="valid",
        max_seq_len=max_seq_len,
        subsample=False,
        pretrained_rqvae_path=pretrained_rqvae_path
    )

    test_dataset = dataset(
        root=dataset_folder,
        train_test_split="test",
        max_seq_len=max_seq_len,
        subsample=False,
        pretrained_rqvae_path=pretrained_rqvae_path
    )

    collate_fn = lambda x: pad_collate(x, pad_id=num_item_embeddings * sem_id_dim)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=10,
        collate_fn=collate_fn)

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=10,
        collate_fn=collate_fn)

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=10,
        collate_fn=collate_fn)

    train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, valid_dataloader, test_dataloader
    )

    logger.info(f"train_dataloader: {len(train_dataloader)}")
    logger.info(f"valid_dataloader: {len(valid_dataloader)}")
    logger.info(f"test_dataloader: {len(test_dataloader)}")

    model = Tiger(
        embedding_dim=embedding_dim,
        attn_dim=attn_dim,
        dropout=dropout,
        num_heads=num_heads,
        n_layers=n_layers,
        num_item_embeddings=num_item_embeddings,
        num_user_embeddings=num_user_embeddings,
        sem_id_dim=sem_id_dim,
        max_pos=max_seq_len * sem_id_dim,
        d_kv=d_kv,
        dim_feedforward=dim_feedforward,
        share_position_bias=share_position_bias,
        scale_attn=scale_attn,
        add_final_norm=add_final_norm,
        decoder_bidirectional=decoder_bidirectional,
    )
    
    if lr_schedule == "none":
        from torch.optim import Adam
        optimizer = Adam(model.parameters(), lr=learning_rate)
        logger.info("Using plain Adam, no scheduler")
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    total_steps = len(train_dataloader) * epochs // gradient_accumulate_every
    lr_scheduler = None
    if lr_schedule == "inverse_sqrt":
        lr_scheduler = InverseSquareRootScheduler(
            optimizer,
            warmup_steps=num_warmup_steps,
        )
    elif lr_schedule == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )

    if lr_scheduler is not None:
        model, optimizer, lr_scheduler = accelerator.prepare(
            model, optimizer, lr_scheduler
        )
    else:
        model, optimizer = accelerator.prepare(model, optimizer)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Device: {device}, Num Parameters: {num_params}")

    if accelerator.is_main_process:
        pbar = tqdm(total=total_steps, dynamic_ncols=True)

    metrics_accumulator = TopKAccumulator(ks=[5, 10])

    valid_item_ids = torch.tensor(
        list(train_dataset.sem_ids_list), dtype=torch.long, device=device
    )
    print("valid_item_ids length=", len(valid_item_ids), 
        "unique=", len(set([tuple(x) for x in valid_item_ids.tolist()])))

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from_checkpoint is not None:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lr_scheduler is not None and "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resumed from epoch {checkpoint['epoch']}, starting at epoch {start_epoch}")

    def save_checkpoint(epoch, path):
        """Save checkpoint in dict format."""
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if lr_scheduler is not None:
            state["scheduler"] = lr_scheduler.state_dict()
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path}")

    def evaluate(dataloader, desc="Evaluation"):
        """Evaluate model on a dataloader."""
        model.eval()
        for data in tqdm(dataloader, desc=desc):
            with torch.inference_mode():
                generated = model.generate(
                    user_input_ids=data["user_input_ids"].to(device),
                    item_input_ids=data["item_input_ids"].to(device),
                    token_type_ids=data["token_type_ids"].to(device),
                    seq_mask=data["seq_mask"].to(device),
                    valid_item_ids=valid_item_ids,
                    n_top_k_candidates=beam_size,
                )
                actual = data["target_input_ids"].to(device)
                topk = generated.sem_ids
                metrics_accumulator.accumulate(actual=actual, top_k=topk)
        metrics = metrics_accumulator.reduce()
        metrics_accumulator.reset()
        return metrics

    # Early stopping state
    best_valid_recall10 = 0.0
    best_epoch = -1
    early_stop_counter = 0
    use_early_stopping = early_stop_patience > 0

    total_loss = 0
    model.train()
    global_step = -1
    for epoch in range(start_epoch, epochs):
        for step, data in enumerate(train_dataloader):
            global_step += 1
            model.train()
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    output = model(
                        user_input_ids=data["user_input_ids"].to(device),
                        item_input_ids=data["item_input_ids"].to(device),
                        token_type_ids=data["token_type_ids"].to(device),
                        target_input_ids=data["target_input_ids"].to(device),
                        target_token_type_ids=data["target_token_type_ids"].to(device),
                        seq_mask=data["seq_mask"].to(device),
                    )
                    loss = output.loss
                    total_loss += loss.item()

                accelerator.backward(loss)

                # sync gradients means it's time to update
                if accelerator.sync_gradients:
                    # only clip when sync gradients
                    total_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    optimizer.zero_grad()

                    if accelerator.is_main_process:
                        pbar.set_description(f'Epoch {epoch}/{epochs} | loss: {loss.item():.4f}')
                        pbar.update(1)
                    if wandb_logging and accelerator.is_main_process and global_step % wandb_log_interval == 0:
                        cur_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else learning_rate
                        wandb.log({
                            "global_step": global_step,
                            "train/learning_rate": cur_lr,
                            "train/loss": total_loss / gradient_accumulate_every,
                        })
                    total_loss = 0

            accelerator.wait_for_everyone()

        # End of epoch - evaluate and save
        log_dict = {"epoch": epoch} if wandb_logging else None

        # Valid evaluation: every epoch if early stopping, else per config
        should_eval_valid = use_early_stopping or ((epoch + 1) % eval_valid_every_epoch == 0)
        if do_eval and should_eval_valid:
            valid_metrics = evaluate(valid_dataloader, desc=f"Valid Eval (Epoch {epoch})")
            logger.info(f"Epoch {epoch} - Valid: {valid_metrics}")
            if wandb_logging and accelerator.is_main_process:
                for k in valid_metrics:
                    log_dict[f"eval/valid_{k}"] = valid_metrics[k]

            # Early stopping check
            if use_early_stopping:
                cur_recall10 = valid_metrics.get("Recall@10", 0.0)
                if cur_recall10 > best_valid_recall10:
                    best_valid_recall10 = cur_recall10
                    best_epoch = epoch
                    early_stop_counter = 0
                    # Save best model
                    if accelerator.is_main_process:
                        save_checkpoint(epoch, os.path.join(save_dir_root, "best_model.pt"))
                        logger.info(f"New best Valid R@10={cur_recall10:.4f} at epoch {epoch}")
                    # Test evaluation on improvement
                    if do_eval:
                        test_metrics = evaluate(test_dataloader, desc=f"Test Eval (Epoch {epoch})")
                        logger.info(f"Epoch {epoch} - Test: {test_metrics}")
                        if wandb_logging and accelerator.is_main_process:
                            for k in test_metrics:
                                log_dict[f"eval/test_{k}"] = test_metrics[k]
                else:
                    early_stop_counter += 1
                    logger.info(f"No improvement. Counter: {early_stop_counter}/{early_stop_patience}")

        # Test evaluation (less frequent, only when not already done by early stopping)
        if do_eval and not use_early_stopping and (epoch + 1) % eval_test_every_epoch == 0:
            test_metrics = evaluate(test_dataloader, desc=f"Test Eval (Epoch {epoch})")
            logger.info(f"Epoch {epoch} - Test: {test_metrics}")
            if wandb_logging and accelerator.is_main_process:
                for k in test_metrics:
                    log_dict[f"eval/test_{k}"] = test_metrics[k]

        # Periodic test eval even with early stopping (when no improvement)
        if do_eval and use_early_stopping and early_stop_counter > 0 and (epoch + 1) % eval_test_every_epoch == 0:
            test_metrics = evaluate(test_dataloader, desc=f"Test Eval (Epoch {epoch})")
            logger.info(f"Epoch {epoch} - Test: {test_metrics}")
            if wandb_logging and accelerator.is_main_process:
                for k in test_metrics:
                    log_dict[f"eval/test_{k}"] = test_metrics[k]

        # Log to wandb
        if wandb_logging and accelerator.is_main_process and len(log_dict) > 1:
            wandb.log(log_dict)
        model.train()

        # Epoch-based checkpoint saving
        if accelerator.is_main_process:
            if (epoch + 1) % save_every_epoch == 0:
                save_checkpoint(
                    epoch,
                    os.path.join(save_dir_root, f"checkpoint_epoch_{epoch}.pt")
                )

        # Early stopping break
        if use_early_stopping and early_stop_counter >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}, Best R@10: {best_valid_recall10:.4f}")
            break

    # Save final checkpoint
    if accelerator.is_main_process:
        save_checkpoint(
            epoch,
            os.path.join(save_dir_root, "checkpoint_final.pt")
        )

    if use_early_stopping:
        logger.info(f"Training done. Best Valid R@10={best_valid_recall10:.4f} at epoch {best_epoch}")

    if wandb_logging and accelerator.is_main_process:
        wandb.finish()

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parse_config()
    train()
