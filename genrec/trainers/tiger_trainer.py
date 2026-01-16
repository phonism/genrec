"""
Trainer for the TIGER model - Generative Retrieval for Sequential Recommendation.
"""
import argparse
import logging
import os
import gin
import torch
import wandb

from accelerate import Accelerator

logger = logging.getLogger(__name__)
from genrec.data.utils import cycle
from genrec.models.tiger import Tiger
from genrec.models.rqvae import RqVae
from genrec.modules.utils import parse_config
from genrec.data.schemas import SeqData
from genrec.modules.metrics import TopKAccumulator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any
from transformers.optimization import get_cosine_schedule_with_warmup

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
    max_item_length = max(len(x.item_ids) for x in batch)
    user_ids = torch.full((len(batch), 1), pad_id, dtype=torch.long, device=device)
    ids = torch.full((len(batch), max_item_length), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(batch), max_item_length), dtype=torch.long, device=device)
    token_type_ids = torch.zeros((len(batch), max_item_length), dtype=torch.long, device=device)
    target_input_ids = torch.full((len(batch), len(batch[0].target_ids)), pad_id, dtype=torch.long, device=device)
    target_token_type_ids = torch.zeros((len(batch), len(batch[0].target_ids)), dtype=torch.long, device=device)

    B = len(batch)
    for i in range(B):
        
        uid = batch[i].user_id
        item_ids = batch[i].item_ids
        user_ids[i, 0] = uid
        if padding_side == "left":
            ids[i, :len(item_ids)] = torch.tensor(item_ids)
            token_type_ids[i, :len(item_ids)] = torch.arange(len(item_ids), device=ids.device) % 3
            mask[i, :len(item_ids)] = 1
            target_input_ids[i, :] = torch.tensor(batch[i].target_ids)
            target_token_type_ids[i, :] = torch.arange(len(batch[i].target_ids), device=target_input_ids.device)
        else:
            ids[i, max_item_length - len(item_ids):] = torch.tensor(item_ids)
            token_type_ids[i, max_item_length - len(item_ids):] = torch.arange(len(item_ids), device=ids.device) % 3
            mask[i, max_item_length - len(item_ids):] = 1
            target_input_ids[i, :] = torch.tensor(batch[i].target_ids)
            target_token_type_ids[i, :] = torch.arange(len(batch[i].target_ids), device=target_input_ids.device)

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
): 
    """
    Trains a Tiger model.
    """
    if wandb_logging:
        params = locals()

    accelerator = Accelerator(
        split_batches=split_batches,
        gradient_accumulation_steps=gradient_accumulate_every,
        mixed_precision=mixed_precision_type if amp else "no",
    )

    device = accelerator.device

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=params
        )
        # Define separate sections for train and eval metrics
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("eval/*", step_metric="epoch")

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
    )
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_steps = len(train_dataloader) * epochs // gradient_accumulate_every
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )

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
        lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resumed from epoch {checkpoint['epoch']}, starting at epoch {start_epoch}")

    def save_checkpoint(epoch, path):
        """Save checkpoint in dict format."""
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
        }
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
                    valid_item_ids=valid_item_ids.to(device)
                )
                actual = data["target_input_ids"].to(device)
                topk = generated.sem_ids
                metrics_accumulator.accumulate(actual=actual, top_k=topk)
        metrics = metrics_accumulator.reduce()
        metrics_accumulator.reset()
        return metrics

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
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    if accelerator.is_main_process:
                        pbar.set_description(f'Epoch {epoch}/{epochs} | loss: {loss.item():.4f}')
                        pbar.update(1)
                    if wandb_logging and accelerator.is_main_process and global_step % wandb_log_interval == 0:
                        wandb.log({
                            "global_step": global_step,
                            "train/learning_rate": lr_scheduler.get_last_lr()[0],
                            "train/loss": total_loss / gradient_accumulate_every,
                        })
                    total_loss = 0

            accelerator.wait_for_everyone()

        # End of epoch - evaluate and save
        log_dict = {"epoch": epoch} if wandb_logging else None

        # Valid evaluation
        if do_eval and (epoch + 1) % eval_valid_every_epoch == 0:
            valid_metrics = evaluate(valid_dataloader, desc=f"Valid Eval (Epoch {epoch})")
            logger.info(f"Epoch {epoch} - Valid: {valid_metrics}")
            if wandb_logging and accelerator.is_main_process:
                for k in valid_metrics:
                    log_dict[f"eval/valid_{k}"] = valid_metrics[k]

        # Test evaluation (less frequent)
        if do_eval and (epoch + 1) % eval_test_every_epoch == 0:
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

    # Save final checkpoint
    if accelerator.is_main_process:
        save_checkpoint(
            epochs - 1,
            os.path.join(save_dir_root, "checkpoint_final.pt")
        )

    if wandb_logging and accelerator.is_main_process:
        wandb.finish()
    
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parse_config()
    train()
