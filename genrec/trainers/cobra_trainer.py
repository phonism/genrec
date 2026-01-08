"""
Trainer for the COBRA model.
COBRA: Sparse Meets Dense - Unified Generative Recommendations with Cascaded Sparse-Dense Representations
"""
import argparse
import os
import gin
import torch
import wandb

from accelerate import Accelerator
from genrec.models.cobra import Cobra
from genrec.modules.utils import parse_config
from genrec.modules.metrics import TopKAccumulator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Any
from transformers.optimization import get_cosine_schedule_with_warmup


def cobra_collate_fn(
    batch: List[Dict[str, Any]],
    pad_id: int = 0,
    n_codebooks: int = 3,
    is_train: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Collate function for COBRA dataset.

    Args:
        batch: List of samples from AmazonCobraDataset
        pad_id: Padding ID for semantic IDs
        n_codebooks: Number of codebooks
        is_train: If True, append target to input for training. If False, keep separate for eval.
    Returns:
        Batched tensors
    """
    max_text_len = batch[0]['encoder_input_ids'].shape[-1]
    B = len(batch)

    if is_train:
        # Training: append target to input so model learns to predict it
        # input = history + target, predict e1, e2, ..., e_T (including target)
        max_items = max(len(x['input_ids']) // n_codebooks + 1 for x in batch)  # +1 for target

        input_ids = torch.full((B, max_items * n_codebooks), pad_id, dtype=torch.long)
        encoder_input_ids = torch.zeros((B, max_items, max_text_len), dtype=torch.long)
        target_sem_ids = torch.zeros((B, n_codebooks), dtype=torch.long)

        for i, sample in enumerate(batch):
            history_len = len(sample['input_ids'])
            n_history_items = history_len // n_codebooks

            # Append target to history semantic IDs
            full_sem_ids = sample['input_ids'] + sample['target_sem_ids']
            input_ids[i, :len(full_sem_ids)] = torch.tensor(full_sem_ids)

            # Append target encoder input to history
            encoder_input_ids[i, :n_history_items] = sample['encoder_input_ids']
            encoder_input_ids[i, n_history_items:n_history_items+1] = sample['target_encoder_input_ids']

            # Target (for reference, though not used in loss since it's in input_ids)
            target_sem_ids[i] = torch.tensor(sample['target_sem_ids'])
    else:
        # Eval: keep history and target separate
        max_items = max(len(x['input_ids']) // n_codebooks for x in batch)

        input_ids = torch.full((B, max_items * n_codebooks), pad_id, dtype=torch.long)
        encoder_input_ids = torch.zeros((B, max_items, max_text_len), dtype=torch.long)
        target_sem_ids = torch.zeros((B, n_codebooks), dtype=torch.long)

        for i, sample in enumerate(batch):
            seq_len = len(sample['input_ids'])
            n_items = seq_len // n_codebooks

            input_ids[i, :seq_len] = torch.tensor(sample['input_ids'])
            encoder_input_ids[i, :n_items] = sample['encoder_input_ids']
            target_sem_ids[i] = torch.tensor(sample['target_sem_ids'])

    return {
        'input_ids': input_ids,
        'encoder_input_ids': encoder_input_ids,
        'target_sem_ids': target_sem_ids,
    }


@gin.configurable
def train(
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    dataset_folder: str = "dataset/amazon",
    save_dir_root: str = "out/cobra/amazon/beauty",
    dataset=None,
    split_batches: bool = True,
    amp: bool = False,
    wandb_logging: bool = False,
    wandb_project: str = "cobra_training",
    wandb_run_name: str = None,  # Run name, auto-generated if None
    wandb_log_interval: int = 10,
    mixed_precision_type: str = "fp16",
    gradient_accumulate_every: int = 1,
    save_every_epoch: int = 10,
    eval_valid_every_epoch: int = 5,
    eval_test_every_epoch: int = 10,
    do_eval: bool = True,
    # Model architecture
    encoder_n_layers: int = 1,
    encoder_hidden_dim: int = 768,
    encoder_num_heads: int = 8,
    encoder_vocab_size: int = 32128,
    id_vocab_size: int = 256,
    n_codebooks: int = 3,
    d_model: int = 384,
    max_len: int = 1024,
    temperature: float = 0.2,
    queue_size: int = 1024,
    # Decoder params (aligned with TIGER)
    decoder_n_layers: int = 8,
    decoder_num_heads: int = 6,
    decoder_dropout: float = 0.1,
    # Encoder type: "light" (random init) or "pretrained" (sentence-t5)
    encoder_type: str = "light",
    # Training
    num_warmup_steps: int = 500,
    max_seq_len: int = 20,
    pretrained_rqvae_path: str = "./out/rqvae/amazon/beauty/checkpoint.pt",
    encoder_model_name: str = "/root/workspace/models_hub/sentence-t5-xl",
    resume_from_checkpoint: str = None,
    # Loss weights
    sparse_loss_weight: float = 1.0,
    dense_loss_weight: float = 1.0,
):
    """
    Trains a COBRA model.
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
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("eval/*", step_metric="epoch")

    # Create datasets
    train_dataset = dataset(
        root=dataset_folder,
        train_test_split="train",
        max_seq_len=max_seq_len,
        pretrained_rqvae_path=pretrained_rqvae_path,
        encoder_model_name=encoder_model_name,
    )

    valid_dataset = dataset(
        root=dataset_folder,
        train_test_split="valid",
        max_seq_len=max_seq_len,
        pretrained_rqvae_path=pretrained_rqvae_path,
        encoder_model_name=encoder_model_name,
    )

    test_dataset = dataset(
        root=dataset_folder,
        train_test_split="test",
        max_seq_len=max_seq_len,
        pretrained_rqvae_path=pretrained_rqvae_path,
        encoder_model_name=encoder_model_name,
    )

    pad_id = id_vocab_size * n_codebooks
    train_collate_fn = lambda x: cobra_collate_fn(x, pad_id=pad_id, n_codebooks=n_codebooks, is_train=True)
    eval_collate_fn = lambda x: cobra_collate_fn(x, pad_id=pad_id, n_codebooks=n_codebooks, is_train=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        collate_fn=train_collate_fn,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=eval_collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=eval_collate_fn,
    )

    train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, valid_dataloader, test_dataloader
    )

    print(f"train_dataloader: {len(train_dataloader)}")
    print(f"valid_dataloader: {len(valid_dataloader)}")
    print(f"test_dataloader: {len(test_dataloader)}")

    # Create model
    model = Cobra(
        encoder_n_layers=encoder_n_layers,
        encoder_hidden_dim=encoder_hidden_dim,
        encoder_num_heads=encoder_num_heads,
        encoder_vocab_size=encoder_vocab_size,
        id_vocab_size=id_vocab_size,
        n_codebooks=n_codebooks,
        d_model=d_model,
        max_len=max_len,
        temperature=temperature,
        queue_size=queue_size,
        decoder_n_layers=decoder_n_layers,
        decoder_num_heads=decoder_num_heads,
        decoder_dropout=decoder_dropout,
        encoder_type=encoder_type,
        encoder_model_name=encoder_model_name,
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
    print(f"Device: {device}, Num Parameters: {num_params:,}")

    if accelerator.is_main_process:
        pbar = tqdm(total=total_steps, dynamic_ncols=True)

    # Resume from checkpoint
    start_epoch = 0
    if resume_from_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {checkpoint['epoch']}, starting at epoch {start_epoch}")

    def save_checkpoint(epoch, path):
        """Save checkpoint in dict format."""
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)
        print(f"Saved checkpoint to {path}")

    # Training loop
    model.train()
    global_step = -1

    for epoch in range(start_epoch, epochs):
        # Reset epoch accumulators
        epoch_acc_correct = 0
        epoch_acc_total = 0
        epoch_recall_correct = 0
        epoch_recall_total = 0

        for step, data in enumerate(train_dataloader):
            global_step += 1
            model.train()

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    output = model(
                        input_ids=data["input_ids"].to(device),
                        encoder_input_ids=data["encoder_input_ids"].to(device),
                    )

                    # Weighted loss
                    loss = (
                        sparse_loss_weight * output.loss_sparse +
                        dense_loss_weight * output.loss_dense
                    )

                accelerator.backward(loss)

                # Accumulate metrics
                epoch_acc_correct += output.acc_correct.item()
                epoch_acc_total += output.acc_total.item()
                epoch_recall_correct += output.recall_correct.item()
                epoch_recall_total += output.recall_total.item()

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    epoch_acc = epoch_acc_correct / max(epoch_acc_total, 1)
                    epoch_recall = epoch_recall_correct / max(epoch_recall_total, 1)

                    if accelerator.is_main_process:
                        pbar.set_description(
                            f'Epoch {epoch} | loss:{loss.item():.2f} sparse:{output.loss_sparse.item():.2f} dense:{output.loss_dense.item():.2f} | '
                            f'acc:{epoch_acc:.4f} recall:{epoch_recall:.4f}'
                        )
                        pbar.update(1)

                    if wandb_logging and accelerator.is_main_process and global_step % wandb_log_interval == 0:
                        wandb.log({
                            "global_step": global_step,
                            "train/learning_rate": lr_scheduler.get_last_lr()[0],
                            "train/loss": loss.item(),
                            "train/loss_sparse": output.loss_sparse.item(),
                            "train/loss_dense": output.loss_dense.item(),
                            "train/acc": epoch_acc,
                            "train/recall": epoch_recall,
                            "train/vec_cos_sim": output.vec_cos_sim.item(),
                            "train/codebook_entropy": output.codebook_entropy.item(),
                        })

            accelerator.wait_for_everyone()

        # End of epoch logging
        epoch_acc = epoch_acc_correct / max(epoch_acc_total, 1)
        epoch_recall = epoch_recall_correct / max(epoch_recall_total, 1)
        print(f"\nEpoch {epoch} - acc: {epoch_acc:.4f}, recall: {epoch_recall:.4f}")

        log_dict = {"epoch": epoch} if wandb_logging else None
        if wandb_logging and accelerator.is_main_process:
            log_dict["train/epoch_acc"] = epoch_acc
            log_dict["train/epoch_recall"] = epoch_recall

        # Evaluation with Recall@K and NDCG@K
        if do_eval and (epoch + 1) % eval_valid_every_epoch == 0:
            model.eval()
            metrics_accumulator = TopKAccumulator()

            # Per-codebook accuracy tracking
            codebook_correct = [0, 0, 0]
            codebook_total = 0

            with torch.no_grad():
                for data in valid_dataloader:
                    # Generate predictions
                    generated = model.generate(
                        input_ids=data["input_ids"].to(device),
                        encoder_input_ids=data["encoder_input_ids"].to(device),
                        n_candidates=10,
                    )
                    # Compare with target
                    target = data["target_sem_ids"].to(device)  # (B, C)
                    topk = generated.sem_ids  # (B, K, C)
                    metrics_accumulator.accumulate(actual=target, top_k=topk)

                    # Per-codebook accuracy (top-1 only)
                    top1 = topk[:, 0, :]  # (B, C)
                    for c in range(3):
                        codebook_correct[c] += (top1[:, c] == target[:, c]).sum().item()
                    codebook_total += target.size(0)

            metrics = metrics_accumulator.reduce()
            # Print per-codebook accuracy
            c0_acc = codebook_correct[0] / max(codebook_total, 1)
            c1_acc = codebook_correct[1] / max(codebook_total, 1)
            c2_acc = codebook_correct[2] / max(codebook_total, 1)
            print(f"Epoch {epoch} - Valid: {metrics}")
            print(f"  Per-codebook acc: c0={c0_acc:.4f}, c1={c1_acc:.4f}, c2={c2_acc:.4f}")

            if wandb_logging and accelerator.is_main_process:
                for k, v in metrics.items():
                    log_dict[f"eval/valid_{k}"] = v

        if wandb_logging and accelerator.is_main_process and log_dict and len(log_dict) > 1:
            wandb.log(log_dict)

        model.train()

        # Save checkpoint
        if accelerator.is_main_process and (epoch + 1) % save_every_epoch == 0:
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
