"""
TIGER trainer (HF T5-based).

Pipeline:
- Flat offset-encoded codebook tokens (token = code + codebook_idx * codebook_size + 1).
- HF T5ForConditionalGeneration via genrec.models.tiger.Tiger.
- Plain Adam (default) or cosine schedule with linear warmup.
- Early stopping on validation Recall@10.
"""

import os
import gin
import torch
import wandb

from genrec.models.tiger import Tiger
from genrec.modules.utils import parse_config, setup_logger
from genrec.data.schemas import SeqData
from genrec.trainers.trainer_utils import setup_wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict


def t5_collate(
    batch: List[SeqData],
    codebook_size: int = 256,
    sem_id_dim: int = 3,
    max_history_tokens: int = 60,  # max_seq_len * sem_id_dim (e.g. 20*3=60)
) -> Dict[str, torch.Tensor]:
    """
    Convert SeqData to flat T5 format with offset encoding.

    Offset encoding: token = code + codebook_idx * codebook_size + 1
    (0 is reserved for padding)
    """
    B = len(batch)

    input_ids = torch.zeros(B, max_history_tokens, dtype=torch.long)
    attention_mask = torch.zeros(B, max_history_tokens, dtype=torch.long)
    label_list = []

    for i, sample in enumerate(batch):
        raw_ids = sample.item_ids  # flat list: [c0, c1, c2, c0, c1, c2, ...]
        offset_ids = []
        for j, code in enumerate(raw_ids):
            cb_idx = j % sem_id_dim
            offset_ids.append(code + cb_idx * codebook_size + 1)

        L = len(offset_ids)
        if L > max_history_tokens:
            offset_ids = offset_ids[-max_history_tokens:]
            L = max_history_tokens

        start = max_history_tokens - L
        input_ids[i, start:] = torch.tensor(offset_ids, dtype=torch.long)
        attention_mask[i, start:] = 1

        target_ids = sample.target_ids
        offset_target = []
        for j, code in enumerate(target_ids):
            cb_idx = j % sem_id_dim
            offset_target.append(code + cb_idx * codebook_size + 1)
        label_list.append(offset_target)

    labels = torch.tensor(label_list, dtype=torch.long)
    raw_targets = torch.tensor([s.target_ids for s in batch], dtype=torch.long)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'raw_targets': raw_targets,
    }


def calculate_pos_index(preds, labels, maxk=20):
    """Calculate position index of ground truth in predictions (vectorized)."""
    labels_expanded = labels.unsqueeze(1).expand_as(preds)  # (B, maxk, seq_len)
    match = (preds == labels_expanded).all(dim=-1)  # (B, maxk)
    first_match_mask = match.int().cumsum(dim=1) == 1
    return match & first_match_mask


def recall_at_k(pos_index, k):
    return pos_index[:, :k].sum(dim=1).float().mean().item()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1).float()
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0))
    return dcg[:, :k].sum(dim=1).mean().item()


@gin.configurable
def train(
    epochs=200,
    batch_size=256,
    infer_batch_size=96,
    learning_rate=1e-4,
    weight_decay=0.0,
    dataset_folder="dataset/amazon",
    save_dir_root="out/tiger/",
    dataset=None,
    wandb_logging=True,
    wandb_project="tiger_training",
    wandb_log_interval=10,
    # Model config
    num_layers=4,
    num_decoder_layers=4,
    d_model=128,
    d_ff=1024,
    num_heads=6,
    d_kv=64,
    dropout_rate=0.1,
    feed_forward_proj='relu',
    # Data config
    codebook_size=256,
    sem_id_dim=3,
    max_seq_len=20,
    pretrained_rqvae_path="./out/tiger/amazon/{split}/rqvae/checkpoint_epoch_4999.pt",
    # Scheduler config
    lr_schedule="none",  # "none" or "cosine"
    num_warmup_steps=0,
    # Eval config
    beam_size=30,
    early_stop_patience=10,
    eval_test_every_epoch=2,
    eval_valid_every_n_epochs=1,
):
    logger = setup_logger(save_dir_root, name="tiger")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = codebook_size * sem_id_dim + 1

    model_config = {
        'num_layers': num_layers,
        'num_decoder_layers': num_decoder_layers,
        'd_model': d_model,
        'd_ff': d_ff,
        'num_heads': num_heads,
        'd_kv': d_kv,
        'dropout_rate': dropout_rate,
        'vocab_size': vocab_size,
        'pad_token_id': 0,
        'eos_token_id': 0,
        'feed_forward_proj': feed_forward_proj,
        'sem_id_dim': sem_id_dim,
    }

    if wandb_logging:
        setup_wandb(
            project=wandb_project,
            config={**model_config, 'lr': learning_rate, 'batch_size': batch_size,
                    'max_seq_len': max_seq_len, 'beam_size': beam_size},
        )

    # Load datasets
    train_dataset = dataset(
        root=dataset_folder,
        train_test_split="train",
        max_seq_len=max_seq_len,
        pretrained_rqvae_path=pretrained_rqvae_path,
    )
    valid_dataset = dataset(
        root=dataset_folder,
        train_test_split="valid",
        max_seq_len=max_seq_len,
        pretrained_rqvae_path=pretrained_rqvae_path,
    )
    test_dataset = dataset(
        root=dataset_folder,
        train_test_split="test",
        max_seq_len=max_seq_len,
        pretrained_rqvae_path=pretrained_rqvae_path,
    )

    max_history_tokens = max_seq_len * sem_id_dim
    collate_fn = lambda x: t5_collate(x, codebook_size, sem_id_dim, max_history_tokens)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=infer_batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=infer_batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )

    logger.info(f"Train: {len(train_loader)} batches, Valid: {len(valid_loader)}, Test: {len(test_loader)}")

    # Model
    model = Tiger(model_config).to(device)
    total_params, emb_params = model.num_parameters
    logger.info(f"Device: {device}, Params: {total_params:,} (emb: {emb_params:,})")

    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # LR scheduler
    scheduler = None
    if lr_schedule == "cosine" and epochs > 0:
        total_steps = len(train_loader) * epochs
        if num_warmup_steps > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=num_warmup_steps)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - num_warmup_steps)
            scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[num_warmup_steps])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        logger.info(f"Using cosine schedule, warmup={num_warmup_steps}, total_steps={total_steps}")
    else:
        logger.info("Using plain Adam, no scheduler")

    # Early stopping state
    best_valid_recall10 = 0.0
    best_epoch = -1
    early_stop_counter = 0

    def evaluate(loader, desc="Eval"):
        model.eval()
        all_recalls = {5: [], 10: []}
        all_ndcgs = {5: [], 10: []}

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for batch in tqdm(loader, desc=desc):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']  # offset-encoded targets
                B_cur = input_ids.size(0)

                preds = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=beam_size,
                )
                # Remove decoder_start_token (first token)
                preds = preds[:, 1:]
                # Reshape: (B * beam_size, sem_id_dim) -> (B, beam_size, sem_id_dim)
                preds = preds.reshape(B_cur, beam_size, -1).cpu()

                pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
                for k in [5, 10]:
                    all_recalls[k].append(recall_at_k(pos_index, k))
                    all_ndcgs[k].append(ndcg_at_k(pos_index, k))

        metrics = {}
        for k in [5, 10]:
            metrics[f'Recall@{k}'] = sum(all_recalls[k]) / len(all_recalls[k])
            metrics[f'NDCG@{k}'] = sum(all_ndcgs[k]) / len(all_ndcgs[k])
        model.train()
        return metrics

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch} - loss: {avg_loss:.4f}")

        log_dict = {"epoch": epoch, "train/loss": avg_loss}

        # Valid evaluation (skip non-eval epochs)
        if (epoch + 1) % eval_valid_every_n_epochs == 0 or epoch == 0:
            valid_metrics = evaluate(valid_loader, desc=f"Valid (Epoch {epoch})")
            logger.info(f"Epoch {epoch} - Valid: {valid_metrics}")

            for k, v in valid_metrics.items():
                log_dict[f"eval/valid_{k}"] = v

            cur_recall10 = valid_metrics['Recall@10']

            # Check improvement
            if cur_recall10 > best_valid_recall10:
                best_valid_recall10 = cur_recall10
                best_epoch = epoch
                early_stop_counter = 0

                # Save best model
                os.makedirs(save_dir_root, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir_root, "best_model.pt"))
                logger.info(f"New best Valid R@10={cur_recall10:.4f} at epoch {epoch}")

                # Test evaluation on improvement
                test_metrics = evaluate(test_loader, desc=f"Test (Epoch {epoch})")
                logger.info(f"Epoch {epoch} - Test: {test_metrics}")
                for k, v in test_metrics.items():
                    log_dict[f"eval/test_{k}"] = v
            else:
                early_stop_counter += 1
                logger.info(f"No improvement. Counter: {early_stop_counter}/{early_stop_patience}")

                # Still run test periodically
                if (epoch + 1) % eval_test_every_epoch == 0:
                    test_metrics = evaluate(test_loader, desc=f"Test (Epoch {epoch})")
                    logger.info(f"Epoch {epoch} - Test: {test_metrics}")
                    for k, v in test_metrics.items():
                        log_dict[f"eval/test_{k}"] = v

        if wandb_logging:
            wandb.log(log_dict)

        if early_stop_counter >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    logger.info(f"Training done. Best Valid R@10={best_valid_recall10:.4f} at epoch {best_epoch}")

    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()
