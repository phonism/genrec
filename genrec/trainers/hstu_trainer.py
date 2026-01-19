"""
HSTU Trainer - Hierarchical Sequential Transduction Unit
"""
import os
import gin
import torch
import wandb
import logging
from datetime import datetime

from accelerate import Accelerator
from genrec.models.hstu import HSTU
from genrec.modules.utils import parse_config
from genrec.data.amazon_hstu import AmazonHSTUDataset, hstu_collate_fn, hstu_eval_collate_fn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


def setup_logger(save_dir: str, name: str = "hstu") -> logging.Logger:
    """Setup logger."""
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(save_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def evaluate(model, dataloader, accelerator, use_temporal_bias=True, top_ks=[1, 5, 10]):
    """Evaluate model with Recall@K and NDCG@K."""
    model.eval()
    device = accelerator.device

    metrics = {f'Recall@{k}': 0.0 for k in top_ks}
    metrics.update({f'NDCG@{k}': 0.0 for k in top_ks})
    total = 0

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_main_process):
            input_ids = data['input_ids'].to(device)
            targets = data['targets'].to(device)
            timestamps = data['timestamps'].to(device) if use_temporal_bias else None
            B = input_ids.size(0)

            logits, _ = accelerator.unwrap_model(model)(input_ids, timestamps)
            last_logits = logits[:, -1, :]
            last_logits[:, 0] = float('-inf')

            max_k = max(top_ks)
            _, top_k_items = torch.topk(last_logits, max_k, dim=-1)

            for i in range(B):
                target = targets[i].item()
                preds = top_k_items[i].tolist()

                for k in top_ks:
                    if target in preds[:k]:
                        metrics[f'Recall@{k}'] += 1.0
                        rank = preds[:k].index(target) + 1
                        metrics[f'NDCG@{k}'] += 1.0 / (torch.log2(torch.tensor(rank + 1.0)).item())

            total += B

    # Gather from all GPUs
    def gather(v):
        t = torch.tensor([v], device=device, dtype=torch.float32)
        return accelerator.reduce(t, reduction="sum").item()

    total = int(gather(total))
    for k in metrics:
        metrics[k] = gather(metrics[k]) / total if total > 0 else 0.0

    return metrics


@gin.configurable
def train(
    epochs=200, batch_size=128, learning_rate=1e-3, weight_decay=0.0,
    max_seq_len=50, embed_dim=64, num_heads=2, num_blocks=2, dropout=0.2,
    num_position_buckets=32, num_time_buckets=64, use_temporal_bias=True,
    dataset_folder="dataset/amazon", split="beauty",
    do_eval=True, eval_every_epoch=10, eval_batch_size=256,
    save_dir_root="out/hstu/amazon/beauty", save_every_epoch=50,
    wandb_logging=False, wandb_project="hstu_training", wandb_log_interval=100,
    amp=True, mixed_precision_type="bf16",
):
    """Train HSTU model."""
    logger = setup_logger(save_dir_root)
    accelerator = Accelerator(mixed_precision=mixed_precision_type if amp else "no")
    device = accelerator.device

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        wandb.init(project=wandb_project, config=locals())
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("eval/*", step_metric="epoch")

    # Dataset
    train_ds = AmazonHSTUDataset(root=dataset_folder, split=split, train_test_split="train", max_seq_len=max_seq_len)
    valid_ds = AmazonHSTUDataset(root=dataset_folder, split=split, train_test_split="valid", max_seq_len=max_seq_len)
    test_ds = AmazonHSTUDataset(root=dataset_folder, split=split, train_test_split="test", max_seq_len=max_seq_len)

    num_items = train_ds.num_items
    logger.info(f"Num items: {num_items}, Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")

    collate_train = lambda x: hstu_collate_fn(x, max_seq_len)
    collate_eval = lambda x: hstu_eval_collate_fn(x, max_seq_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_train)
    valid_dl = DataLoader(valid_ds, batch_size=eval_batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_eval)
    test_dl = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_eval)

    # Model
    model = HSTU(
        num_items=num_items,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
        num_position_buckets=num_position_buckets,
        num_time_buckets=num_time_buckets,
        use_temporal_bias=use_temporal_bias,
    )

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.98))
    train_dl, valid_dl, test_dl = accelerator.prepare(train_dl, valid_dl, test_dl)
    model, optimizer = accelerator.prepare(model, optimizer)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {num_params:,}, Temporal bias: {use_temporal_bias}")

    # Training
    global_step = 0
    best_recall = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}", disable=not accelerator.is_main_process)
        for data in pbar:
            input_ids = data['input_ids']
            targets = data['targets']
            timestamps = data['timestamps'] if use_temporal_bias else None

            _, loss = model(input_ids, timestamps, targets)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if wandb_logging and accelerator.is_main_process and global_step % wandb_log_interval == 0:
                wandb.log({"global_step": global_step, "train/loss": loss.item()})

        avg_loss = epoch_loss / len(train_dl)
        logger.info(f"Epoch {epoch} - loss: {avg_loss:.4f}")

        # Evaluation
        if do_eval and (epoch + 1) % eval_every_epoch == 0:
            metrics = evaluate(model, valid_dl, accelerator, use_temporal_bias)
            if accelerator.is_main_process:
                logger.info(f"Epoch {epoch} - Valid: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))
                if wandb_logging:
                    wandb.log({"epoch": epoch, **{f"eval/{k}": v for k, v in metrics.items()}})

                if metrics['Recall@10'] > best_recall:
                    best_recall = metrics['Recall@10']
                    save_path = os.path.join(save_dir_root, "best_model.pt")
                    torch.save(accelerator.unwrap_model(model).state_dict(), save_path)
                    logger.info(f"New best Recall@10: {best_recall:.4f}, saved to {save_path}")

            model.train()

        # Save checkpoint
        if accelerator.is_main_process and (epoch + 1) % save_every_epoch == 0:
            save_path = os.path.join(save_dir_root, f"checkpoint_epoch_{epoch}.pt")
            torch.save(accelerator.unwrap_model(model).state_dict(), save_path)
            logger.info(f"Saved checkpoint to {save_path}")

    # Final test
    if accelerator.is_main_process:
        best_path = os.path.join(save_dir_root, "best_model.pt")
        if os.path.exists(best_path):
            accelerator.unwrap_model(model).load_state_dict(torch.load(best_path))

    test_metrics = evaluate(model, test_dl, accelerator, use_temporal_bias)
    if accelerator.is_main_process:
        logger.info(f"Test Results: " + ", ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()]))
        if wandb_logging:
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
            wandb.finish()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    parse_config()
    train()
