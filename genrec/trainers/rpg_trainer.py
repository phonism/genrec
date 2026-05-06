"""
RPG Trainer - Recommendation with Parallel Generation (Meta, KDD 2025)

Trains a GPT-2 backbone with parallel ResBlock prediction heads using
OPQ-quantized semantic IDs. Supports graph-constrained decoding at test time.
"""
import os
import gin
import torch
import wandb

from genrec.models.rpg import RPGModel
from genrec.data.amazon_rpg import (
    AmazonRPGDataset,
    rpg_train_collate_fn,
    rpg_eval_collate_fn,
)
from genrec.modules.utils import parse_config, setup_logger
from genrec.trainers.trainer_utils import (
    setup_accelerator,
    setup_wandb,
    log_training_info,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup


def evaluate(model, dataloader, accelerator, top_ks=[1, 5, 10], use_graph=False):
    """
    Evaluate RPG model with Recall@K and NDCG@K.

    Args:
        model: RPG model.
        dataloader: Evaluation dataloader.
        accelerator: Accelerator instance.
        top_ks: List of K values for metrics.
        use_graph: Whether to use graph-constrained decoding.

    Returns:
        Dict of metric name -> value.
    """
    model.eval()
    device = accelerator.device
    raw_model = accelerator.unwrap_model(model)
    raw_model.use_graph_decoding = use_graph

    metrics = {f"Recall@{k}": 0.0 for k in top_ks}
    metrics.update({f"NDCG@{k}": 0.0 for k in top_ks})
    total = 0

    max_k = max(top_ks)

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_main_process):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            seq_lens = data["seq_lens"].to(device)
            labels = data["labels"].to(device)  # [B, 1]
            B = input_ids.size(0)

            preds = raw_model.generate(
                input_ids, attention_mask, seq_lens, n_return=max_k
            )

            if isinstance(preds, tuple):
                preds, _ = preds  # Discard visited counts from graph decoding

            preds = preds.squeeze(-1)  # [B, max_k]
            targets = labels.squeeze(-1)  # [B]

            for i in range(B):
                target = targets[i].item()
                pred_list = preds[i].tolist()

                for k in top_ks:
                    if target in pred_list[:k]:
                        metrics[f"Recall@{k}"] += 1.0
                        rank = pred_list[:k].index(target) + 1
                        metrics[f"NDCG@{k}"] += 1.0 / (
                            torch.log2(torch.tensor(rank + 1.0)).item()
                        )

            total += B

    # Gather from all GPUs
    def gather_scalar(v):
        t = torch.tensor([v], device=device, dtype=torch.float32)
        return accelerator.reduce(t, reduction="sum").item()

    total = int(gather_scalar(total))
    for k in metrics:
        metrics[k] = gather_scalar(metrics[k]) / total if total > 0 else 0.0

    raw_model.use_graph_decoding = False
    return metrics


@gin.configurable
def train_rpg(
    # Model
    n_embd=448,
    n_layer=2,
    n_head=4,
    n_inner=1024,
    n_codebook=32,
    codebook_size=256,
    temperature=0.07,
    embd_pdrop=0.5,
    attn_pdrop=0.5,
    resid_pdrop=0.0,
    # Graph decoding
    num_beams=50,
    n_edges=50,
    propagation_steps=3,
    chunk_size=1024,
    # Training
    epochs=150,
    batch_size=256,
    learning_rate=3e-4,
    weight_decay=0.01,
    warmup_steps=10000,
    max_grad_norm=1.0,
    # Dataset
    max_seq_len=50,
    dataset_folder="dataset/amazon",
    split="beauty",
    encoder_model_name="sentence-transformers/sentence-t5-xl",
    sent_emb_pca=0,
    # Evaluation
    do_eval=True,
    eval_every_epoch=1,
    eval_batch_size=256,
    eval_with_graph=False,
    patience=20,
    # Checkpoint
    save_dir_root="out/rpg/amazon/{split}",
    save_every_epoch=50,
    # Logging
    wandb_logging=False,
    wandb_project="rpg_training",
    wandb_run_name=None,
    wandb_log_interval=100,
    # Accelerator
    amp=True,
    mixed_precision_type="bf16",
):
    """Train RPG model."""
    save_dir_root = save_dir_root.replace("{split}", split)
    logger = setup_logger(save_dir_root, name="rpg")
    accelerator = setup_accelerator(amp=amp, mixed_precision_type=mixed_precision_type)
    device = accelerator.device

    if wandb_logging and accelerator.is_main_process:
        setup_wandb(
            project=wandb_project,
            run_name=wandb_run_name,
            config=locals(),
            step_metrics={"train/*": "global_step", "eval/*": "epoch"},
        )

    # Dataset
    ds_kwargs = dict(
        root=dataset_folder,
        split=split,
        max_seq_len=max_seq_len,
        n_codebook=n_codebook,
        codebook_size=codebook_size,
        encoder_model_name=encoder_model_name,
        sent_emb_pca=sent_emb_pca,
    )
    train_ds = AmazonRPGDataset(train_test_split="train", **ds_kwargs)
    valid_ds = AmazonRPGDataset(train_test_split="valid", **ds_kwargs)
    test_ds = AmazonRPGDataset(train_test_split="test", **ds_kwargs)

    logger.info(
        f"Items: {train_ds.num_items - 1}, "
        f"Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}"
    )

    collate_train = lambda x: rpg_train_collate_fn(x, max_seq_len)
    collate_eval = lambda x: rpg_eval_collate_fn(x, max_seq_len)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_train,
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_eval,
    )
    test_dl = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_eval,
    )

    # Model
    item_id2tokens = train_ds.item_id2tokens.to(device)
    model = RPGModel(
        n_items=train_ds.num_items,
        n_codebook=n_codebook,
        codebook_size=codebook_size,
        item_id2tokens=item_id2tokens,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_inner,
        max_seq_len=max_seq_len,
        temperature=temperature,
        embd_pdrop=embd_pdrop,
        attn_pdrop=attn_pdrop,
        resid_pdrop=resid_pdrop,
        num_beams=num_beams,
        n_edges=n_edges,
        propagation_steps=propagation_steps,
        chunk_size=chunk_size,
    )

    if accelerator.is_main_process:
        logger.info(model.n_parameters_str)

    optimizer = AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    train_dl, valid_dl, test_dl = accelerator.prepare(train_dl, valid_dl, test_dl)
    total_steps = len(train_dl) * epochs
    num_warmup = warmup_steps

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup, total_steps
    )
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    log_training_info(logger, model, train_dl, valid_dl, test_dl, device)

    # Training loop
    global_step = 0
    best_recall = 0.0
    wait = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            train_dl, desc=f"Epoch {epoch}",
            disable=not accelerator.is_main_process,
        )
        for data in pbar:
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            labels = data["labels"]

            out = model(input_ids, attention_mask, labels=labels)
            loss = out["loss"]

            accelerator.backward(loss)
            if max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if (
                wandb_logging
                and accelerator.is_main_process
                and global_step % wandb_log_interval == 0
            ):
                wandb.log({
                    "global_step": global_step,
                    "train/loss": loss.item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                })

        avg_loss = epoch_loss / len(train_dl)
        logger.info(f"Epoch {epoch} - loss: {avg_loss:.4f}")

        # Evaluation
        if do_eval and (epoch + 1) % eval_every_epoch == 0:
            metrics = evaluate(model, valid_dl, accelerator)
            test_metrics = evaluate(model, test_dl, accelerator)

            if accelerator.is_main_process:
                logger.info(
                    f"Epoch {epoch} - Valid: "
                    + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                )
                logger.info(
                    f"Epoch {epoch} - Test:  "
                    + ", ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()])
                )

                if wandb_logging:
                    wandb.log({
                        "epoch": epoch,
                        **{f"eval/{k}": v for k, v in metrics.items()},
                        **{f"test_epoch/{k}": v for k, v in test_metrics.items()},
                    })

                # Early stopping
                if metrics["NDCG@10"] > best_recall:
                    best_recall = metrics["NDCG@10"]
                    save_path = os.path.join(save_dir_root, "best_model.pt")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(
                        accelerator.unwrap_model(model).state_dict(), save_path
                    )
                    logger.info(
                        f"New best NDCG@10: {best_recall:.4f}, saved to {save_path}"
                    )
                    wait = 0
                else:
                    wait += 1
                    logger.info(f"No improvement for {wait}/{patience} epochs")
                    if wait >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

            model.train()

        # Save checkpoint
        if accelerator.is_main_process and (epoch + 1) % save_every_epoch == 0:
            save_path = os.path.join(save_dir_root, f"checkpoint_epoch_{epoch}.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(accelerator.unwrap_model(model).state_dict(), save_path)
            logger.info(f"Saved checkpoint to {save_path}")

    # Final test with graph decoding
    if accelerator.is_main_process:
        best_path = os.path.join(save_dir_root, "best_model.pt")
        if os.path.exists(best_path):
            accelerator.unwrap_model(model).load_state_dict(torch.load(best_path))
            logger.info(f"Loaded best model from {best_path}")

    # Standard test
    test_metrics = evaluate(model, test_dl, accelerator)
    if accelerator.is_main_process:
        logger.info(
            "Test Results (direct): "
            + ", ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()])
        )

    # Graph-constrained test (use smaller batch size to avoid OOM)
    if eval_with_graph:
        graph_test_dl = DataLoader(
            test_ds, batch_size=32, shuffle=False,
            num_workers=4, pin_memory=True, collate_fn=lambda x: rpg_eval_collate_fn(x, max_seq_len),
        )
        graph_test_dl = accelerator.prepare(graph_test_dl)
        graph_metrics = evaluate(model, graph_test_dl, accelerator, use_graph=True)
        if accelerator.is_main_process:
            logger.info(
                "Test Results (graph): "
                + ", ".join([f"{k}={v:.4f}" for k, v in graph_metrics.items()])
            )
            if wandb_logging:
                wandb.log({f"test_graph/{k}": v for k, v in graph_metrics.items()})

    if accelerator.is_main_process:
        if wandb_logging:
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
            wandb.finish()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    parse_config()
    train_rpg()
