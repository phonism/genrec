"""
GRAM Trainer - Generative Retrieval via Multi-Granularity Late Fusion (ACL 2025)

Training pipeline:
1. Load pre-processed data (user sequences, item text, lexical IDs)
2. Train T5-based GRAM model with cross-entropy loss
3. Evaluate with constrained beam search using Trie prefix tree
4. Report Hit@K and NDCG@K metrics

Uses gin for configuration, accelerate for distributed training, and wandb for logging.
"""
import os
import gin
import math
import torch
import wandb
import numpy as np

from genrec.models.gram import GRAM
from genrec.data.amazon_gram import AmazonGRAMDataset, GRAMCollator
from genrec.utils.generation_trie import Trie, prefix_allowed_tokens_fn
from genrec.modules.utils import parse_config, setup_logger, get_run_split
from genrec.trainers.trainer_utils import setup_accelerator, setup_wandb, set_seed, save_run_results

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup


def canonical_metric_name(name):
    """hit@K -> Recall@K (identical under leave-one-out), ndcg@K -> NDCG@K.

    Keeps GRAM's reported keys aligned with the repo-wide TopKAccumulator
    naming that aggregate_results.py / run_matrix.py consume.
    """
    kind, k = name.lower().split("@")
    return f"Recall@{k}" if kind == "hit" else f"NDCG@{k}"


def hit_at_k(relevance_lists, k):
    """Compute Hit@K: fraction of users with a hit in top-K."""
    correct = 0.0
    for row in relevance_lists:
        if sum(row[:k]) > 0:
            correct += 1
    return correct


def ndcg_at_k(relevance_lists, k):
    """Compute NDCG@K: leave-one-out so IDCG = 1.0."""
    ndcg = 0.0
    for row in relevance_lists:
        rel = row[:k]
        for i, r in enumerate(rel):
            ndcg += r / math.log(i + 2, 2)
    return ndcg


def compute_rel_results(predictions, targets, scores, k):
    """
    Compute relevance results for a batch.

    Args:
        predictions: List of decoded prediction strings
        targets: List of ground truth strings
        scores: Tensor of prediction scores
        k: Number of beams per sample

    Returns:
        List of relevance lists (one per sample in batch)
    """
    results = []
    batch_length = len(targets)
    for b in range(batch_length):
        batch_preds = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]
        pairs = list(zip(batch_preds, batch_scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        gt = targets[b]
        rel = [1 if pred == gt else 0 for pred, _ in pairs]
        results.append(rel)
    return results


def build_candidate_trie(all_items, tokenizer):
    """Build a Trie from all candidate lexical IDs for constrained beam search."""
    encoded_candidates = []
    for candidate in all_items:
        encoded = [0] + tokenizer.encode(candidate)
        encoded_candidates.append(encoded)
    return Trie(encoded_candidates), encoded_candidates


@gin.configurable
def train_gram(
    # Model
    backbone="t5-small",
    max_his=20,
    use_position_embedding=True,
    # Training
    epochs=30,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=0.01,
    warmup_ratio=0.05,
    gradient_accumulate_every=2,
    max_grad_norm=1.0,
    # Data
    dataset=None,
    dataset_folder="dataset/gram",
    split="Beauty",
    item_prompt_max_len=128,
    target_max_len=50,
    item_id_file="item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt",
    item_prompt="all_text",
    top_k_similar_item=10,
    cf_model="sasrec",
    his_sep=" ; ",
    skip_empty_his=True,
    reverse_history=True,
    # Evaluation
    num_beams=50,
    eval_batch_size=1,
    length_penalty=0.0,
    do_eval=True,
    eval_every_epoch=1,
    metrics="hit@5,hit@10,ndcg@5,ndcg@10",
    # Checkpointing
    save_dir_root="out/gram/{split}",
    save_every_epoch=5,
    # Wandb
    wandb_logging=False,
    wandb_project="gram",
    wandb_run_name=None,
    wandb_log_interval=10,
    # Accelerator
    split_batches=True,
    amp=True,
    mixed_precision_type="bf16",
    # Debug
    max_train_samples=0,
    max_eval_samples=0,
    eval_only=False,
    checkpoint_path=None,
    seed=42,
):
    """Train GRAM model."""
    _run_config = dict(locals())
    save_dir_root = save_dir_root.format(split=split)
    logger = setup_logger(save_dir_root)
    set_seed(seed)
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
            step_metrics={"train/*": "global_step", "eval/*": "epoch"},
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    logger.info(f"Loaded tokenizer from {backbone}, vocab_size={len(tokenizer)}")

    # Create model
    model = GRAM.from_pretrained_t5(
        backbone,
        max_item_num=max_his,
        use_position_embedding=use_position_embedding,
    )
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"GRAM model: {num_params:,} params, {trainable_params:,} trainable")

    # Load datasets
    ds_kwargs = dict(
        root=dataset_folder,
        split=split,
        max_seq_len=max_his,
        max_text_len=item_prompt_max_len,
        item_id_file=item_id_file,
        item_prompt=item_prompt,
        top_k_similar_item=top_k_similar_item,
        cf_model=cf_model,
        his_sep=his_sep,
        skip_empty_his=skip_empty_his,
        reverse_history=reverse_history,
    )

    dataset_cls = dataset or AmazonGRAMDataset
    train_ds = dataset_cls(train_test_split="train", **ds_kwargs)
    valid_ds = dataset_cls(train_test_split="valid", **ds_kwargs)
    test_ds = dataset_cls(train_test_split="test", **ds_kwargs)

    if max_train_samples > 0:
        train_ds.data = train_ds.data[:max_train_samples]
        logger.info(f"Limited train samples to {len(train_ds.data)}")
    if max_eval_samples > 0:
        valid_ds.data = valid_ds.data[:max_eval_samples]
        test_ds.data = test_ds.data[:max_eval_samples]
        logger.info(f"Limited eval samples to {len(valid_ds.data)}")

    # GRAMCollator is stateless per-call — one instance serves all loaders.
    collator = GRAMCollator(
        tokenizer=tokenizer,
        item_prompt_max_len=item_prompt_max_len,
        target_max_len=target_max_len,
        max_item_num=max_his,
    )

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True, collate_fn=collator,
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collator,
    )
    test_dl = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collator,
    )

    # Build Trie for constrained beam search
    candidate_trie, encoded_candidates = build_candidate_trie(train_ds.all_items, tokenizer)
    max_gen_length = max(len(c) for c in encoded_candidates)
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
    logger.info(f"Built Trie with {len(encoded_candidates)} candidates, max_gen_length={max_gen_length}")

    # Parse metrics
    metric_names = [m.strip() for m in metrics.split(",")]
    generate_num = max(int(m.split("@")[1]) for m in metric_names)
    generate_num = max(generate_num, num_beams)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_dl, valid_dl, test_dl = accelerator.prepare(train_dl, valid_dl, test_dl)

    total_steps = len(train_dl) * epochs // gradient_accumulate_every
    warmup_steps = int(total_steps * warmup_ratio)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    logger.info(f"Device: {device}, Total steps: {total_steps}, Warmup: {warmup_steps}")
    logger.info(f"Train: {len(train_ds)} samples, Valid: {len(valid_ds)}, Test: {len(test_ds)}")

    if checkpoint_path:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(state_dict, strict=False)

    if eval_only:
        logger.info("Running eval-only mode...")
        _evaluate(
            model, valid_dl, accelerator, tokenizer,
            prefix_allowed_tokens, max_gen_length, generate_num,
            length_penalty, metric_names, logger, epoch=0, mode="valid",
        )
        _evaluate(
            model, test_dl, accelerator, tokenizer,
            prefix_allowed_tokens, max_gen_length, generate_num,
            length_penalty, metric_names, logger, epoch=0, mode="test",
        )
        return

    # Training loop
    pbar = tqdm(total=total_steps, dynamic_ncols=True) if accelerator.is_main_process else None
    global_step = 0
    best_recall10 = 0.0
    last_test_metrics = {}
    # Rank-consistent test gating (same pattern as the OneRec trainers): _evaluate
    # returns globally reduced metrics on every rank, so all ranks agree on is_best.
    has_cached_test = False

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):
                input_ids = batch["item_text_ids"]
                attention_mask = batch["item_text_masks"]
                labels = batch["target_ids"]

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )
                loss = outputs.loss

                accelerator.backward(loss)
                epoch_loss += loss.item()
                epoch_steps += 1

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if pbar:
                        pbar.set_description(f"Epoch {epoch} | loss: {loss.item():.4f}")
                        pbar.update(1)

                    if wandb_logging and accelerator.is_main_process and global_step % wandb_log_interval == 0:
                        wandb.log({
                            "global_step": global_step,
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            "train/loss": loss.item(),
                        })

            accelerator.wait_for_everyone()

        avg_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(f"Epoch {epoch} - avg_loss: {avg_loss:.4f}")

        # Evaluation
        if do_eval and (epoch + 1) % eval_every_epoch == 0:
            valid_metrics = _evaluate(
                model, valid_dl, accelerator, tokenizer,
                prefix_allowed_tokens, max_gen_length, generate_num,
                length_penalty, metric_names, logger, epoch=epoch, mode="valid",
            )

            # 50-beam constrained decoding is the dominant cost: only rerun the
            # test set when valid improved (plus once for the baseline).
            current_recall10 = valid_metrics.get("Recall@10", 0.0)
            is_best = current_recall10 > best_recall10
            need_test = is_best or not has_cached_test
            if is_best:
                best_recall10 = current_recall10

            if need_test:
                test_metrics = _evaluate(
                    model, test_dl, accelerator, tokenizer,
                    prefix_allowed_tokens, max_gen_length, generate_num,
                    length_penalty, metric_names, logger, epoch=epoch, mode="test",
                )
                has_cached_test = True
                last_test_metrics = dict(test_metrics)
            elif accelerator.is_main_process:
                logger.info("--- Test (cached from previous best, skipped) ---")

            if accelerator.is_main_process:
                if wandb_logging:
                    log_dict = {"epoch": epoch}
                    for name, val in valid_metrics.items():
                        log_dict[f"eval/valid_{name}"] = val
                    for name, val in last_test_metrics.items():
                        log_dict[f"eval/test_{name}"] = val
                    wandb.log(log_dict)

                if is_best:
                    save_path = os.path.join(save_dir_root, "checkpoint_best.pt")
                    os.makedirs(save_dir_root, exist_ok=True)
                    torch.save(
                        accelerator.unwrap_model(model).state_dict(),
                        save_path,
                    )
                    logger.info(f"New best Recall@10: {best_recall10:.4f}, saved to {save_path}")

            model.train()

        # Periodic checkpoint
        if accelerator.is_main_process and (epoch + 1) % save_every_epoch == 0:
            save_path = os.path.join(save_dir_root, f"checkpoint_epoch_{epoch}.pt")
            os.makedirs(save_dir_root, exist_ok=True)
            torch.save(
                accelerator.unwrap_model(model).state_dict(),
                save_path,
            )
            logger.info(f"Saved checkpoint to {save_path}")

        accelerator.wait_for_everyone()

    # Final save
    if accelerator.is_main_process:
        save_path = os.path.join(save_dir_root, "checkpoint_final.pt")
        os.makedirs(save_dir_root, exist_ok=True)
        torch.save(
            accelerator.unwrap_model(model).state_dict(),
            save_path,
        )
        logger.info(f"Saved final checkpoint to {save_path}")
        if pbar:
            pbar.close()

    if wandb_logging and accelerator.is_main_process:
        wandb.finish()

    if accelerator.is_main_process and last_test_metrics:
        save_run_results(
            save_dir=save_dir_root,
            model="gram",
            split=get_run_split(default=split),
            seed=seed,
            metrics=last_test_metrics,
            config=_run_config,
            extra={"best_valid_recall@10": best_recall10, "backbone": backbone},
        )

    accelerator.wait_for_everyone()
    accelerator.end_training()


@torch.no_grad()
def _evaluate(
    model, dataloader, accelerator, tokenizer,
    prefix_allowed_tokens, max_gen_length, generate_num,
    length_penalty, metric_names, logger, epoch=0, mode="test",
):
    """
    Evaluate with constrained beam search.

    Returns dict of {metric_name: value}.
    """
    model.eval()
    unwrapped = accelerator.unwrap_model(model)

    total_samples = 0
    metrics_sum = np.zeros(len(metric_names))
    sample_count = 0

    for batch in tqdm(dataloader, desc=f"Eval {mode}", dynamic_ncols=True, disable=not accelerator.is_main_process):
        input_ids = batch["item_text_ids"]
        attention_mask = batch["item_text_masks"]
        target_ids = batch["target_ids"]

        prediction = unwrapped.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_gen_length,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
            num_beams=generate_num,
            num_return_sequences=generate_num,
            output_scores=True,
            return_dict_in_generate=True,
            length_penalty=length_penalty,
        )

        prediction_ids = prediction["sequences"]
        prediction_scores = prediction["sequences_scores"]

        # Decode predictions and targets
        # Replace -100 with 0 for decoding
        clean_target_ids = torch.where(target_ids == -100, 0, target_ids)
        gold_sents = tokenizer.batch_decode(clean_target_ids, skip_special_tokens=True)
        generated_sents = tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)

        # Compute relevance results
        rel_results = compute_rel_results(
            generated_sents, gold_sents, prediction_scores, generate_num,
        )

        total_samples += len(rel_results)

        # Compute metrics
        for i, name in enumerate(metric_names):
            if name.lower().startswith("hit"):
                k = int(name.split("@")[1])
                metrics_sum[i] += hit_at_k(rel_results, k)
            elif name.lower().startswith("ndcg"):
                k = int(name.split("@")[1])
                metrics_sum[i] += ndcg_at_k(rel_results, k)

        # Log first few examples
        if sample_count < 5 and accelerator.is_main_process:
            logger.info(f"  [{mode}] GT: {gold_sents[0]} | Top-1: {generated_sents[0]}")
        sample_count += 1

    # Aggregate across processes
    metrics_tensor = torch.tensor(metrics_sum, device=accelerator.device)
    total_tensor = torch.tensor(total_samples, device=accelerator.device)
    metrics_tensor = accelerator.reduce(metrics_tensor, reduction="sum")
    total_tensor = accelerator.reduce(total_tensor, reduction="sum")

    # Build the result on EVERY rank (the reduced tensors are identical
    # everywhere) — the caller's is_best/need_test gating must agree across
    # ranks or the next collective op deadlocks. Logging stays main-only.
    result = {}
    total = total_tensor.item()
    if total > 0:
        for i, name in enumerate(metric_names):
            result[canonical_metric_name(name)] = metrics_tensor[i].item() / total

    if accelerator.is_main_process and total > 0:
        logger.info(f"\n--- {mode.upper()} Epoch {epoch} ({int(total)} samples) ---")
        for name, val in result.items():
            logger.info(f"  {name}: {val:.4f}")

    return result


if __name__ == "__main__":
    parse_config()
    train_gram()
