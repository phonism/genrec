"""
OneRec SFT Trainer - Stage 2 of OneRec pipeline.

Trains the LLM with 4 tasks: sidsft, title2sid, sid2title, fusionseqrec.
Reuses collate_fn, evaluate, log_metrics from lcrec_trainer.
"""
import os
import gin
import torch
import wandb
from typing import List, Dict, Any

from genrec.models.onerec import OneRec
from genrec.modules.utils import parse_config, setup_logger, get_run_split
from genrec.modules.metrics import TopKAccumulator
from genrec.trainers.trainer_utils import setup_accelerator, setup_wandb, set_seed, save_run_results
from genrec.trainers.lcrec_trainer import (
    lcrec_collate_fn,
    ConstrainedDecodingHelper,
    evaluate as lcrec_evaluate,
    log_metrics as lcrec_log_metrics,
)
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup


def onerec_collate_fn(batch, tokenizer, max_length=512, num_codebooks=3, is_eval=False):
    """Collate function for OneRec SFT dataset.

    Reuses lcrec_collate_fn logic but with OneRec task names.
    The key difference: sidsft/title2sid map to seqrec/item2index for evaluation.
    """
    # Map OneRec task names to LCRec-compatible names for collate
    mapped_batch = []
    for s in batch:
        mapped = dict(s)
        # Map task names for collate_fn compatibility
        task = s['task']
        if task == 'sidsft':
            mapped['task'] = 'seqrec'
        elif task == 'title2sid':
            mapped['task'] = 'item2index'
        elif task == 'sid2title':
            mapped['task'] = 'index2item'
        # fusionseqrec stays the same
        mapped_batch.append(mapped)

    result = lcrec_collate_fn(mapped_batch, tokenizer, max_length, num_codebooks, is_eval)

    # Store original task names for logging
    result['original_tasks'] = [s['task'] for s in batch]
    return result


def evaluate_onerec(model, dataloader, accelerator, tokenizer, helper, num_codebooks,
                    beam_width=10, logger=None, epoch=0, debug=False, eval_tasks=None):
    """Evaluate OneRec model. Wraps lcrec_evaluate.

    `eval_tasks` defaults to {'seqrec', 'item2index'} — i.e. skip index2item, which
    requires 50-token free generation and is ~10x slower than the constrained
    seqrec path. Pass eval_tasks=None to evaluate all three.
    """
    if eval_tasks is None:
        eval_tasks = {'seqrec', 'item2index'}
    return lcrec_evaluate(
        model, dataloader, accelerator, tokenizer, helper,
        num_codebooks, beam_width, logger, epoch, debug, eval_tasks=eval_tasks,
    )


@gin.configurable
def train_sft(
    epochs=4, batch_size=8, learning_rate=3e-4, weight_decay=0.01,
    warmup_steps=20, warmup_ratio=0.0, adam_beta2=0.999,
    gradient_accumulate_every=2, max_length=512,
    pretrained_path="Qwen/Qwen2.5-1.5B", use_lora=True,
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    num_codebooks=3, codebook_size=8192,
    dataset=None, dataset_folder="dataset/amazon2023", max_seq_len=20, max_text_len=128,
    do_eval=True, eval_every_epoch=1, eval_batch_size=64, eval_beam_width=10,
    save_dir_root="out/onerec/amazon2023/{split}/sft", save_every_epoch=1,
    wandb_logging=False, wandb_project="onerec_sft", wandb_run_name=None, wandb_log_interval=10,
    split_batches=True, amp=True, mixed_precision_type="bf16",
    max_train_samples=0, max_eval_samples=0, debug_logging=False,
    eval_only=False, checkpoint_path=None,
    # OneRec-specific
    early_stopping_patience=3,
    attention_dropout=0.0,
    label_smoothing=0.0,
    enabled_tasks=None,
    seed=42,
):
    """Train OneRec SFT model."""
    _run_config = dict(locals())
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

    # Model setup
    print(f"[DEBUG] Loading OneRec from {pretrained_path}...", flush=True)
    model = OneRec(pretrained_path=pretrained_path)
    if attention_dropout > 0:
        model.model.config.attention_dropout = attention_dropout
        for layer in model.model.model.layers:
            layer.self_attn.attention_dropout = attention_dropout
        logger.info(f"Set attention_dropout={attention_dropout} on {len(model.model.model.layers)} layers")
    print(f"[DEBUG] Model loaded, vocab={model.model.config.vocab_size}", flush=True)
    print(f"[DEBUG] Adding codebook tokens ({num_codebooks}x{codebook_size})...", flush=True)
    model.add_codebook_tokens(num_codebooks=num_codebooks, codebook_size=codebook_size)
    print(f"[DEBUG] Codebook tokens added, new vocab={model.model.config.vocab_size}", flush=True)

    if use_lora:
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            model.model = get_peft_model(model.model, LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ))
            if accelerator.is_main_process:
                model.model.print_trainable_parameters()
            # Set peft FSDP auto wrap policy (use_orig_params=False mode)
            if accelerator.distributed_type == accelerator.distributed_type.FSDP:
                from peft.utils.other import fsdp_auto_wrap_policy
                fsdp_plugin = accelerator.state.fsdp_plugin
                fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model.model)
        except ImportError:
            logger.warning("peft not installed, training without LoRA")

    model.gradient_checkpointing_enable(use_reentrant=not use_lora)
    tokenizer = model.tokenizer
    helper = ConstrainedDecodingHelper(num_codebooks, codebook_size, tokenizer)

    # Dataset setup (tokenizer_type, paths etc. configured via gin)
    ds_kwargs = dict(root=dataset_folder, max_seq_len=max_seq_len, max_text_len=max_text_len)
    train_ds = dataset(train_test_split="train", **ds_kwargs)
    valid_ds = dataset(train_test_split="valid", **ds_kwargs)
    test_ds = dataset(train_test_split="test", **ds_kwargs)

    if max_train_samples > 0:
        train_ds.samples = train_ds.samples[:max_train_samples]
        logger.info(f"Limited train samples to {len(train_ds.samples)}")
    if max_eval_samples > 0:
        valid_ds.samples = valid_ds.samples[:max_eval_samples]
        test_ds.samples = test_ds.samples[:max_eval_samples]
        logger.info(f"Limited eval samples to {len(valid_ds.samples)}")

    collate_train = lambda x: onerec_collate_fn(x, tokenizer, max_length, num_codebooks, is_eval=False)
    collate_eval = lambda x: onerec_collate_fn(x, tokenizer, max_length, num_codebooks, is_eval=True)

    train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True, shuffle=True,
                          num_workers=4, pin_memory=True, collate_fn=collate_train)
    valid_dl = DataLoader(valid_ds, batch_size=eval_batch_size, shuffle=False,
                          num_workers=4, pin_memory=True, collate_fn=collate_eval)
    test_dl = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False,
                         num_workers=4, pin_memory=True, collate_fn=collate_eval)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                      betas=(0.9, adam_beta2))
    train_dl, valid_dl, test_dl = accelerator.prepare(train_dl, valid_dl, test_dl)

    total_steps = len(train_dl) * epochs // gradient_accumulate_every
    num_warmup = int(total_steps * warmup_ratio) if warmup_ratio > 0 else warmup_steps
    logger.info(f"Total steps: {total_steps}, Warmup: {num_warmup} (ratio={warmup_ratio}, beta2={adam_beta2})")

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup, total_steps)
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    logger.info(f"Device: {device}, Params: {sum(p.numel() for p in model.parameters()):,}, "
                f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if checkpoint_path:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        accelerator.unwrap_model(model).load_pretrained(checkpoint_path)

    if eval_only:
        logger.info("Running eval-only mode...")
        metrics, topk = evaluate_onerec(model, valid_dl, accelerator, tokenizer, helper,
                                        num_codebooks, eval_beam_width, logger, 0, debug_logging)
        if accelerator.is_main_process:
            lcrec_log_metrics(metrics, topk, num_codebooks, 0, logger)
        accelerator.wait_for_everyone()
        return

    # Training loop with early stopping
    pbar = tqdm(total=total_steps, dynamic_ncols=True) if accelerator.is_main_process else None
    global_step = 0
    best_recall10 = 0.0
    patience_counter = 0
    last_test_topk: Dict[str, float] = {}
    last_test_task_exact: Dict[str, float] = {}

    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_steps = 0.0, 0

        for data in train_dl:
            with accelerator.accumulate(model):
                if label_smoothing > 0:
                    outputs = model(input_ids=data["input_ids"], attention_mask=data["attention_mask"])
                    logits = outputs.logits[:, :-1, :].contiguous()
                    labels = data["labels"][:, 1:].contiguous()
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), labels.view(-1),
                        ignore_index=-100, label_smoothing=label_smoothing,
                    )
                else:
                    outputs = model(input_ids=data["input_ids"], attention_mask=data["attention_mask"],
                                    labels=data["labels"])
                    loss = outputs.loss
                accelerator.backward(loss)
                epoch_loss += loss.item()
                epoch_steps += 1

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if pbar:
                        pbar.set_description(f'Epoch {epoch} | loss: {loss.item():.4f}')
                        pbar.update(1)

                    if wandb_logging and accelerator.is_main_process and global_step % wandb_log_interval == 0:
                        wandb.log({
                            "global_step": global_step,
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            "train/loss": loss.item(),
                        })

            accelerator.wait_for_everyone()

        logger.info(f"Epoch {epoch} - avg_loss: {epoch_loss / max(epoch_steps, 1):.4f}")

        # Evaluation
        if do_eval and (epoch + 1) % eval_every_epoch == 0:
            metrics, topk = evaluate_onerec(model, valid_dl, accelerator, tokenizer, helper,
                                            num_codebooks, eval_beam_width, logger, epoch, debug_logging)

            # `topk` is already reduced across ranks inside lcrec_evaluate, so
            # `current_recall10` is consistent everywhere — safe to compare without sync.
            current_recall10 = topk.get("Recall@10", 0.0)
            is_best = current_recall10 > best_recall10
            # First eval (no cached test yet) also forces a test pass for the baseline.
            need_test = is_best or not last_test_topk

            if accelerator.is_main_process:
                logger.info(f"--- Valid ---")
                lcrec_log_metrics(metrics, topk, num_codebooks, epoch, logger, wandb_logging)

            if need_test:
                test_metrics, test_topk = evaluate_onerec(model, test_dl, accelerator, tokenizer, helper,
                                                          num_codebooks, eval_beam_width, logger, epoch, debug_logging)
                if accelerator.is_main_process:
                    logger.info(f"--- Test (refreshed) ---")
                    lcrec_log_metrics(test_metrics, test_topk, num_codebooks, epoch, logger, wandb_log=False)
                    last_test_topk = dict(test_topk)
                    last_test_task_exact = {
                        task: (test_metrics[task]['exact'] / test_metrics[task]['total'])
                        for task in ('seqrec', 'item2index')
                        if task in test_metrics and test_metrics[task].get('total', 0) > 0
                    }
                    if wandb_logging:
                        test_log = {"epoch": epoch}
                        for task, v in last_test_task_exact.items():
                            test_log[f"test/{task}_exact"] = v
                        for k, v in test_topk.items():
                            test_log[f"test/seqrec_{k}"] = v
                        wandb.log(test_log)
            elif accelerator.is_main_process:
                logger.info(f"--- Test (cached from previous best, skipped) ---")

            # Early stopping bookkeeping (kept consistent on all ranks).
            if is_best:
                best_recall10 = current_recall10
                patience_counter = 0
            else:
                patience_counter += 1
                if accelerator.is_main_process:
                    logger.info(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")

            # Save best model - ALL ranks must participate for FSDP state dict gathering
            is_best_tensor = torch.tensor([1 if is_best else 0], device=device)
            is_best_tensor = accelerator.reduce(is_best_tensor, reduction="max")
            if is_best_tensor.item() >= 1:
                save_path = os.path.join(save_dir_root, "checkpoint_best")
                if accelerator.is_main_process:
                    os.makedirs(save_path, exist_ok=True)
                accelerator.wait_for_everyone()
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(
                    save_path,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(model),
                )
                if accelerator.is_main_process:
                    logger.info(f"New best Recall@10: {best_recall10:.4f}, saved to {save_path}")

            model.train()

            # Broadcast early stopping decision
            should_stop = torch.tensor([patience_counter >= early_stopping_patience], device=device)
            should_stop = accelerator.reduce(should_stop, reduction="max")
            if should_stop.item() >= 1:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Save checkpoint - ALL ranks participate for FSDP
        if (epoch + 1) % save_every_epoch == 0:
            save_path = os.path.join(save_dir_root, f"checkpoint_epoch_{epoch}")
            if accelerator.is_main_process:
                os.makedirs(save_path, exist_ok=True)
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(
                save_path,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
            )
            if accelerator.is_main_process:
                logger.info(f"Saved checkpoint to {save_path}")

    # Final save - ALL ranks participate for FSDP
    save_path = os.path.join(save_dir_root, "checkpoint_final")
    if accelerator.is_main_process:
        os.makedirs(save_path, exist_ok=True)
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        save_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )
    if accelerator.is_main_process:
        logger.info(f"Saved final checkpoint to {save_path}")
        if pbar:
            pbar.close()

    if wandb_logging and accelerator.is_main_process:
        wandb.finish()

    if accelerator.is_main_process and last_test_topk:
        save_run_results(
            save_dir=save_dir_root,
            model="onerec_sft",
            split=get_run_split(),
            seed=seed,
            metrics=last_test_topk,
            config=_run_config,
            extra={
                "best_valid_recall@10": best_recall10,
                "task_exact": last_test_task_exact,
            },
        )

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parse_config()
    train_sft()
