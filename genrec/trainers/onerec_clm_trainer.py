"""
OneRec CLM Trainer - Causal Language Modeling (OpenOneRec-style).

Trains the LLM on flat token streams: one user sequence = one training sample.
Each item is represented as codebook tokens separated by <|item_sep|>.
Standard causal LM loss at every position.
"""
import os
import re
import gin
import torch
import wandb
from typing import List, Dict, Any, Callable, Optional, Tuple

from genrec.models.onerec import OneRec
from genrec.data.amazon_onerec_clm import AmazonOneRecCLMDataset
from genrec.modules.utils import parse_config, setup_logger, get_run_split
from genrec.modules.metrics import SemanticIdItemMapper, TopKAccumulator
from genrec.trainers.trainer_utils import setup_accelerator, setup_wandb, set_seed, save_run_results
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup


class CLMTokenMapper:
    """Precomputes codebook token ID mappings for CLM format."""

    def __init__(self, tokenizer, num_codebooks: int, codebook_size: int):
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        self.item_sep_id = tokenizer.convert_tokens_to_ids("<|item_sep|>")

        # Precompute: codebook_token_ids[c][code] = token_id
        self.codebook_token_ids = []
        for c in range(num_codebooks):
            code_to_id = {}
            for code in range(codebook_size):
                ids = tokenizer(f"<C{c}_{code}>", add_special_tokens=False).input_ids
                assert len(ids) == 1, f"<C{c}_{code}> encoded to {len(ids)} tokens"
                code_to_id[code] = ids[0]
            self.codebook_token_ids.append(code_to_id)

    def item_to_tokens(self, sem_ids: List[int]) -> List[int]:
        """Convert one item's semantic IDs to token IDs."""
        return [self.codebook_token_ids[c][code] for c, code in enumerate(sem_ids)]

    def sequence_to_tokens(self, item_ids: List[int], sem_ids_list: List[List[int]]) -> List[int]:
        """Convert a sequence of item IDs to a flat CLM token stream.

        Format: C0 C1 C2 <sep> C0 C1 C2 <sep> ... C0 C1 C2
        No separator after last item.
        """
        tokens = []
        for i, item_id in enumerate(item_ids):
            sem_ids = sem_ids_list[item_id] if item_id < len(sem_ids_list) else [0] * self.num_codebooks
            tokens.extend(self.item_to_tokens(sem_ids))
            if i < len(item_ids) - 1:
                tokens.append(self.item_sep_id)
        return tokens


class CLMConstrainedDecodingHelper:
    """Constrained decoding helper for CLM format (no instruction template)."""

    def __init__(self, num_codebooks: int, codebook_size: int, tokenizer):
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.pattern = re.compile(r'<C(\d+)_(\d+)>')

        vocab_size = len(tokenizer)

        # Build position_mask: [num_steps, vocab_size]
        # Step 0 → C0 tokens, Step 1 → C1 tokens, ..., Step num_codebooks → EOS
        self.position_mask = torch.zeros(num_codebooks + 1, vocab_size, dtype=torch.bool)
        self.code_map = torch.full((num_codebooks, vocab_size), -1, dtype=torch.long)

        for c in range(num_codebooks):
            for code in range(codebook_size):
                ids = tokenizer(f"<C{c}_{code}>", add_special_tokens=False).input_ids
                if len(ids) == 1:
                    self.position_mask[c, ids[0]] = True
                    self.code_map[c, ids[0]] = code

        # Last step: EOS only
        self.position_mask[num_codebooks, tokenizer.eos_token_id] = True

    def extract_sem_ids(self, text: str) -> Optional[List[int]]:
        matches = self.pattern.findall(text)
        if len(matches) >= self.num_codebooks:
            return [int(matches[i][1]) for i in range(self.num_codebooks)]
        return None


def clm_collate_fn(batch, token_mapper, sem_ids_list, max_length=512):
    """Collate function for CLM training.

    Converts item_seq → flat token stream + EOS, right-pads.
    Labels = input_ids with padding masked to -100 (HF CausalLM shifts internally).
    """
    all_input_ids = []

    for sample in batch:
        item_seq = sample['item_seq']
        # Truncate to max_seq_len most recent items
        token_ids = token_mapper.sequence_to_tokens(item_seq, sem_ids_list)
        token_ids.append(token_mapper.eos_id)
        # Truncate from LEFT if too long (keep recent items)
        if len(token_ids) > max_length:
            token_ids = token_ids[-max_length:]
        all_input_ids.append(token_ids)

    max_len = min(max(len(ids) for ids in all_input_ids), max_length)

    padded_input_ids = []
    padded_labels = []
    attention_masks = []

    for ids in all_input_ids:
        pad_len = max_len - len(ids)
        padded_input_ids.append(ids + [token_mapper.pad_id] * pad_len)
        padded_labels.append(ids + [-100] * pad_len)
        attention_masks.append([1] * len(ids) + [0] * pad_len)

    return {
        'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        'labels': torch.tensor(padded_labels, dtype=torch.long),
    }


def clm_eval_collate_fn(batch, token_mapper, sem_ids_list, max_length=512, num_codebooks=3):
    """Collate function for CLM evaluation.

    History → token stream + trailing <|item_sep|> (to trigger next-item prediction).
    Left-pads for decoder generation.
    """
    all_token_ids = []
    target_sem_ids = []
    target_items = []
    tasks = []

    for sample in batch:
        history = sample['item_seq']
        # Build CLM prefix with trailing separator
        tokens = token_mapper.sequence_to_tokens(history, sem_ids_list)
        tokens.append(token_mapper.item_sep_id)  # signal: "predict next item"
        # Truncate from left
        if len(tokens) > max_length:
            tokens = tokens[-max_length:]
        all_token_ids.append(tokens)
        target_sem_ids.append(sample.get('target_sem_ids', [0] * num_codebooks))
        target_items.append(sample.get('target', -1))
        tasks.append('seqrec')

    max_len = min(max(len(ids) for ids in all_token_ids), max_length)

    # Left-pad for generation
    padded_ids = []
    attention_masks = []
    for ids in all_token_ids:
        pad_len = max_len - len(ids)
        padded_ids.append([token_mapper.pad_id] * pad_len + ids)
        attention_masks.append([0] * pad_len + [1] * len(ids))

    return {
        'input_ids': torch.tensor(padded_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        'target_sem_ids': torch.tensor(target_sem_ids, dtype=torch.long),
        'target_items': torch.tensor(target_items, dtype=torch.long),
        'tasks': tasks,
    }


def evaluate_clm(model, dataloader, accelerator, helper, num_codebooks,
                  beam_width=10, logger=None, epoch=0, item_mapper=None):
    """Evaluate CLM model using constrained beam search."""
    if item_mapper is None:
        raise ValueError("item_mapper is required for honest item-level evaluation")
    model.eval()
    device = accelerator.device
    metrics = {'correct': [0] * num_codebooks, 'total': 0, 'exact': 0}
    topk_acc = TopKAccumulator(ks=[1, 5, 10])
    sid_topk_acc = TopKAccumulator(ks=[1, 5, 10])

    raw_model = accelerator.unwrap_model(model)

    # FSDP: need to gather full params for inference (beam search needs 2-D weights)
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    is_fsdp = isinstance(model, FSDP) or any(isinstance(m, FSDP) for m in model.modules())
    fsdp_ctx = FSDP.summon_full_params(model, writeback=False, recurse=True) if is_fsdp else None

    if fsdp_ctx is not None:
        fsdp_ctx.__enter__()

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=is_fsdp):
        for data in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_main_process):
            inp = data['input_ids'].to(device)
            attn = data['attention_mask'].to(device)
            tgt = data['target_sem_ids'].to(device)
            target_items = data['target_items'].to(device)
            if (target_items < 0).any():
                raise ValueError("CLM eval batch is missing raw target item IDs")
            N = inp.size(0)

            pred_ids, _ = raw_model.generate_constrained_beam(
                inp, attn, helper.position_mask, helper.code_map,
                num_codebooks, beam_width=beam_width, topk=beam_width,
            )

            # Metrics
            top1 = pred_ids[:, 0, :]  # [N, num_codebooks]
            per_code_match = (top1 == tgt)
            exact_match = per_code_match.all(dim=-1)

            for c in range(num_codebooks):
                metrics['correct'][c] += per_code_match[:, c].sum().item()
            metrics['total'] += N
            metrics['exact'] += exact_match.sum().item()

            pred_items = item_mapper.map_predictions(pred_ids)
            topk_acc.accumulate(
                target_items.unsqueeze(-1), pred_items.unsqueeze(-1)
            )
            sid_topk_acc.accumulate(tgt, pred_ids)

    if fsdp_ctx is not None:
        fsdp_ctx.__exit__(None, None, None)

    # All-reduce the top-k counts BEFORE computing ratios — otherwise the
    # returned Recall@K/NDCG@K cover only this rank's shard, and the is_best/
    # need_test decisions derived from them diverge across ranks (NCCL hang).
    topk_acc.sync_across_processes(accelerator)
    topk_metrics = topk_acc.reduce()
    sid_topk_acc.sync_across_processes(accelerator)
    metrics['sid_topk'] = sid_topk_acc.reduce()

    # Gather from all GPUs
    def gather(v):
        t = torch.tensor([v], device=device, dtype=torch.float32)
        return int(accelerator.reduce(t, reduction="sum").item())

    for c in range(num_codebooks):
        metrics['correct'][c] = gather(metrics['correct'][c])
    metrics['total'] = gather(metrics['total'])
    metrics['exact'] = gather(metrics['exact'])

    return metrics, topk_metrics


def log_clm_metrics(metrics, topk_metrics, num_codebooks, epoch, logger, wandb_log=False, prefix="eval"):
    """Log evaluation metrics."""
    total = metrics['total']
    if total == 0:
        return

    c_accs = [metrics['correct'][c] / total for c in range(num_codebooks)]
    exact = metrics['exact'] / total

    logger.info(f"Epoch {epoch} - exact={exact:.4f}, " +
                ", ".join([f"c{c}={c_accs[c]:.4f}" for c in range(num_codebooks)]))
    logger.info(f"Epoch {epoch} - TopK (ITEM-level): " +
                ", ".join([f"{k}={v:.4f}" for k, v in topk_metrics.items()]))
    sid_topk = metrics.get('sid_topk', {})
    if sid_topk:
        logger.info(f"Epoch {epoch} - TopK (SID-level pre-dedup diagnostic): " +
                    ", ".join([f"{k}={v:.4f}" for k, v in sid_topk.items()]))

    if wandb_log:
        log_dict = {"epoch": epoch, f"{prefix}/exact": exact}
        for c in range(num_codebooks):
            log_dict[f"{prefix}/c{c}"] = c_accs[c]
        for k, v in topk_metrics.items():
            log_dict[f"{prefix}/{k}"] = v
        for k, v in metrics.get('sid_topk', {}).items():
            log_dict[f"{prefix}/sid_ref_{k}"] = v
        wandb.log(log_dict)


@gin.configurable
def train_clm(
    epochs=5, batch_size=32, learning_rate=5e-5, weight_decay=0.01,
    warmup_steps=50, gradient_accumulate_every=2, max_length=256,
    pretrained_path="Qwen/Qwen2.5-1.5B", use_lora=False,
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    num_codebooks=3, codebook_size=256,
    dataset=None, dataset_folder="dataset/amazon", max_seq_len=50,
    do_eval=True, eval_every_epoch=1, eval_batch_size=64, eval_beam_width=10,
    save_dir_root="out/onerec/amazon/{split}/clm", save_every_epoch=1,
    wandb_logging=False, wandb_project="onerec_clm", wandb_run_name=None, wandb_log_interval=10,
    split_batches=True, amp=True, mixed_precision_type="bf16",
    max_train_samples=0, max_eval_samples=0, debug_logging=False,
    eval_only=False, checkpoint_path=None,
    early_stopping_patience=3,
    seed=42,
):
    """Train OneRec CLM model."""
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
    model = OneRec(pretrained_path=pretrained_path)
    model.add_codebook_tokens(num_codebooks=num_codebooks, codebook_size=codebook_size)
    model.add_item_sep_token()

    if use_lora:
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            model.model = get_peft_model(model.model, LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ))
            if accelerator.is_main_process:
                model.model.print_trainable_parameters()
        except ImportError:
            logger.warning("peft not installed, training without LoRA")

    model.gradient_checkpointing_enable()
    tokenizer = model.tokenizer
    token_mapper = CLMTokenMapper(tokenizer, num_codebooks, codebook_size)
    helper = CLMConstrainedDecodingHelper(num_codebooks, codebook_size, tokenizer)

    # Dataset setup (gin may bind another class via train_clm.dataset)
    dataset_cls = dataset or AmazonOneRecCLMDataset
    ds_kwargs = dict(root=dataset_folder, max_seq_len=max_seq_len)
    train_ds = dataset_cls(train_test_split="train", **ds_kwargs)
    valid_ds = dataset_cls(train_test_split="valid", **ds_kwargs)
    test_ds = dataset_cls(train_test_split="test", **ds_kwargs)
    _run_config["evaluation_protocol"] = train_ds.evaluation_protocol

    sem_ids_list = train_ds.sem_ids_list  # shared across splits
    item_mapper = SemanticIdItemMapper.from_sequences(
        sem_ids_list, train_ds.sequences
    )
    logger.info(
        "Honest SID resolver: most_popular(training-only), unique=%d/%d, "
        "collision_rate=%.2f%%, max_group=%d",
        item_mapper.num_unique_sids, item_mapper.num_items,
        100.0 * item_mapper.collision_rate, item_mapper.max_collision_group,
    )

    if max_train_samples > 0:
        train_ds.samples = train_ds.samples[:max_train_samples]
        logger.info(f"Limited train samples to {len(train_ds.samples)}")
    if max_eval_samples > 0:
        valid_ds.samples = valid_ds.samples[:max_eval_samples]
        test_ds.samples = test_ds.samples[:max_eval_samples]
        logger.info(f"Limited eval samples to {len(valid_ds.samples)}")

    collate_train = lambda x: clm_collate_fn(x, token_mapper, sem_ids_list, max_length)
    collate_eval = lambda x: clm_eval_collate_fn(x, token_mapper, sem_ids_list, max_length, num_codebooks)

    train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True, shuffle=True,
                          num_workers=4, pin_memory=True, collate_fn=collate_train)
    valid_dl = DataLoader(valid_ds, batch_size=eval_batch_size, shuffle=False,
                          num_workers=4, pin_memory=True, collate_fn=collate_eval)
    test_dl = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False,
                         num_workers=4, pin_memory=True, collate_fn=collate_eval)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_dl, valid_dl, test_dl = accelerator.prepare(train_dl, valid_dl, test_dl)

    total_steps = len(train_dl) * epochs // gradient_accumulate_every
    logger.info(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
    logger.info(f"Train: {len(train_ds)} samples, Valid: {len(valid_ds)}, Test: {len(test_ds)}")

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    logger.info(f"Device: {device}, Params: {sum(p.numel() for p in model.parameters()):,}, "
                f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if checkpoint_path:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        accelerator.unwrap_model(model).load_pretrained(checkpoint_path)

    if eval_only:
        logger.info("Running eval-only mode...")
        metrics, topk = evaluate_clm(model, valid_dl, accelerator, helper,
                                     num_codebooks, eval_beam_width, logger, 0,
                                     item_mapper=item_mapper)
        if accelerator.is_main_process:
            log_clm_metrics(metrics, topk, num_codebooks, 0, logger)
        accelerator.wait_for_everyone()
        return

    # Training loop with early stopping
    pbar = tqdm(total=total_steps, dynamic_ncols=True) if accelerator.is_main_process else None
    global_step = 0
    best_recall10 = float("-inf")
    best_epoch = -1
    patience_counter = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_steps = 0.0, 0

        for data in train_dl:
            with accelerator.accumulate(model):
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
            metrics, topk = evaluate_clm(model, valid_dl, accelerator, helper,
                                         num_codebooks, eval_beam_width, logger, epoch,
                                         item_mapper=item_mapper)

            current_recall10 = topk.get("Recall@10", 0.0)
            is_best = current_recall10 > best_recall10

            if accelerator.is_main_process:
                logger.info("--- Valid ---")
                log_clm_metrics(metrics, topk, num_codebooks, epoch, logger, wandb_logging, "eval")

            # Keep best_recall10 / patience_counter consistent across ranks
            # so subsequent is_best comparisons don't desync collective eval calls.
            if is_best:
                best_recall10 = current_recall10
                best_epoch = epoch
                patience_counter = 0
                unwrapped = accelerator.unwrap_model(model)
                trainable_names = {
                    name for name, parameter in unwrapped.named_parameters()
                    if parameter.requires_grad
                }
                current_state = accelerator.get_state_dict(model)
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in current_state.items()
                    if name in trainable_names
                }
                if accelerator.is_main_process:
                    save_path = os.path.join(save_dir_root, "checkpoint_best")
                    os.makedirs(save_path, exist_ok=True)
                    accelerator.unwrap_model(model).save_pretrained(save_path)
                    logger.info(f"New best Recall@10: {best_recall10:.4f}, saved to {save_path}")
            else:
                patience_counter += 1
                if accelerator.is_main_process:
                    logger.info(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")

            model.train()

            should_stop = torch.tensor([patience_counter >= early_stopping_patience], device=device)
            should_stop = accelerator.reduce(should_stop, reduction="max")
            if should_stop.item() >= 1:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Save checkpoint
        if accelerator.is_main_process and (epoch + 1) % save_every_epoch == 0:
            save_path = os.path.join(save_dir_root, f"checkpoint_epoch_{epoch}")
            os.makedirs(save_path, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(save_path)
            logger.info(f"Saved checkpoint to {save_path}")

    # Final save
    if accelerator.is_main_process:
        save_path = os.path.join(save_dir_root, "checkpoint_final")
        os.makedirs(save_path, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(save_path)
        logger.info(f"Saved final checkpoint to {save_path}")
        if pbar:
            pbar.close()

    accelerator.wait_for_everyone()
    if best_state is not None:
        accelerator.unwrap_model(model).load_state_dict(best_state, strict=False)
    test_metrics, test_topk = evaluate_clm(
        model, test_dl, accelerator, helper,
        num_codebooks, eval_beam_width, logger, best_epoch,
        item_mapper=item_mapper,
    )
    last_test_topk = dict(test_topk)
    last_test_sid_topk = dict(test_metrics.get('sid_topk', {}))
    last_test_exact = (
        test_metrics['exact'] / test_metrics['total']
        if test_metrics.get('total', 0) > 0 else 0.0
    )
    if accelerator.is_main_process:
        logger.info("--- Test (one-shot after validation selection) ---")
        log_clm_metrics(
            test_metrics, test_topk, num_codebooks, best_epoch, logger,
            wandb_log=False, prefix="test",
        )

    if wandb_logging and accelerator.is_main_process:
        wandb.log({
            "epoch": best_epoch,
            "test/exact": last_test_exact,
            **{f"test/{key}": value for key, value in test_topk.items()},
        })
        wandb.finish()

    if accelerator.is_main_process:
        save_run_results(
            save_dir=save_dir_root,
            model="onerec_clm",
            split=get_run_split(),
            seed=seed,
            metrics=last_test_topk,
            config=_run_config,
            extra={
                "best_valid_recall@10": best_recall10,
                "best_epoch": best_epoch,
                "test_policy": "one_shot_after_validation_selection",
                "evaluation_level": "item",
                "sid_resolution": "most_popular_training_only",
                "sid_reference_metrics": last_test_sid_topk,
                "test_exact": last_test_exact,
            },
        )

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parse_config()
    train_clm()
