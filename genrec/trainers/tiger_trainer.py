"""
TIGER trainer (HF T5-based).

Pipeline:
- Flat offset-encoded codebook tokens (token = code + codebook_idx * codebook_size + 1).
- HF T5ForConditionalGeneration via genrec.models.tiger.Tiger.
- Plain Adam (default) or cosine schedule with linear warmup.
- Early stopping on validation Recall@10.
"""

import math
import os
import gin
import torch
import wandb
from transformers import Adafactor

from genrec.models.tiger import Tiger
from genrec.modules.utils import parse_config, setup_logger, get_run_split
from genrec.modules.metrics import SemanticIdItemMapper, TopKAccumulator
from genrec.data.schemas import SeqData
from genrec.trainers.trainer_utils import setup_wandb, set_seed, save_run_results
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict


def t5_collate(
    batch: List[SeqData],
    codebook_size: int = 256,
    sem_id_dim: int = 3,
    max_history_tokens: int = 60,  # max_seq_len * sem_id_dim (e.g. 20*3=60)
    include_user_token: bool = False,
    num_user_tokens: int = 2000,
    padding_side: str = "left",
) -> Dict[str, torch.Tensor]:
    """
    Convert SeqData to flat T5 format with offset encoding.

    Offset encoding: token = code + codebook_idx * codebook_size + 1
    (0 is reserved for padding)
    """
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'")
    if include_user_token and num_user_tokens <= 0:
        raise ValueError("num_user_tokens must be positive when user tokens are enabled")

    B = len(batch)
    total_input_tokens = max_history_tokens + int(include_user_token)

    input_ids = torch.zeros(B, total_input_tokens, dtype=torch.long)
    attention_mask = torch.zeros(B, total_input_tokens, dtype=torch.long)
    label_list = []

    for i, sample in enumerate(batch):
        raw_ids = sample.item_ids  # flat list: [c0, c1, c2, c0, c1, c2, ...]
        if len(raw_ids) % sem_id_dim:
            raise ValueError(
                "flattened history length must be divisible by sem_id_dim: "
                f"len={len(raw_ids)}, sem_id_dim={sem_id_dim}"
            )
        if any(code < 0 or code >= codebook_size for code in raw_ids):
            raise ValueError("history contains a code outside the codebook range")
        offset_ids = []
        for j, code in enumerate(raw_ids):
            cb_idx = j % sem_id_dim
            offset_ids.append(code + cb_idx * codebook_size + 1)

        if len(offset_ids) > max_history_tokens:
            offset_ids = offset_ids[-max_history_tokens:]

        sequence_ids = offset_ids
        if include_user_token:
            # SID tokens occupy [1, codebook_size * sem_id_dim]. User tokens
            # use the following, disjoint range, matching TIGER's 1024 + 2000
            # vocabulary layout for 4x256 item IDs.
            user_token = (
                codebook_size * sem_id_dim
                + 1
                + int(sample.user_id) % num_user_tokens
            )
            sequence_ids = [user_token] + sequence_ids

        L = len(sequence_ids)
        start = total_input_tokens - L if padding_side == "left" else 0
        end = start + L
        input_ids[i, start:end] = torch.tensor(sequence_ids, dtype=torch.long)
        attention_mask[i, start:end] = 1

        target_ids = sample.target_ids
        if len(target_ids) != sem_id_dim:
            raise ValueError(
                "target SID width must equal sem_id_dim: "
                f"len={len(target_ids)}, sem_id_dim={sem_id_dim}"
            )
        if any(code < 0 or code >= codebook_size for code in target_ids):
            raise ValueError("target contains a code outside the codebook range")
        offset_target = []
        for j, code in enumerate(target_ids):
            cb_idx = j % sem_id_dim
            offset_target.append(code + cb_idx * codebook_size + 1)
        label_list.append(offset_target)

    labels = torch.tensor(label_list, dtype=torch.long)
    raw_targets = torch.tensor([s.target_ids for s in batch], dtype=torch.long)
    target_items = torch.tensor([s.target_item for s in batch], dtype=torch.long)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'raw_targets': raw_targets,
        'target_items': target_items,
    }


def inverse_sqrt_lr_lambda(step: int, warmup_steps: int) -> float:
    """TIGER schedule: constant through warmup, then inverse square root."""
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be positive for inverse_sqrt")
    effective_step = max(step, 1)
    if effective_step <= warmup_steps:
        return 1.0
    return math.sqrt(warmup_steps / effective_step)


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
    max_steps=0,
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
    include_user_token=False,
    num_user_tokens=2000,
    padding_side="left",
    pretrained_rqvae_path="./out/tiger/amazon/{split}/rqvae/checkpoint_epoch_4999.pt",
    # Scheduler config
    optimizer_name="adam",
    lr_schedule="none",  # "none" or "cosine"
    num_warmup_steps=0,
    z_loss=0.0,
    # Eval config
    beam_size=30,
    early_stop_patience=10,
    eval_test_every_epoch=2,
    eval_valid_every_n_epochs=1,
    eval_every_steps=0,
    test_during_training=False,
    resume_from_checkpoint=None,
    save_training_state_every_steps=0,
    seed=42,
):
    _run_config = dict(locals())
    logger = setup_logger(save_dir_root, name="tiger")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if sem_id_dim <= 0 or codebook_size <= 0:
        raise ValueError("sem_id_dim and codebook_size must be positive")
    if include_user_token and num_user_tokens <= 0:
        raise ValueError("num_user_tokens must be positive when user tokens are enabled")
    if (
        max_steps < 0
        or eval_every_steps < 0
        or save_training_state_every_steps < 0
    ):
        raise ValueError(
            "max_steps, eval_every_steps and save_training_state_every_steps "
            "cannot be negative"
        )

    vocab_size = (
        codebook_size * sem_id_dim
        + (num_user_tokens if include_user_token else 0)
        + 1
    )

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
    test_dataset = None
    if test_during_training:
        test_dataset = dataset(
            root=dataset_folder,
            train_test_split="test",
            max_seq_len=max_seq_len,
            pretrained_rqvae_path=pretrained_rqvae_path,
        )
    _run_config["evaluation_protocol"] = train_dataset.evaluation_protocol

    observed_sid_dim = len(train_dataset.sem_ids_list[0])
    if observed_sid_dim != sem_id_dim:
        raise ValueError(
            "dataset/model Semantic ID width mismatch: "
            f"dataset={observed_sid_dim}, train.sem_id_dim={sem_id_dim}. "
            "When add_disambiguation=True, sem_id_dim must include the suffix."
        )
    item_mapper = SemanticIdItemMapper.from_sequences(
        train_dataset.sem_ids_list, train_dataset.sequences
    )
    raw_mapper = SemanticIdItemMapper(
        getattr(train_dataset, "raw_sem_ids_list", train_dataset.sem_ids_list)
    )
    logger.info(
        "Raw RQ SID collisions: unique=%d/%d, collision_rate=%.2f%%, "
        "max_group=%d",
        raw_mapper.num_unique_sids, raw_mapper.num_items,
        100.0 * raw_mapper.collision_rate, raw_mapper.max_collision_group,
    )
    logger.info(
        "Final SID resolver: unique=%d/%d, collision_rate=%.2f%%, max_group=%d",
        item_mapper.num_unique_sids, item_mapper.num_items,
        100.0 * item_mapper.collision_rate, item_mapper.max_collision_group,
    )

    max_history_tokens = max_seq_len * sem_id_dim
    collate_fn = lambda x: t5_collate(
        x,
        codebook_size,
        sem_id_dim,
        max_history_tokens,
        include_user_token=include_user_token,
        num_user_tokens=num_user_tokens,
        padding_side=padding_side,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=infer_batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=infer_batch_size, shuffle=False,
            num_workers=4, pin_memory=True, collate_fn=collate_fn,
        )

    logger.info(
        "Train: %d batches, Valid: %d, Test: %s",
        len(train_loader), len(valid_loader),
        len(test_loader) if test_loader is not None else "disabled",
    )

    # Model
    model = Tiger(model_config).to(device)
    total_params, emb_params = model.num_parameters
    logger.info(f"Device: {device}, Params: {total_params:,} (emb: {emb_params:,})")

    # Optimizer. The paper does not name the seq2seq optimizer; retain Adam as
    # the explicit reproducible choice while aligning its published LR curve.
    if optimizer_name == "adam":
        optimizer = Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "adafactor":
        # TIGER is implemented with T5X.  Its standard T5 optimizer is
        # Adafactor with parameter-scale multiplication, no momentum and an
        # externally supplied learning-rate schedule.  Hugging Face uses a
        # negative exponent for the same 0.8 second-moment decay convention.
        optimizer = Adafactor(
            model.parameters(),
            lr=learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=weight_decay,
            scale_parameter=True,
            relative_step=False,
            warmup_init=False,
        )
    else:
        raise ValueError(f"unsupported optimizer_name={optimizer_name!r}")

    # LR scheduler
    scheduler = None
    planned_steps = max_steps if max_steps > 0 else len(train_loader) * epochs
    if lr_schedule == "cosine" and planned_steps > 0:
        if num_warmup_steps > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=num_warmup_steps)
            cosine_scheduler = CosineAnnealingLR(
                optimizer, T_max=max(planned_steps - num_warmup_steps, 1)
            )
            scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[num_warmup_steps])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=planned_steps)
        logger.info(
            "Using cosine schedule, warmup=%d, total_steps=%d",
            num_warmup_steps, planned_steps,
        )
    elif lr_schedule == "inverse_sqrt":
        if num_warmup_steps <= 0:
            raise ValueError("inverse_sqrt requires num_warmup_steps > 0")
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: inverse_sqrt_lr_lambda(
                step, num_warmup_steps
            ),
        )
        logger.info(
            "Using constant-then-inverse-sqrt schedule, base_lr=%g, "
            "constant_steps=%d, total_steps=%d",
            learning_rate, num_warmup_steps, planned_steps,
        )
    elif lr_schedule == "none":
        logger.info("Using plain Adam, no scheduler")
    else:
        raise ValueError(f"unsupported lr_schedule={lr_schedule!r}")

    # Early stopping state
    # Recall is bounded below by zero.  Starting below its range ensures that
    # even a deliberately tiny smoke run writes a usable best checkpoint.
    best_valid_recall10 = -1.0
    best_epoch = -1
    best_step = -1
    early_stop_counter = 0
    best_valid_metrics: Dict[str, float] = {}
    best_valid_sid_metrics: Dict[str, float] = {}
    best_valid_diagnostics: Dict[str, float] = {}
    best_test_metrics: Dict[str, float] = {}
    best_test_sid_metrics: Dict[str, float] = {}

    start_epoch = 0
    global_step = 0
    if resume_from_checkpoint:
        state = torch.load(
            resume_from_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state.get("epoch", -1)) + 1
        global_step = int(state.get("global_step", 0))
        best_valid_recall10 = float(
            state.get("best_valid_recall10", best_valid_recall10)
        )
        best_epoch = int(state.get("best_epoch", best_epoch))
        best_step = int(state.get("best_step", best_step))
        best_valid_metrics = dict(
            state.get("best_valid_metrics", best_valid_metrics)
        )
        best_valid_sid_metrics = dict(
            state.get("best_valid_sid_metrics", best_valid_sid_metrics)
        )
        best_valid_diagnostics = dict(
            state.get("best_valid_diagnostics", best_valid_diagnostics)
        )
        logger.info(
            "Resumed training state from %s at epoch %d, step %d",
            resume_from_checkpoint, start_epoch, global_step,
        )

    def save_training_state(epoch: int) -> None:
        os.makedirs(save_dir_root, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best_valid_recall10": best_valid_recall10,
                "best_epoch": best_epoch,
                "best_step": best_step,
                "best_valid_metrics": best_valid_metrics,
                "best_valid_sid_metrics": best_valid_sid_metrics,
                "best_valid_diagnostics": best_valid_diagnostics,
                "model_config": model_config,
            },
            os.path.join(save_dir_root, "latest_training_state.pt"),
        )

    def evaluate(loader, desc="Eval"):
        model.eval()
        item_acc = TopKAccumulator(ks=[5, 10])
        sid_acc = TopKAccumulator(ks=[5, 10])
        total_generated = 0
        valid_generated = 0
        coverage_at_5 = 0
        coverage_at_10 = 0
        total_examples = 0

        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            for batch in tqdm(loader, desc=desc):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']  # offset-encoded targets
                target_items = batch['target_items']
                if (target_items < 0).any():
                    raise ValueError("TIGER eval batch is missing raw target item IDs")
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
                sid_acc.accumulate(labels, preds)

                # Decode position-offset tokens back to raw semantic IDs, map
                # each SID to one deployable item, then deduplicate the ranking.
                raw_pred_sids = preds.clone()
                for codebook_idx in range(sem_id_dim):
                    raw_pred_sids[:, :, codebook_idx] -= (
                        codebook_idx * codebook_size + 1
                    )
                pred_items = item_mapper.map_predictions(raw_pred_sids)
                raw_rows = raw_pred_sids.tolist()
                total_generated += B_cur * beam_size
                valid_generated += sum(
                    tuple(int(v) for v in sid) in item_mapper.sid_to_item
                    for row in raw_rows
                    for sid in row
                )
                valid_per_row = pred_items.ne(item_mapper.padding_item_id).sum(1)
                coverage_at_5 += valid_per_row.ge(5).sum().item()
                coverage_at_10 += valid_per_row.ge(10).sum().item()
                total_examples += B_cur
                item_acc.accumulate(
                    target_items.unsqueeze(-1), pred_items.unsqueeze(-1)
                )

        metrics = item_acc.reduce()
        sid_metrics = sid_acc.reduce()
        diagnostics = {
            "invalid_sid_rate": (
                1.0 - valid_generated / total_generated
                if total_generated else 0.0
            ),
            "valid_candidates_at_least_5_rate": (
                coverage_at_5 / total_examples if total_examples else 0.0
            ),
            "valid_candidates_at_least_10_rate": (
                coverage_at_10 / total_examples if total_examples else 0.0
            ),
        }
        logger.info(
            f"{desc} SID-level pre-dedup diagnostic (not directly comparable): "
            f"{sid_metrics}"
        )
        logger.info(f"{desc} generation diagnostics: {diagnostics}")
        model.train()
        return metrics, sid_metrics, diagnostics

    # Training loop. ``max_steps`` takes precedence so the paper's 200k-step
    # budget is independent of dataset size or sliding-window count.
    max_epochs = epochs
    if max_steps > 0:
        max_epochs = max(epochs, math.ceil(max_steps / len(train_loader)))
    next_eval_step = None
    if eval_every_steps > 0:
        next_eval_step = (
            (global_step // eval_every_steps) + 1
        ) * eval_every_steps
    next_state_save_step = None
    if save_training_state_every_steps > 0:
        next_state_save_step = (
            (global_step // save_training_state_every_steps) + 1
        ) * save_training_state_every_steps

    for epoch in range(start_epoch, max_epochs):
        model.train()
        total_loss = 0.0
        batches_this_epoch = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            if max_steps > 0 and global_step >= max_steps:
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            if z_loss:
                # T5X's standard T5 loss includes a small log-normalizer
                # penalty.  It is especially useful with the paper's large
                # nominal learning rate and is zero by default for legacy runs.
                log_z = torch.logsumexp(logits.float(), dim=-1)
                loss = loss + z_loss * log_z.square().mean()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
            batches_this_epoch += 1
            global_step += 1

        if batches_this_epoch == 0:
            break
        avg_loss = total_loss / batches_this_epoch
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d - step %d - loss: %.4f - lr: %.6g",
            epoch, global_step, avg_loss, current_lr,
        )

        log_dict = {
            "epoch": epoch,
            "step": global_step,
            "train/loss": avg_loss,
            "train/lr": current_lr,
        }

        reached_final_step = max_steps > 0 and global_step >= max_steps
        if next_eval_step is not None:
            should_eval = global_step >= next_eval_step or reached_final_step
            while next_eval_step <= global_step:
                next_eval_step += eval_every_steps
        else:
            should_eval = (
                (epoch + 1) % eval_valid_every_n_epochs == 0 or epoch == 0
            )

        if should_eval:
            valid_metrics, valid_sid_metrics, valid_diagnostics = evaluate(
                valid_loader, desc=f"Valid (step {global_step})"
            )
            logger.info(
                "Epoch %d - step %d - Valid: %s",
                epoch, global_step, valid_metrics,
            )

            for k, v in valid_metrics.items():
                log_dict[f"eval/valid_{k}"] = v
            for k, v in valid_diagnostics.items():
                log_dict[f"eval/valid_{k}"] = v

            cur_recall10 = valid_metrics['Recall@10']

            # Check improvement
            if cur_recall10 > best_valid_recall10:
                best_valid_recall10 = cur_recall10
                best_epoch = epoch
                best_step = global_step
                best_valid_metrics = dict(valid_metrics)
                best_valid_sid_metrics = dict(valid_sid_metrics)
                best_valid_diagnostics = dict(valid_diagnostics)
                early_stop_counter = 0

                # Save best model
                os.makedirs(save_dir_root, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir_root, "best_model.pt"))
                logger.info(
                    "New best Valid R@10=%.4f at epoch %d, step %d",
                    cur_recall10, epoch, global_step,
                )

                # Legacy behavior remains available, but paper-alignment
                # searches disable it so test is untouched until config freeze.
                if test_loader is not None:
                    test_metrics, test_sid_metrics, _ = evaluate(
                        test_loader, desc=f"Test (step {global_step})"
                    )
                    logger.info(
                        "Epoch %d - step %d - Test: %s",
                        epoch, global_step, test_metrics,
                    )
                    for k, v in test_metrics.items():
                        log_dict[f"eval/test_{k}"] = v
                    best_test_metrics = dict(test_metrics)
                    best_test_sid_metrics = dict(test_sid_metrics)
            else:
                early_stop_counter += 1
                logger.info(f"No improvement. Counter: {early_stop_counter}/{early_stop_patience}")

                # Still run test periodically
                if (
                    test_loader is not None
                    and eval_test_every_epoch > 0
                    and (epoch + 1) % eval_test_every_epoch == 0
                ):
                    test_metrics, test_sid_metrics, _ = evaluate(
                        test_loader, desc=f"Test (step {global_step})"
                    )
                    logger.info(f"Epoch {epoch} - Test: {test_metrics}")
                    for k, v in test_metrics.items():
                        log_dict[f"eval/test_{k}"] = v

        if wandb_logging:
            wandb.log(log_dict)

        if (
            next_state_save_step is not None
            and (global_step >= next_state_save_step or reached_final_step)
        ):
            save_training_state(epoch)
            logger.info(
                "Saved resumable training state at epoch %d, step %d",
                epoch, global_step,
            )
            while next_state_save_step <= global_step:
                next_state_save_step += save_training_state_every_steps

        if early_stop_counter >= early_stop_patience:
            logger.info(
                "Early stopping at epoch %d, step %d. Best epoch=%d, step=%d",
                epoch, global_step, best_epoch, best_step,
            )
            break
        if reached_final_step:
            break

    logger.info(
        "Training done at step %d. Best Valid R@10=%.4f at epoch %d, step %d",
        global_step, best_valid_recall10, best_epoch, best_step,
    )

    if not test_during_training:
        # Default benchmark policy: select solely on validation, then touch the
        # test split once with the frozen best checkpoint.
        best_path = os.path.join(save_dir_root, "best_model.pt")
        if not os.path.exists(best_path):
            raise RuntimeError("TIGER training finished without a best checkpoint")
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_dataset = dataset(
            root=dataset_folder,
            train_test_split="test",
            max_seq_len=max_seq_len,
            pretrained_rqvae_path=pretrained_rqvae_path,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=infer_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        best_test_metrics, best_test_sid_metrics, _ = evaluate(
            test_loader, desc="Test (one-shot after validation selection)"
        )
        logger.info("One-shot test at best step %d: %s", best_step, best_test_metrics)

    if wandb_logging:
        if best_test_metrics:
            wandb.log({
                "epoch": best_epoch,
                "step": best_step,
                **{f"test/{key}": value for key, value in best_test_metrics.items()},
            })
        wandb.finish()

    save_run_results(
        save_dir=save_dir_root,
        model="tiger",
        split=get_run_split(),
        seed=seed,
        metrics=best_test_metrics,
        config=_run_config,
        extra={
            "best_valid_recall@10": best_valid_recall10,
            "best_epoch": best_epoch,
            "best_step": best_step,
            "best_valid_metrics": best_valid_metrics,
            "best_valid_sid_metrics": best_valid_sid_metrics,
            "best_valid_generation_diagnostics": best_valid_diagnostics,
            "evaluation_level": "item",
            "result_scope": (
                "test" if best_test_metrics else "validation_only"
            ),
            "raw_sid_collision_rate": raw_mapper.collision_rate,
            "final_sid_collision_rate": item_mapper.collision_rate,
            "sid_resolution": (
                "unique_suffix"
                if item_mapper.collision_rate == 0.0
                else "most_popular_training_only"
            ),
            "sid_reference_metrics": best_test_sid_metrics,
        },
    )


if __name__ == "__main__":
    parse_config()
    train()
