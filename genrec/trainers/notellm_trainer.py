"""Train and evaluate NoteLLM with exact full-catalog item retrieval."""

from __future__ import annotations

import math
import os
from typing import Dict, List, Sequence

import gin
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup

from genrec.data.amazon_notellm import (
    AmazonNoteLLMDataset,
    format_item_csft_prompt,
    format_item_prompt,
    notellm_collate_fn,
)
from genrec.models.notellm import Query2Embedding
from genrec.modules.utils import get_run_split, parse_config, setup_logger
from genrec.trainers.trainer_utils import (
    save_run_results,
    set_seed,
    setup_accelerator,
    setup_wandb,
)


def _device_tokens(model: Query2Embedding, prompts: List[str], device, max_length: int):
    tokens = model.tokenize(
        prompts, truncation=True, max_length=max_length
    )
    return {key: value.to(device) for key, value in tokens.items()}


def _metric_case(metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        key.replace("recall", "Recall").replace("ndcg", "NDCG"): value
        for key, value in metrics.items()
    }


def _trainable_checkpoint(model, accelerator) -> Dict[str, torch.Tensor]:
    """Return only trainable parameters for a compact, exact best checkpoint."""
    unwrapped = accelerator.unwrap_model(model)
    trainable_names = {
        name for name, parameter in unwrapped.named_parameters()
        if parameter.requires_grad
    }
    full_state = accelerator.get_state_dict(model)
    return {
        name: tensor.detach().cpu()
        for name, tensor in full_state.items()
        if name in trainable_names
    }


@torch.no_grad()
def evaluate_full_catalog(
    model,
    dataloader,
    dataset: AmazonNoteLLMDataset,
    accelerator,
    max_text_length: int,
    catalog_batch_size: int,
    topks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Encode every item once and rank exact item IDs for every query."""
    model.eval()
    raw_model: Query2Embedding = accelerator.unwrap_model(model)
    device = accelerator.device
    item_ids = torch.arange(1, dataset.num_items + 1, device=device)

    catalog_parts = []
    for start in tqdm(
        range(0, dataset.num_items, catalog_batch_size),
        desc="Encoding NoteLLM catalog",
        disable=not accelerator.is_main_process,
    ):
        batch_ids = range(start + 1, min(dataset.num_items, start + catalog_batch_size) + 1)
        prompts = [
            format_item_prompt(dataset.item_texts[item_id], raw_model.item_token)
            for item_id in batch_ids
        ]
        tokens = _device_tokens(raw_model, prompts, device, max_text_length)
        catalog_parts.append(model(**tokens, return_loss=False)["sentence_embedding"])
    catalog_embeddings = torch.cat(catalog_parts, dim=0)

    query_parts = []
    query_id_parts = []
    target_parts = []
    for batch in tqdm(
        dataloader,
        desc="Encoding NoteLLM queries",
        disable=not accelerator.is_main_process,
    ):
        prompts = [
            format_item_prompt(dataset.item_texts[int(item_id)], raw_model.item_token)
            for item_id in batch["queries"]
        ]
        tokens = _device_tokens(raw_model, prompts, device, max_text_length)
        embeddings = model(**tokens, return_loss=False)["sentence_embedding"]
        query_ids = torch.as_tensor(
            batch["queries"], device=device, dtype=torch.long
        )
        targets = torch.as_tensor(batch["targets"], device=device, dtype=torch.long)
        query_parts.append(accelerator.gather_for_metrics(embeddings))
        query_id_parts.append(accelerator.gather_for_metrics(query_ids))
        target_parts.append(accelerator.gather_for_metrics(targets))

    queries = torch.cat(query_parts, dim=0)
    query_ids = torch.cat(query_id_parts, dim=0)
    targets = torch.cat(target_parts, dim=0)
    metrics = raw_model.compute_full_catalog_metrics(
        query_embeddings=queries,
        catalog_embeddings=catalog_embeddings,
        target_item_ids=targets,
        catalog_item_ids=item_ids,
        excluded_item_ids=query_ids,
        topks=topks,
        query_batch_size=catalog_batch_size,
    )
    model.train()
    return _metric_case(metrics)


@gin.configurable
def train(
    epochs: int = 5,
    batch_size: int = 4,
    gradient_accumulate_every: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    pretrained_path: str = "Qwen/Qwen2.5-1.5B",
    freeze_lm: bool = False,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    dataset_folder: str = "dataset/amazon",
    split: str = "beauty",
    max_seq_len: int = 50,
    max_text_length: int = 256,
    eval_batch_size: int = 32,
    catalog_batch_size: int = 64,
    patience: int = 2,
    enable_csft: bool = True,
    csft_alpha: float = 0.01,
    save_dir_root: str = "out/notellm/amazon/{split}",
    wandb_logging: bool = False,
    wandb_project: str = "notellm_training",
    wandb_run_name: str | None = None,
    amp: bool = True,
    mixed_precision_type: str = "bf16",
    seed: int = 42,
):
    if epochs <= 0 or batch_size <= 0 or eval_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if max_text_length <= 0:
        raise ValueError("text length must be positive")
    if csft_alpha < 0:
        raise ValueError("csft_alpha cannot be negative")

    save_dir_root = save_dir_root.replace("{split}", split)
    run_config = dict(locals())
    logger = setup_logger(save_dir_root, name="notellm")
    set_seed(seed)
    accelerator = setup_accelerator(
        gradient_accumulate_every=gradient_accumulate_every,
        amp=amp,
        mixed_precision_type=mixed_precision_type,
    )
    device = accelerator.device

    if wandb_logging and accelerator.is_main_process:
        setup_wandb(
            project=wandb_project,
            run_name=wandb_run_name,
            config=run_config,
            step_metrics={"train/*": "global_step", "eval/*": "epoch"},
        )

    train_ds = AmazonNoteLLMDataset(
        root=dataset_folder,
        split=split,
        train_test_split="train",
        max_seq_len=max_seq_len,
    )
    valid_ds = AmazonNoteLLMDataset(
        root=dataset_folder,
        split=split,
        train_test_split="valid",
        max_seq_len=max_seq_len,
    )
    test_ds = AmazonNoteLLMDataset(
        root=dataset_folder,
        split=split,
        train_test_split="test",
        max_seq_len=max_seq_len,
    )
    run_config["evaluation_protocol"] = train_ds.evaluation_protocol

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=notellm_collate_fn,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=notellm_collate_fn,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=notellm_collate_fn,
    )

    model = Query2Embedding(
        pretrained_path=pretrained_path,
        freeze_lm=freeze_lm,
        gradient_checkpointing=not freeze_lm,
        alpha=csft_alpha,
    )
    if use_lora and not freeze_lm:
        try:
            from peft import LoraConfig, TaskType, get_peft_model

            model.model = get_peft_model(
                model.model,
                LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=[
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                    ],
                ),
            )
            model.model.enable_input_require_grads()
        except ImportError as exc:
            raise RuntimeError(
                "train.use_lora=True requires the optional 'peft' dependency; "
                "refusing to silently fall back to full-parameter training"
            ) from exc

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    update_steps = max(
        1,
        math.ceil(len(train_dl) / max(gradient_accumulate_every, 1)) * epochs,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(update_steps * warmup_ratio),
        num_training_steps=update_steps,
    )
    model, optimizer, scheduler, train_dl, valid_dl, test_dl = accelerator.prepare(
        model, optimizer, scheduler, train_dl, valid_dl, test_dl
    )
    raw_model: Query2Embedding = accelerator.unwrap_model(model)
    logger.info(
        "NoteLLM protocol=%s, items=%d, train=%d, valid=%d, test=%d, "
        "trainable_params=%d",
        train_ds.evaluation_protocol,
        train_ds.num_items,
        len(train_ds),
        len(valid_ds),
        len(test_ds),
        sum(parameter.numel() for parameter in trainable),
    )

    best_valid = float("-inf")
    best_epoch = -1
    stale_epochs = 0
    best_path = os.path.join(save_dir_root, "best_model.pt")
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in tqdm(
            train_dl,
            desc=f"NoteLLM train epoch {epoch}",
            disable=not accelerator.is_main_process,
        ):
            pair_prompts: List[str] = []
            pair_targets: List[str] = []
            for query, target in zip(batch["queries"], batch["targets"]):
                pair_ids = (int(query), int(target))
                if enable_csft:
                    pair_prompts.extend([
                        format_item_csft_prompt(
                            train_ds.item_texts[item_id], raw_model.item_token
                        )
                        for item_id in pair_ids
                    ])
                    pair_targets.extend([
                        train_ds.item_categories[item_id] for item_id in pair_ids
                    ])
                else:
                    pair_prompts.extend([
                        format_item_prompt(
                            train_ds.item_texts[item_id], raw_model.item_token
                        )
                        for item_id in pair_ids
                    ])
            if enable_csft:
                tokens = raw_model.tokenize_with_targets(
                    pair_prompts, pair_targets, max_text_length
                )
                tokens = {key: value.to(device) for key, value in tokens.items()}
            else:
                tokens = _device_tokens(
                    raw_model, pair_prompts, device, max_text_length
                )
            tokens["pair_item_ids"] = torch.tensor(
                [item_id for pair in zip(batch["queries"], batch["targets"])
                 for item_id in pair],
                device=device,
                dtype=torch.long,
            )

            with accelerator.accumulate(model):
                output = model(**tokens, return_loss=True)
                loss = output["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            total_loss += float(loss.detach())
            batches += 1

        valid_metrics = evaluate_full_catalog(
            model,
            valid_dl,
            valid_ds,
            accelerator,
            max_text_length,
            catalog_batch_size,
        )
        valid_recall = valid_metrics["Recall@10"]
        logger.info(
            "Epoch %d loss=%.6f valid=%s",
            epoch,
            total_loss / max(batches, 1),
            valid_metrics,
        )

        if valid_recall > best_valid:
            best_valid = valid_recall
            best_epoch = epoch
            stale_epochs = 0
            accelerator.wait_for_everyone()
            state = _trainable_checkpoint(model, accelerator)
            if accelerator.is_main_process:
                os.makedirs(save_dir_root, exist_ok=True)
                torch.save(
                    {
                        "trainable_state_dict": state,
                        "best_epoch": epoch,
                        "best_valid_recall@10": best_valid,
                    },
                    best_path,
                )
                raw_model.tokenizer.save_pretrained(
                    os.path.join(save_dir_root, "tokenizer")
                )
                if use_lora and hasattr(raw_model.model, "peft_config"):
                    raw_model.model.save_pretrained(
                        os.path.join(save_dir_root, "adapter")
                    )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.info("Early stopping after epoch %d", epoch)
                break

    accelerator.wait_for_everyone()
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    accelerator.unwrap_model(model).load_state_dict(
        checkpoint["trainable_state_dict"], strict=False
    )
    test_metrics = evaluate_full_catalog(
        model,
        test_dl,
        test_ds,
        accelerator,
        max_text_length,
        catalog_batch_size,
    )
    if accelerator.is_main_process:
        logger.info("Final exact full-catalog test: %s", test_metrics)
        save_run_results(
            save_dir=save_dir_root,
            model="notellm",
            split=get_run_split(default=split),
            seed=seed,
            metrics=test_metrics,
            config=run_config,
            extra={
                "evaluation_level": "item",
                "candidate_set": "exact_full_catalog_excluding_query_item",
                "query_semantics": "last_history_item_i2i",
                "training_variant": (
                    "gcl_csft_category_amazon_adaptation"
                    if enable_csft else "gcl_only_amazon_adaptation"
                ),
                "gcl_negatives": (
                    "all_ranks_all_nonself_rows_masking_exact_item_duplicates"
                ),
                "embedding_position": "token_before_emb_marker",
                "csft_alpha": csft_alpha if enable_csft else 0.0,
                "best_valid_recall@10": best_valid,
                "best_epoch": best_epoch,
            },
        )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    parse_config()
    train()
