"""
Trainer for the COBRA model.
COBRA: Sparse Meets Dense - Unified Generative Recommendations with Cascaded Sparse-Dense Representations
"""
import os
import gin
import torch
import wandb

from genrec.models.cobra import Cobra
from genrec.modules.utils import parse_config, setup_logger, get_run_split
from genrec.modules.metrics import TopKAccumulator
from genrec.trainers.trainer_utils import setup_accelerator, setup_wandb, set_seed, save_run_results
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Any, Optional
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
        item_ids = torch.full((B, max_items), -1, dtype=torch.long)
        encoder_input_ids = torch.zeros((B, max_items, max_text_len), dtype=torch.long)
        target_sem_ids = torch.zeros((B, n_codebooks), dtype=torch.long)

        for i, sample in enumerate(batch):
            history_len = len(sample['input_ids'])
            n_history_items = history_len // n_codebooks

            # Append target to history semantic IDs
            full_sem_ids = sample['input_ids'] + sample['target_sem_ids']
            input_ids[i, :len(full_sem_ids)] = torch.tensor(full_sem_ids)
            full_item_ids = sample['history_item_ids'] + [sample['target_item']]
            item_ids[i, :len(full_item_ids)] = torch.tensor(full_item_ids)

            # Append target encoder input to history
            encoder_input_ids[i, :n_history_items] = sample['encoder_input_ids']
            encoder_input_ids[i, n_history_items:n_history_items+1] = sample['target_encoder_input_ids']

            # Target (for reference, though not used in loss since it's in input_ids)
            target_sem_ids[i] = torch.tensor(sample['target_sem_ids'])
    else:
        # Eval: keep history and target separate
        max_items = max(len(x['input_ids']) // n_codebooks for x in batch)

        input_ids = torch.full((B, max_items * n_codebooks), pad_id, dtype=torch.long)
        item_ids = torch.full((B, max_items), -1, dtype=torch.long)
        encoder_input_ids = torch.zeros((B, max_items, max_text_len), dtype=torch.long)
        target_sem_ids = torch.zeros((B, n_codebooks), dtype=torch.long)

        for i, sample in enumerate(batch):
            seq_len = len(sample['input_ids'])
            n_items = seq_len // n_codebooks

            input_ids[i, :seq_len] = torch.tensor(sample['input_ids'])
            item_ids[i, :n_items] = torch.tensor(sample['history_item_ids'])
            encoder_input_ids[i, :n_items] = sample['encoder_input_ids']
            target_sem_ids[i] = torch.tensor(sample['target_sem_ids'])

    target_item_ids = torch.tensor([s['target_item'] for s in batch], dtype=torch.long)

    return {
        'input_ids': input_ids,
        'item_ids': item_ids,
        'encoder_input_ids': encoder_input_ids,
        'target_sem_ids': target_sem_ids,
        'target_item_ids': target_item_ids,  # ground-truth item id for honest item-level eval
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
    max_train_batches: Optional[int] = None,
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
    encoder_pooling: str = "mean",
    dense_negative_scope: str = "cross_sequence",
    use_dense_memory_queue: bool = False,
    # Training
    num_warmup_steps: int = 500,
    max_seq_len: int = 20,
    pretrained_rqvae_path: str = "./out/rqvae/amazon/beauty/checkpoint.pt",
    encoder_model_name: str = "sentence-transformers/sentence-t5-xl",
    resume_from_checkpoint: str = None,
    # Loss weights
    sparse_loss_weight: float = 1.0,
    dense_loss_weight: float = 1.0,
    # BeamFusion inference.  Catalog constraints keep the finite beam budget on
    # semantic IDs that can resolve to real items.
    eval_n_beam: int = 32,
    eval_tau: float = 1.0,
    eval_psi: float = 16.0,
    constrain_to_catalog: bool = False,
    # Paper's "w/o Dense" ablation: train the sparse branch alone (dense inputs
    # zeroed, InfoNCE off). Eval-side BeamFusion then ranks buckets with the
    # untrained pretrained-encoder table — read the Sparse-only metrics.
    use_dense: bool = True,
    seed: int = 42,
):
    """
    Trains a COBRA model.
    """
    if max_train_batches is not None and max_train_batches <= 0:
        raise ValueError("max_train_batches must be positive when provided")

    _run_config = dict(locals())
    # Setup logger
    logger = setup_logger(save_dir_root, name="cobra")
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
            step_metrics={"train/*": "global_step", "eval/*": "epoch"}
        )

    # Create datasets
    train_dataset = dataset(
        root=dataset_folder,
        train_test_split="train",
        max_seq_len=max_seq_len,
        pretrained_rqvae_path=pretrained_rqvae_path,
        # NOTE: do NOT pass the model's encoder (sentence-t5-base) here. The dataset's
        # item embeddings must be produced by the SAME encoder the RQ-VAE was trained on
        # (sentence-t5-xl, via the AmazonCobraDataset.encoder_model_name gin binding),
        # otherwise the embeddings live in the wrong space and the semantic IDs collapse
        # (e.g. 6901 -> 529 unique on Beauty). base/xl share the T5 tokenizer, so the
        # text the model's encoder sees is unchanged.
    )

    valid_dataset = dataset(
        root=dataset_folder,
        train_test_split="valid",
        max_seq_len=max_seq_len,
        pretrained_rqvae_path=pretrained_rqvae_path,
        # NOTE: do NOT pass the model's encoder (sentence-t5-base) here. The dataset's
        # item embeddings must be produced by the SAME encoder the RQ-VAE was trained on
        # (sentence-t5-xl, via the AmazonCobraDataset.encoder_model_name gin binding),
        # otherwise the embeddings live in the wrong space and the semantic IDs collapse
        # (e.g. 6901 -> 529 unique on Beauty). base/xl share the T5 tokenizer, so the
        # text the model's encoder sees is unchanged.
    )

    test_dataset = dataset(
        root=dataset_folder,
        train_test_split="test",
        max_seq_len=max_seq_len,
        pretrained_rqvae_path=pretrained_rqvae_path,
        # NOTE: do NOT pass the model's encoder (sentence-t5-base) here. The dataset's
        # item embeddings must be produced by the SAME encoder the RQ-VAE was trained on
        # (sentence-t5-xl, via the AmazonCobraDataset.encoder_model_name gin binding),
        # otherwise the embeddings live in the wrong space and the semantic IDs collapse
        # (e.g. 6901 -> 529 unique on Beauty). base/xl share the T5 tokenizer, so the
        # text the model's encoder sees is unchanged.
    )
    _run_config["evaluation_protocol"] = train_dataset.evaluation_protocol

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

    logger.info(f"train_dataloader: {len(train_dataloader)}")
    logger.info(f"valid_dataloader: {len(valid_dataloader)}")
    logger.info(f"test_dataloader: {len(test_dataloader)}")

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
        encoder_pooling=encoder_pooling,
        dense_negative_scope=dense_negative_scope,
        use_dense_memory_queue=use_dense_memory_queue,
        use_dense=use_dense,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_batches_per_epoch = len(train_dataloader)
    if max_train_batches is not None:
        train_batches_per_epoch = min(train_batches_per_epoch, max_train_batches)
    total_steps = (
        train_batches_per_epoch * epochs + gradient_accumulate_every - 1
    ) // gradient_accumulate_every
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Device: {device}, Num Parameters: {num_params:,}")

    if accelerator.is_main_process:
        pbar = tqdm(total=total_steps, dynamic_ncols=True)

    # Resume from checkpoint
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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path}")

    # Prepare item data for BeamFusion evaluation
    logger.info("Preparing item data for BeamFusion evaluation...")
    item_sem_ids = torch.tensor(train_dataset.sem_ids_list, dtype=torch.long, device=device)  # (N, C)
    n_items = len(train_dataset.sem_ids_list)
    logger.info(f"Total items: {n_items}")

    # Build sem_id -> items index for coarse-to-fine retrieval
    from collections import defaultdict
    sem_id_to_items = defaultdict(list)
    for item_idx, sem_ids in enumerate(train_dataset.sem_ids_list):
        sem_id_tuple = tuple(sem_ids)
        sem_id_to_items[sem_id_tuple].append(item_idx)
    sem_id_to_items = dict(sem_id_to_items)
    logger.info(f"Built sem_id_to_items index with {len(sem_id_to_items)} unique semantic IDs")

    # Pre-compute item dense vectors (in batches to avoid OOM)
    def compute_item_dense_vecs(model, dataset, batch_size=64):
        """Compute dense vectors for all items using the model's encoder."""
        model.eval()
        all_vecs = []
        n_items = len(dataset.item_texts)

        with torch.no_grad():
            for start_idx in range(0, n_items, batch_size):
                end_idx = min(start_idx + batch_size, n_items)
                item_ids = list(range(start_idx, end_idx))

                # Tokenize item texts
                texts = [dataset.item_texts.get(i, f"item_{i}") for i in item_ids]
                encoded = dataset.tokenizer(
                    texts,
                    padding='max_length',
                    truncation=True,
                    max_length=dataset.max_text_len,
                    return_tensors='pt'
                )
                encoder_input_ids = encoded['input_ids'].to(device)  # (batch, L)

                # Get dense vectors from encoder
                # encoder expects (B, T, L), we have (B, L), so add T=1 dimension
                encoder_input_ids = encoder_input_ids.unsqueeze(1)  # (batch, 1, L)
                vecs = accelerator.unwrap_model(model).encoder(encoder_input_ids)  # (batch, 1, D)
                vecs = vecs.squeeze(1)  # (batch, D)
                vecs = torch.nn.functional.normalize(vecs, p=2, dim=-1)
                all_vecs.append(vecs.cpu())

        # The caller enters evaluation mode before computing the catalog and
        # immediately uses the same model for BeamFusion.  Do not silently
        # re-enable dropout/gradient checkpointing here: that made query-side
        # dense vectors stochastic while catalog vectors were deterministic.
        return torch.cat(all_vecs, dim=0).to(device)  # (N, D)

    @torch.no_grad()
    def evaluate_recommendation_split(dataloader, item_dense_vecs, description):
        """Evaluate one split without using it for any other decision."""
        item_accumulator = TopKAccumulator()
        sid_accumulator = TopKAccumulator()
        sparse_accumulator = TopKAccumulator()
        codebook_correct = [0] * n_codebooks
        codebook_total = 0

        for data in tqdm(
            dataloader, description,
            disable=not accelerator.is_main_process,
        ):
            generated = accelerator.unwrap_model(model).beam_fusion(
                input_ids=data["input_ids"].to(device),
                encoder_input_ids=data["encoder_input_ids"].to(device),
                item_dense_vecs=item_dense_vecs,
                item_sem_ids=item_sem_ids,
                sem_id_to_items=sem_id_to_items,
                n_candidates=10,
                n_beam=eval_n_beam,
                tau=eval_tau,
                psi=eval_psi,
                constrain_to_catalog=constrain_to_catalog,
            )
            target_item = data["target_item_ids"].to(device)
            target_sid = data["target_sem_ids"].to(device)
            item_accumulator.accumulate(
                actual=target_item.unsqueeze(-1),
                top_k=generated.item_ids.unsqueeze(-1),
            )
            sid_accumulator.accumulate(actual=target_sid, top_k=generated.sem_ids)

            sparse_gen = accelerator.unwrap_model(model).generate(
                input_ids=data["input_ids"].to(device),
                encoder_input_ids=data["encoder_input_ids"].to(device),
                n_candidates=10,
            )
            sparse_accumulator.accumulate(
                actual=target_sid, top_k=sparse_gen.sem_ids
            )
            top1 = generated.sem_ids[:, 0, :]
            for codebook in range(n_codebooks):
                codebook_correct[codebook] += (
                    top1[:, codebook] == target_sid[:, codebook]
                ).sum().item()
            codebook_total += target_sid.size(0)

        item_accumulator.sync_across_processes(accelerator)
        sid_accumulator.sync_across_processes(accelerator)
        sparse_accumulator.sync_across_processes(accelerator)
        counts = torch.tensor(
            codebook_correct + [codebook_total],
            device=device,
            dtype=torch.float64,
        )
        counts = accelerator.reduce(counts, reduction="sum")
        total = max(int(counts[-1].item()), 1)
        codebook_accuracy = [float(value) / total for value in counts[:-1]]
        return (
            item_accumulator.reduce(),
            sid_accumulator.reduce(),
            sparse_accumulator.reduce(),
            codebook_accuracy,
        )

    # Training loop
    model.train()
    global_step = -1

    best_valid_recall10 = float("-inf")
    best_epoch = -1
    best_model_state: Optional[Dict[str, torch.Tensor]] = None
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
                        item_ids=data["item_ids"].to(device),
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

            if max_train_batches is not None and step + 1 >= max_train_batches:
                break

        # End of epoch logging
        epoch_acc = epoch_acc_correct / max(epoch_acc_total, 1)
        epoch_recall = epoch_recall_correct / max(epoch_recall_total, 1)
        logger.info(f"Epoch {epoch} - acc: {epoch_acc:.4f}, recall: {epoch_recall:.4f}")

        log_dict = {"epoch": epoch} if wandb_logging else None
        if wandb_logging and accelerator.is_main_process:
            log_dict["train/epoch_acc"] = epoch_acc
            log_dict["train/epoch_recall"] = epoch_recall

        # Evaluation with Recall@K and NDCG@K using BeamFusion
        if do_eval and (epoch + 1) % eval_valid_every_epoch == 0:
            model.eval()
            item_dense_vecs = compute_item_dense_vecs(model, train_dataset, batch_size=64)
            item_metrics, metrics, sparse_metrics, c_accs = (
                evaluate_recommendation_split(
                    valid_dataloader,
                    item_dense_vecs,
                    f"Valid Eval (Epoch {epoch})",
                )
            )
            logger.info(f"Epoch {epoch} - Valid (BeamFusion ITEM-level): {item_metrics}")
            logger.info(
                f"Epoch {epoch} - Valid (BeamFusion sem-id pre-resolution "
                f"diagnostic; not directly comparable): {metrics}"
            )
            logger.info(f"Epoch {epoch} - Valid (Sparse-only sem-id): {sparse_metrics}")
            logger.info(f"  Per-codebook acc: " + ", ".join([f"c{c}={c_accs[c]:.4f}" for c in range(n_codebooks)]))

            if wandb_logging and accelerator.is_main_process:
                for k, v in item_metrics.items():
                    log_dict[f"eval/valid_{k}"] = v

            valid_recall10 = item_metrics.get("Recall@10", 0.0)
            if valid_recall10 > best_valid_recall10:
                best_valid_recall10 = valid_recall10
                best_epoch = epoch
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in accelerator.get_state_dict(model).items()
                }
                if accelerator.is_main_process:
                    best_path = os.path.join(save_dir_root, "best_model.pt")
                    os.makedirs(save_dir_root, exist_ok=True)
                    torch.save(
                        {
                            "epoch": best_epoch,
                            "valid_metrics": item_metrics,
                            "model": best_model_state,
                        },
                        best_path,
                    )
                    logger.info(
                        "New best validation Recall@10 %.4f; saved %s",
                        best_valid_recall10, best_path,
                    )

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

    last_test_metrics: Dict[str, float] = {}
    last_test_sparse_metrics: Dict[str, float] = {}
    if do_eval:
        # Strict one-shot test after validation-only checkpoint selection.
        accelerator.wait_for_everyone()
        if best_model_state is not None:
            accelerator.unwrap_model(model).load_state_dict(
                best_model_state, strict=True
            )
        model.eval()
        best_item_dense_vecs = compute_item_dense_vecs(
            model, train_dataset, batch_size=64
        )
        test_item_metrics, test_sid_metrics, test_sparse_metrics, _ = (
            evaluate_recommendation_split(
                test_dataloader,
                best_item_dense_vecs,
                "Test Eval (one-shot)",
            )
        )
        last_test_metrics = dict(test_item_metrics)
        last_test_sparse_metrics = dict(test_sparse_metrics)
        if accelerator.is_main_process:
            logger.info(
                "Test (one-shot, best epoch %d; BeamFusion ITEM-level): %s",
                best_epoch, test_item_metrics,
            )
            logger.info(
                "Test (BeamFusion sem-id diagnostic; not directly comparable): %s",
                test_sid_metrics,
            )
            logger.info("Test (Sparse-only sem-id): %s", test_sparse_metrics)
            if wandb_logging:
                wandb.log({
                    "epoch": best_epoch,
                    **{f"test/{key}": value for key, value in test_item_metrics.items()},
                    **{f"test_semid/{key}": value for key, value in test_sid_metrics.items()},
                })

    if wandb_logging and accelerator.is_main_process:
        wandb.finish()

    if accelerator.is_main_process and last_test_metrics:
        save_run_results(
            save_dir=save_dir_root,
            model="cobra",
            split=get_run_split(),
            seed=seed,
            metrics=last_test_metrics,
            config=_run_config,
            extra={
                "evaluation_level": "item",
                "best_epoch": best_epoch,
                "best_valid_recall@10": best_valid_recall10,
                "test_policy": "one_shot_after_validation_selection",
                "sid_reference_note": "logged only; not used for selection or headline",
                "sparse_only_metrics": last_test_sparse_metrics,
                "beam_fusion": {
                    "n_beam": eval_n_beam,
                    "tau": eval_tau,
                    "psi": eval_psi,
                    "constrain_to_catalog": constrain_to_catalog,
                },
            },
        )

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parse_config()
    train()
