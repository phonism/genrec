"""
rqvae trainer
"""
import gin
import os
import numpy as np
import time
import torch
import wandb

from genrec.data.p5_amazon import P5AmazonReviewsItemDataset
from genrec.data.utils import cycle
from genrec.models.rqvae import RqVae, QuantizeForwardMode
from genrec.modules.utils import parse_config, setup_logger, get_run_split
from genrec.trainers.trainer_utils import setup_accelerator, setup_wandb, set_seed, save_run_results
import torch.distributed as dist
from torch.optim import Adagrad, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import DistributedSampler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


def summarize_semantic_ids(semantic_ids, codebook_size):
    """Return full-tuple collision and per-layer codebook health metrics."""
    semantic_ids = torch.as_tensor(semantic_ids, dtype=torch.long).cpu()
    if semantic_ids.ndim != 2:
        raise ValueError(
            f"semantic_ids must have shape [items, layers], got "
            f"{tuple(semantic_ids.shape)}"
        )

    total_items = int(semantic_ids.shape[0])
    unique_ids = int(torch.unique(semantic_ids, dim=0).shape[0])
    collision_rate = (
        (total_items - unique_ids) / total_items if total_items else 0.0
    )
    layer_usage = []
    layer_max_frequency = []
    for layer_ids in semantic_ids.unbind(dim=1):
        counts = torch.bincount(layer_ids, minlength=codebook_size)
        # Divide integer counts in Python so an exact paper threshold such as
        # 80% is not perturbed by float32 reduction noise.
        layer_usage.append(int((counts > 0).sum().item()) / codebook_size)
        layer_max_frequency.append(
            float(counts.max().item() / total_items) if total_items else 0.0
        )

    return {
        "collision_rate": collision_rate,
        "total_items": total_items,
        "unique_semantic_ids": unique_ids,
        "layer_usage": layer_usage,
        "layer_max_frequency": layer_max_frequency,
        "min_codebook_usage": min(layer_usage, default=0.0),
    }


@torch.no_grad()
def compute_codebook_health(
    model, dataloader, device, codebook_size, accelerator=None
):
    """Encode the catalog once and summarize SID and layer-level health."""
    raw_model = accelerator.unwrap_model(model) if accelerator else model
    raw_model.eval()
    all_indices = []

    for batch in dataloader:
        data = batch.to(device)
        output = raw_model.get_semantic_ids(data, gumbel_t=0.001)
        semantic_ids = output.sem_ids
        if accelerator is not None:
            semantic_ids = accelerator.gather_for_metrics(semantic_ids)
        all_indices.append(semantic_ids.cpu())  # [batch, n_layers]

    semantic_ids = torch.cat(all_indices, dim=0)
    return summarize_semantic_ids(semantic_ids, codebook_size)


def snapshot_eval_metrics(eval_losses, health):
    """Final-eval metrics for results.json (shared by both training modes)."""
    return {
        "eval_total_loss": float(eval_losses[0]),
        "eval_reconstruction_loss": float(eval_losses[1]),
        "eval_rqvae_loss": float(eval_losses[2]),
        "collision_rate": float(health["collision_rate"]),
        "unique_semantic_ids": int(health["unique_semantic_ids"]),
        "total_items": int(health["total_items"]),
        "layer_usage": list(health["layer_usage"]),
        "layer_max_frequency": list(health["layer_max_frequency"]),
        "min_codebook_usage": float(health["min_codebook_usage"]),
    }


def health_selection_key(health, min_codebook_usage):
    """Prefer paper-healthy checkpoints, then collision and usage quality."""
    usage = float(health["min_codebook_usage"])
    collision = float(health["collision_rate"])
    if usage >= min_codebook_usage:
        return (1, -collision, usage)
    return (0, usage, -collision)


@gin.configurable
def train(
    # Training mode: use epochs or iterations (mutually exclusive)
    epochs=None,
    iterations=None,
    warmup_epochs=0,
    batch_size=64,
    learning_rate=0.0001,
    weight_decay=0.01,
    dataset_folder="dataset/amazon",
    dataset=P5AmazonReviewsItemDataset,
    pretrained_rqvae_path=None,
    save_dir_root="out/rqvae/amazon/",
    use_kmeans_init=True,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    wandb_project="rqvae_training",
    wandb_run_name=None,  # Run name, auto-generated if None
    wandb_log_interval=100,
    do_eval=True,
    mixed_precision_type="fp16",
    save_model_every=1000000,
    eval_every=50000,
    commitment_weight=0.25,
    vae_n_cat_feats=18,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_codebook_mode=QuantizeForwardMode.GUMBEL_SOFTMAX,
    vae_codebook_last_layer_mode=QuantizeForwardMode.SINKHORN,
    vae_sim_vq=False,
    vae_n_layers=3,
    mlp_activation="silu",
    mlp_bias=False,
    mlp_weight_init="default",
    decoder_normalize=True,
    loss_reduction="sum",
    encoder_model_name="./models_hub/sentence-t5-xl",
    optimizer_name="adamw",
    adagrad_initial_accumulator_value=0.0,
    adagrad_eps=1e-10,
    lr_schedule="linear",
    train_item_split="train",
    eval_item_split="eval",
    kmeans_init_samples=20000,
    min_codebook_usage=0.8,
    seed=42,
):
    """
    train rqvae
    """
    _run_config = dict(locals())
    # Validate training mode
    if epochs is None and iterations is None:
        raise ValueError("Must specify either 'epochs' or 'iterations'")
    if epochs is not None and iterations is not None:
        raise ValueError("Cannot specify both 'epochs' and 'iterations'")
    if not 0.0 <= min_codebook_usage <= 1.0:
        raise ValueError("min_codebook_usage must be in [0, 1]")

    use_epochs = epochs is not None

    set_seed(seed)

    # setup accelerator
    accelerator = setup_accelerator(
        split_batches=split_batches,
        amp=amp,
        mixed_precision_type=mixed_precision_type,
    )

    device = accelerator.device

    train_dataset = dataset(
        root=dataset_folder,
        train_test_split=train_item_split,
        encoder_model_name=encoder_model_name,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        collate_fn=lambda batch: torch.tensor(batch, dtype=torch.float32),
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=10,
        persistent_workers=True)

    if do_eval:
        eval_dataset = dataset(
            root=dataset_folder,
            train_test_split=eval_item_split,
            encoder_model_name=encoder_model_name,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size, collate_fn=lambda batch: torch.tensor(batch, dtype=torch.float32),
            num_workers=16,
            pin_memory=True,
            prefetch_factor=10,
            persistent_workers=True)

    # Calculate total steps
    steps_per_epoch = len(train_dataloader)
    if use_epochs:
        total_steps = epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch
    else:
        total_steps = iterations
        warmup_steps = 0

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    train_dataloader = accelerator.prepare(train_dataloader)

    model = RqVae(
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=vae_hidden_dims,
        codebook_size=vae_codebook_size,
        codebook_kmeans_init=use_kmeans_init and pretrained_rqvae_path is None,
        codebook_normalize=vae_codebook_normalize,
        codebook_sim_vq=vae_sim_vq,
        codebook_mode=vae_codebook_mode,
        codebook_last_layer_mode=vae_codebook_last_layer_mode,
        n_layers=vae_n_layers,
        n_cat_features=vae_n_cat_feats,
        commitment_weight=commitment_weight,
        mlp_activation=mlp_activation,
        mlp_bias=mlp_bias,
        mlp_weight_init=mlp_weight_init,
        decoder_normalize=decoder_normalize,
        loss_reduction=loss_reduction,
    )

    if optimizer_name == "adamw":
        optimizer = AdamW(
            params=model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adagrad":
        optimizer = Adagrad(
            params=model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            initial_accumulator_value=adagrad_initial_accumulator_value,
            eps=adagrad_eps,
        )
    else:
        raise ValueError(f"unsupported optimizer_name={optimizer_name!r}")

    if lr_schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    elif lr_schedule == "constant":
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    else:
        raise ValueError(f"unsupported lr_schedule={lr_schedule!r}")

    if wandb_logging and accelerator.is_main_process:
        setup_wandb(
            project=wandb_project,
            run_name=wandb_run_name,
            config=locals(),
        )

    start_epoch = 0
    start_step = 0
    best_health_key = None
    if pretrained_rqvae_path is not None:
        model.load_pretrained(pretrained_rqvae_path)
        state = torch.load(pretrained_rqvae_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        if state.get("best_health_key") is not None:
            best_health_key = tuple(state["best_health_key"])
        if "epoch" in state:
            start_epoch = state["epoch"] + 1
            start_step = start_epoch * steps_per_epoch
        elif "iter" in state:
            start_step = state["iter"] + 1
            start_epoch = start_step // steps_per_epoch

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # Training loop
    global_step = start_step
    losses = [[], [], []]
    print_loss, print_rec_loss, print_vae_loss = 0, 0, 0
    final_eval_metrics: dict = {}

    if use_epochs:
        # Epoch-based training with single progress bar
        with tqdm(
            total=total_steps,
            initial=start_step,
            desc="Training",
            disable=not accelerator.is_main_process
        ) as pbar:
            for epoch in range(start_epoch, epochs):
                model.train()

                for batch_idx, data in enumerate(train_dataloader):
                    t = 0.2

                    # KMeans init on first step
                    if global_step == 0 and use_kmeans_init and batch_idx == 0:
                        buf = [data]
                        seen = data.size(0)
                        want = max(int(kmeans_init_samples), data.size(0))
                        if seen < want:
                            for extra_data in train_dataloader:
                                buf.append(extra_data)
                                seen += extra_data.size(0)
                                if seen >= want:
                                    break
                        big_batch = torch.cat(buf, dim=0).to(device)
                        model(big_batch, t)

                    optimizer.zero_grad()
                    data = data.to(device)

                    with accelerator.autocast():
                        model_output = model(data, gumbel_t=t)
                        loss = model_output.loss

                    accelerator.backward(loss)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    # Track losses
                    losses[0].append(loss.cpu().item())
                    losses[1].append(model_output.reconstruction_loss.cpu().item())
                    losses[2].append(model_output.rqvae_loss.cpu().item())
                    losses[0] = losses[0][-1000:]
                    losses[1] = losses[1][-1000:]
                    losses[2] = losses[2][-1000:]

                    if global_step % 100 == 0:
                        print_loss = np.mean(losses[0])
                        print_rec_loss = np.mean(losses[1])
                        print_vae_loss = np.mean(losses[2])

                    pbar.set_postfix({
                        'epoch': f'{epoch+1}/{epochs}',
                        'loss': f'{print_loss:.4f}',
                        'rec': f'{print_rec_loss:.4f}',
                        'vq': f'{print_vae_loss:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })

                    # Wandb logging
                    if accelerator.is_main_process and wandb_logging and (global_step + 1) % wandb_log_interval == 0:
                        emb_norms_avg = model_output.embs_norm.mean(axis=0)
                        emb_norms_avg_log = {
                            f"emb_avg_norm_{i}": emb_norms_avg[i].cpu().item() for i in range(vae_n_layers)
                        }
                        wandb.log({
                            "epoch": epoch,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "total_loss": loss.cpu().item(),
                            "reconstruction_loss": model_output.reconstruction_loss.cpu().item(),
                            "rqvae_loss": model_output.rqvae_loss.cpu().item(),
                            "temperature": t,
                            "p_unique_ids": model_output.p_unique_ids.cpu().item(),
                            **emb_norms_avg_log,
                        })

                    global_step += 1
                    pbar.update(1)

                # End of epoch: eval and save
                if do_eval and ((epoch + 1) % eval_every == 0 or epoch + 1 == epochs):
                    model.eval()
                    eval_losses = [[], [], []]
                    for batch in eval_dataloader:
                        data = batch.to(device)
                        with torch.no_grad():
                            eval_model_output = model(data, gumbel_t=0.2)
                        eval_losses[0].append(eval_model_output.loss.cpu().item())
                        eval_losses[1].append(eval_model_output.reconstruction_loss.cpu().item())
                        eval_losses[2].append(eval_model_output.rqvae_loss.cpu().item())

                    eval_losses = np.array(eval_losses).mean(axis=-1)

                    # Compute full-SID collisions and per-layer codebook usage.
                    health = compute_codebook_health(
                        model, train_dataloader, device, vae_codebook_size,
                        accelerator=accelerator,
                    )

                    if accelerator.is_main_process:
                        tqdm.write(
                            f"Epoch {epoch+1} Eval - loss: {eval_losses[0]:.4f}, "
                            f"rec: {eval_losses[1]:.4f}, vq: {eval_losses[2]:.4f}, "
                            f"collision: {health['collision_rate']:.4f} "
                            f"({health['unique_semantic_ids']}/{health['total_items']}), "
                            f"layer_usage: "
                            f"{[round(v, 4) for v in health['layer_usage']]}"
                        )
                        if wandb_logging:
                            wandb.log({
                                "eval_total_loss": eval_losses[0],
                                "eval_reconstruction_loss": eval_losses[1],
                                "eval_rqvae_loss": eval_losses[2],
                                **health,
                            })
                        final_eval_metrics = snapshot_eval_metrics(eval_losses, health)

                        candidate_key = health_selection_key(
                            health, min_codebook_usage)
                        if best_health_key is None or candidate_key > best_health_key:
                            best_health_key = candidate_key
                            is_healthy = (
                                health["min_codebook_usage"] >= min_codebook_usage
                            )
                            state = {
                                "epoch": epoch,
                                "model": accelerator.unwrap_model(model).state_dict(),
                                "model_config": accelerator.unwrap_model(model).config,
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "best_health_key": best_health_key,
                                "codebook_health": health,
                            }
                            os.makedirs(save_dir_root, exist_ok=True)
                            torch.save(
                                state,
                                os.path.join(
                                    save_dir_root,
                                    (
                                        "checkpoint_best_health.pt"
                                        if is_healthy
                                        else "checkpoint_best_candidate.pt"
                                    ),
                                ),
                            )
                            tqdm.write(
                                (
                                    "Saved new best healthy tokenizer: "
                                    if is_healthy
                                    else "Saved new best candidate (below health gate): "
                                )
                                + f"min_usage={health['min_codebook_usage']:.2%}, "
                                f"collision={health['collision_rate']:.4%}"
                            )

                if accelerator.is_main_process:
                    if (epoch + 1) % save_model_every == 0 or epoch + 1 == epochs:
                        state = {
                            "epoch": epoch,
                            "model": accelerator.unwrap_model(model).state_dict(),
                            "model_config": accelerator.unwrap_model(model).config,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "best_health_key": best_health_key,
                        }
                        if not os.path.exists(save_dir_root):
                            os.makedirs(save_dir_root)
                        torch.save(state, save_dir_root + f"/checkpoint_epoch_{epoch}.pt")

    else:
        # Iteration-based training (original behavior)
        train_dataloader_iter = cycle(train_dataloader)

        with tqdm(
            initial=start_step,
            total=total_steps,
            disable=not accelerator.is_main_process
        ) as pbar:
            for iter in range(start_step, total_steps):
                model.train()
                t = 0.2

                if iter == 0 and use_kmeans_init:
                    buf = []
                    seen = 0
                    want = max(int(kmeans_init_samples), batch_size)
                    while seen < want:
                        data = next(train_dataloader_iter).to(device)
                        buf.append(data)
                        seen += data.size(0)
                    big_batch = torch.cat(buf, dim=0)
                    model(big_batch, t)

                optimizer.zero_grad()
                data = next(train_dataloader_iter).to(device)

                with accelerator.autocast():
                    model_output = model(data, gumbel_t=t)
                    loss = model_output.loss

                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                losses[0].append(loss.cpu().item())
                losses[1].append(model_output.reconstruction_loss.cpu().item())
                losses[2].append(model_output.rqvae_loss.cpu().item())
                losses[0] = losses[0][-1000:]
                losses[1] = losses[1][-1000:]
                losses[2] = losses[2][-1000:]
                if iter % 100 == 0:
                    print_loss = np.mean(losses[0])
                    print_rec_loss = np.mean(losses[1])
                    print_vae_loss = np.mean(losses[2])

                pbar.set_description(f'loss: {print_loss:.4f}, rl: {print_rec_loss:.4f}, vl: {print_vae_loss:.4f}')

                accelerator.wait_for_everyone()

                id_diversity_log = {}
                if accelerator.is_main_process and wandb_logging:
                    emb_norms_avg = model_output.embs_norm.mean(axis=0)
                    emb_norms_avg_log = {
                        f"emb_avg_norm_{i}": emb_norms_avg[i].cpu().item() for i in range(vae_n_layers)
                    }
                    train_log = {
                        "learning_rate": scheduler.get_last_lr()[0],
                        "total_loss": loss.cpu().item(),
                        "reconstruction_loss": model_output.reconstruction_loss.cpu().item(),
                        "rqvae_loss": model_output.rqvae_loss.cpu().item(),
                        "temperature": t,
                        "p_unique_ids": model_output.p_unique_ids.cpu().item(),
                        **emb_norms_avg_log,
                    }

                if do_eval and ((iter + 1) % eval_every == 0 or iter + 1 == total_steps):
                    model.eval()
                    eval_losses = [[], [], []]
                    for batch in eval_dataloader:
                        data = batch.to(device)
                        with torch.no_grad():
                            eval_model_output = model(data, gumbel_t=0.2)
                        eval_losses[0].append(eval_model_output.loss.cpu().item())
                        eval_losses[1].append(eval_model_output.reconstruction_loss.cpu().item())
                        eval_losses[2].append(eval_model_output.rqvae_loss.cpu().item())

                    eval_losses = np.array(eval_losses).mean(axis=-1)
                    id_diversity_log["eval_total_loss"] = eval_losses[0]
                    id_diversity_log["eval_reconstruction_loss"] = eval_losses[1]
                    id_diversity_log["eval_rqvae_loss"] = eval_losses[2]

                    health = compute_codebook_health(
                        model, train_dataloader, device, vae_codebook_size,
                        accelerator=accelerator,
                    )
                    id_diversity_log.update(health)
                    if accelerator.is_main_process:
                        tqdm.write(
                            f"Iter {iter+1} - collision_rate: "
                            f"{health['collision_rate']:.4f} "
                            f"({health['unique_semantic_ids']}/"
                            f"{health['total_items']}), layer_usage: "
                            f"{[round(v, 4) for v in health['layer_usage']]}"
                        )
                        final_eval_metrics = snapshot_eval_metrics(eval_losses, health)

                        candidate_key = health_selection_key(
                            health, min_codebook_usage)
                        if best_health_key is None or candidate_key > best_health_key:
                            best_health_key = candidate_key
                            is_healthy = (
                                health["min_codebook_usage"] >= min_codebook_usage
                            )
                            state = {
                                "iter": iter,
                                "model": accelerator.unwrap_model(model).state_dict(),
                                "model_config": accelerator.unwrap_model(model).config,
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "best_health_key": best_health_key,
                                "codebook_health": health,
                            }
                            os.makedirs(save_dir_root, exist_ok=True)
                            torch.save(
                                state,
                                os.path.join(
                                    save_dir_root,
                                    (
                                        "checkpoint_best_health.pt"
                                        if is_healthy
                                        else "checkpoint_best_candidate.pt"
                                    ),
                                ),
                            )
                            tqdm.write(
                                (
                                    "Saved new best healthy tokenizer: "
                                    if is_healthy
                                    else "Saved new best candidate (below health gate): "
                                )
                                + f"min_usage={health['min_codebook_usage']:.2%}, "
                                f"collision={health['collision_rate']:.4%}"
                            )

                if accelerator.is_main_process:
                    if (iter + 1) % save_model_every == 0 or iter + 1 == total_steps:
                        state = {
                            "iter": iter,
                            "model": accelerator.unwrap_model(model).state_dict(),
                            "model_config": accelerator.unwrap_model(model).config,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "best_health_key": best_health_key,
                        }

                        if not os.path.exists(save_dir_root):
                            os.makedirs(save_dir_root)

                        torch.save(state, save_dir_root + f"/checkpoint_{iter}.pt")

                    if wandb_logging and (iter + 1) % wandb_log_interval == 0:
                        wandb.log({
                            **train_log,
                            **id_diversity_log
                        })

                pbar.update(1)

    if wandb_logging:
        wandb.finish()

    if accelerator.is_main_process:
        save_run_results(
            save_dir=save_dir_root,
            model="rqvae",
            split=get_run_split(),
            seed=seed,
            metrics=final_eval_metrics,
            config=_run_config,
        )


if __name__ == "__main__":
    parse_config()
    train()
