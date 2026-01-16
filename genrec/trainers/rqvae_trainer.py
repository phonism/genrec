"""
rqvae trainer
"""
import gin
import os
import numpy as np
import time
import wandb

from accelerate import Accelerator
from genrec.data.p5_amazon import P5AmazonReviewsItemDataset
from genrec.data.utils import cycle
from genrec.models.rqvae import RqVae, QuantizeForwardMode
from genrec.modules.utils import parse_config
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import DistributedSampler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


@torch.no_grad()
def compute_collision_rate(model, dataloader, device):
    """
    Compute collision rate across entire dataset.
    collision_rate = (total_items - unique_semantic_ids) / total_items
    """
    model.eval()
    indices_set = set()
    num_sample = 0

    for batch in dataloader:
        data = batch.to(device)
        output = model.get_semantic_ids(data, gumbel_t=0.001)
        indices = output.sem_ids.cpu().numpy()  # [batch, n_layers]

        for idx in indices:
            code = "-".join([str(int(i)) for i in idx])
            indices_set.add(code)
        num_sample += len(data)

    collision_rate = (num_sample - len(indices_set)) / num_sample
    return collision_rate, num_sample, len(indices_set)


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
    encoder_model_name="./models_hub/sentence-t5-xl"
):
    """
    train rqvae
    """
    # Validate training mode
    if epochs is None and iterations is None:
        raise ValueError("Must specify either 'epochs' or 'iterations'")
    if epochs is not None and iterations is not None:
        raise ValueError("Cannot specify both 'epochs' and 'iterations'")

    use_epochs = epochs is not None

    if wandb_logging:
        params = locals()

    # setup accelerator
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    device = accelerator.device

    train_dataset = dataset(root=dataset_folder, train_test_split="train", encoder_model_name=encoder_model_name)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        collate_fn=lambda batch: torch.tensor(batch, dtype=torch.float32),
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=10,
        persistent_workers=True)

    if do_eval:
        eval_dataset = dataset(root=dataset_folder, train_test_split="eval", encoder_model_name=encoder_model_name)
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
        commitment_weight=commitment_weight
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Setup lr scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=params
        )

    start_epoch = 0
    start_step = 0
    if pretrained_rqvae_path is not None:
        model.load_pretrained(pretrained_rqvae_path)
        state = torch.load(pretrained_rqvae_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
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
                        want = 20000
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

                    # Compute collision rate on training data
                    collision_rate, total_items, unique_ids = compute_collision_rate(
                        model, train_dataloader, device
                    )

                    if accelerator.is_main_process:
                        tqdm.write(f"Epoch {epoch+1} Eval - loss: {eval_losses[0]:.4f}, rec: {eval_losses[1]:.4f}, vq: {eval_losses[2]:.4f}, collision: {collision_rate:.4f} ({unique_ids}/{total_items})")
                        if wandb_logging:
                            wandb.log({
                                "eval_total_loss": eval_losses[0],
                                "eval_reconstruction_loss": eval_losses[1],
                                "eval_rqvae_loss": eval_losses[2],
                                "collision_rate": collision_rate,
                                "unique_semantic_ids": unique_ids,
                            })

                if accelerator.is_main_process:
                    if (epoch + 1) % save_model_every == 0 or epoch + 1 == epochs:
                        state = {
                            "epoch": epoch,
                            "model": model.state_dict(),
                            "model_config": model.config,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict()
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
                    want = 20000
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
                            eval_model_output = model(data, gumbel_t=t)
                        eval_losses[0].append(eval_model_output.loss.cpu().item())
                        eval_losses[1].append(eval_model_output.reconstruction_loss.cpu().item())
                        eval_losses[2].append(eval_model_output.rqvae_loss.cpu().item())

                    eval_losses = np.array(eval_losses).mean(axis=-1)
                    id_diversity_log["eval_total_loss"] = eval_losses[0]
                    id_diversity_log["eval_reconstruction_loss"] = eval_losses[1]
                    id_diversity_log["eval_rqvae_loss"] = eval_losses[2]

                    # Compute collision rate on training data
                    collision_rate, total_items, unique_ids = compute_collision_rate(
                        model, train_dataloader, device
                    )
                    id_diversity_log["collision_rate"] = collision_rate
                    id_diversity_log["unique_semantic_ids"] = unique_ids
                    if accelerator.is_main_process:
                        tqdm.write(f"Iter {iter+1} - collision_rate: {collision_rate:.4f} ({unique_ids}/{total_items})")

                if accelerator.is_main_process:
                    if (iter + 1) % save_model_every == 0 or iter + 1 == total_steps:
                        state = {
                            "iter": iter,
                            "model": model.state_dict(),
                            "model_config": model.config,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict()
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


if __name__ == "__main__":
    parse_config()
    train()
