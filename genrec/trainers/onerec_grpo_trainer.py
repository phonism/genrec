"""
OneRec GRPO Trainer - Stage 3 of OneRec pipeline.

Implements Group Relative Policy Optimization for recommendation.
Key components:
- PrefixTrie: Constrained decoding allowing only valid SID combinations
- Reward functions: exact match and NDCG-based
- GRPO training loop with KL divergence regularization
"""
import os
import re
import gin
import torch
import wandb
import math
from typing import List, Dict, Optional

from genrec.models.onerec import OneRec
from genrec.modules.utils import parse_config, setup_logger, get_run_split
from genrec.modules.metrics import SemanticIdItemMapper
from genrec.modules.prefix_trie import PrefixTrie
from genrec.trainers.trainer_utils import setup_accelerator, setup_wandb, set_seed, save_run_results
from genrec.trainers.lcrec_trainer import (
    lcrec_collate_fn,
    ConstrainedDecodingHelper,
    evaluate as lcrec_evaluate,
    log_metrics as lcrec_log_metrics,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup


def compute_rule_reward(
    generated_sem_ids: List[List[int]],
    target_sem_ids: List[int],
) -> List[float]:
    """
    Exact match reward: 1.0 if generated == target, else 0.0.

    Args:
        generated_sem_ids: List of G generated SID sequences
        target_sem_ids: Ground truth SID sequence

    Returns:
        List of G reward values
    """
    rewards = []
    for gen in generated_sem_ids:
        if gen == target_sem_ids:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def compute_hierarchical_reward(
    generated_sem_ids: List[List[int]],
    target_sem_ids: List[int],
    num_codebooks: int,
) -> List[float]:
    """
    Hierarchical matching reward aligned with OpenOneRec.

    100x scale difference between match levels:
    - All codebooks match: 100.0
    - Partial prefix match (k layers): 10^(k-1)
      e.g. for 3 codebooks: match 1 -> 1.0, match 2 -> 10.0, match 3 -> 100.0
    - No match: 0.0

    Args:
        generated_sem_ids: List of G generated SID sequences
        target_sem_ids: Ground truth SID sequence
        num_codebooks: Number of codebooks

    Returns:
        List of G reward values
    """
    rewards = []
    for gen in generated_sem_ids:
        match_len = 0
        for i in range(min(len(gen), len(target_sem_ids), num_codebooks)):
            if gen[i] == target_sem_ids[i]:
                match_len += 1
            else:
                break
        if match_len == 0:
            rewards.append(0.0)
        else:
            # OpenOneRec style: 1, 10, 100
            rewards.append(10.0 ** (match_len - 1))
    return rewards


def compute_rank_reward(
    generated_sem_ids: List[List[int]],
    target_sem_ids: List[int],
    num_codebooks: int,
) -> List[float]:
    """
    MiniOneRec-style Rule + Rank-aware penalty reward.

    - Rule: 1.0 if exact match, else 0.0
    - Rank penalty: ONLY when at least one match exists, penalize wrong items
      by their generation rank (higher-ranked wrong items get larger penalties).
      When NO match exists, all rewards are 0.0 (skip this group).
    - Combined: R_rule + R_rank
    """
    G = len(generated_sem_ids)

    # Rule reward: binary correctness
    rule_rewards = []
    has_match = False
    for gen in generated_sem_ids:
        if gen == target_sem_ids:
            rule_rewards.append(1.0)
            has_match = True
        else:
            rule_rewards.append(0.0)

    # If no match, return all zeros (group will be skipped naturally)
    if not has_match:
        return [0.0] * G

    # Rank-aware penalty: only when a match exists
    rank_penalties = []
    penalty_sum = 0.0
    for k, gen in enumerate(generated_sem_ids):
        if gen == target_sem_ids:
            rank_penalties.append(0.0)
        else:
            penalty = -1.0 / math.log(k + 2)
            rank_penalties.append(penalty)
            penalty_sum += abs(penalty)

    if penalty_sum > 0:
        rank_penalties = [p / penalty_sum for p in rank_penalties]

    rewards = [rule_rewards[i] + rank_penalties[i] for i in range(G)]
    return rewards


def rl_collate_fn(batch, tokenizer, max_length=512, num_codebooks=3):
    """Collate function for RL dataset (prompt-only, left padding for generation)."""
    prompts = [s['prompt'] for s in batch]

    orig_side = tokenizer.padding_side
    orig_trunc = tokenizer.truncation_side
    tokenizer.padding_side = 'left'
    # Truncate from the left too: an over-length prompt must keep its tail
    # ("### Response:" plus the most recent history) — the default right
    # truncation would cut off the marker the prefix trie keys on.
    tokenizer.truncation_side = 'left'
    encoded = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    tokenizer.padding_side = orig_side
    tokenizer.truncation_side = orig_trunc

    default_sem_ids = [0] * num_codebooks
    target_sem_ids = [s.get('target_sem_ids', default_sem_ids) for s in batch]
    target_items = [s.get('target_item', -1) for s in batch]

    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'target_sem_ids': torch.tensor(target_sem_ids, dtype=torch.long),
        'target_items': torch.tensor(target_items, dtype=torch.long),
        'tasks': [s.get('task', 'sidsft') for s in batch],
    }


@gin.configurable
def train_grpo(
    # Training
    epochs=2, batch_size=4, learning_rate=1e-5, weight_decay=0.01,
    warmup_steps=10, gradient_accumulate_every=4, max_length=512,
    max_grad_norm=0.3,
    # GRPO
    num_generations=16, temperature=1.0, beta=1e-3, clip_eps=0.2,
    reward_type="exact", use_beam_search=False, add_gt=False,
    # Model
    num_codebooks=3, codebook_size=8192,
    sft_checkpoint_path="./out/onerec/amazon2023/{split}/sft/checkpoint_final",
    use_lora=True, lora_r=16, lora_alpha=32, lora_dropout=0.05,
    # Dataset
    rl_dataset=None, dataset_folder="dataset/amazon2023", max_seq_len=20,
    # Eval
    eval_dataset=None,
    do_eval=True, eval_every_epoch=1, eval_batch_size=4, eval_beam_width=10,
    # Save
    save_dir_root="out/onerec/amazon2023/{split}/grpo", save_every_epoch=1,
    # Logging
    wandb_logging=False, wandb_project="onerec_grpo", wandb_run_name=None, wandb_log_interval=10,
    # Accelerator
    split_batches=True, amp=True, mixed_precision_type="bf16",
    # Debug
    max_train_samples=0, max_eval_samples=0, debug_logging=False,
    seed=42,
):
    """Train OneRec with GRPO."""
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

    # Load SFT checkpoint as policy model
    logger.info(f"Loading SFT checkpoint from {sft_checkpoint_path}")
    model = OneRec(pretrained_path=sft_checkpoint_path)
    # Non-reentrant checkpointing: required for the DDP-wrapped training forward
    # (reentrant GC breaks DDP's reducer) and for LoRA's frozen-base inputs.
    model.gradient_checkpointing_enable(use_reentrant=False)
    # Codebook tokens should already be in the checkpoint's tokenizer
    tokenizer = model.tokenizer

    # Reference model strategy depends on LoRA usage and beta
    ref_model = None  # Will be set if not using LoRA and beta > 0
    if beta > 0:
        if use_lora:
            # With LoRA: base model IS the ref model, only LoRA adapters are trained.
            # Disable adapters temporarily to get ref model behavior — no extra GPU memory.
            try:
                from peft import get_peft_model, LoraConfig, TaskType
                model.model = get_peft_model(model.model, LoraConfig(
                    task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
                ))
                if accelerator.is_main_process:
                    model.model.print_trainable_parameters()
                logger.info("Using LoRA — ref model via adapter disable (no separate copy)")
            except ImportError:
                logger.warning("peft not installed, falling back to separate ref model")
                use_lora = False

        if not use_lora:
            # Without LoRA: create a separate frozen ref model
            logger.info("Creating frozen reference model...")
            ref_model = OneRec(pretrained_path=sft_checkpoint_path)
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False
    else:
        logger.info("beta=0: skipping reference model (no KL penalty)")

    # Build prefix trie from all items' sem_ids
    logger.info("Building prefix trie...")
    # Load sem_ids from RL dataset (tokenizer_type, paths etc. configured via gin)
    rl_ds = rl_dataset(
        root=dataset_folder, max_seq_len=max_seq_len,
    )
    _run_config["evaluation_protocol"] = rl_ds.evaluation_protocol
    sem_ids_list = rl_ds.sem_ids_list
    item_mapper = SemanticIdItemMapper.from_sequences(
        sem_ids_list, rl_ds.sequences
    )
    logger.info(
        "Honest SID resolver: most_popular(training-only), unique=%d/%d, "
        "collision_rate=%.2f%%, max_group=%d",
        item_mapper.num_unique_sids, item_mapper.num_items,
        100.0 * item_mapper.collision_rate, item_mapper.max_collision_group,
    )
    prefix_trie = PrefixTrie(sem_ids_list, num_codebooks, codebook_size, tokenizer)
    prefix_fn = prefix_trie.get_prefix_allowed_tokens_fn()

    # Also build ConstrainedDecodingHelper for evaluation
    eval_helper = ConstrainedDecodingHelper(num_codebooks, codebook_size, tokenizer)
    pattern = re.compile(r'<C(\d+)_(\d+)>')

    if max_train_samples > 0:
        rl_ds.samples = rl_ds.samples[:max_train_samples]
        logger.info(f"Limited RL samples to {len(rl_ds.samples)}")

    rl_collate = lambda x: rl_collate_fn(x, tokenizer, max_length, num_codebooks)
    rl_dl = DataLoader(rl_ds, batch_size=batch_size, drop_last=True, shuffle=True,
                       num_workers=4, pin_memory=True, collate_fn=rl_collate)

    # Eval datasets (SFT format for seqrec evaluation)
    eval_dl = None
    test_dl = None

    # Map task names for eval collate
    def eval_collate(batch):
        mapped = []
        for s in batch:
            m = dict(s)
            if m['task'] == 'sidsft':
                m['task'] = 'seqrec'
            elif m['task'] == 'title2sid':
                m['task'] = 'item2index'
            elif m['task'] == 'sid2title':
                m['task'] = 'index2item'
            mapped.append(m)
        return lcrec_collate_fn(mapped, tokenizer, max_length, num_codebooks, is_eval=True)

    if do_eval and eval_dataset is not None:
        eval_ds = eval_dataset(
            root=dataset_folder, train_test_split="valid",
            max_seq_len=max_seq_len,
        )
        test_ds = eval_dataset(
            root=dataset_folder, train_test_split="test",
            max_seq_len=max_seq_len,
        )
        if max_eval_samples > 0:
            eval_ds.samples = eval_ds.samples[:max_eval_samples]
            test_ds.samples = test_ds.samples[:max_eval_samples]

        eval_dl = DataLoader(eval_ds, batch_size=eval_batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, collate_fn=eval_collate)
        test_dl = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, collate_fn=eval_collate)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rl_dl = accelerator.prepare(rl_dl)
    if eval_dl is not None:
        eval_dl = accelerator.prepare(eval_dl)
    if test_dl is not None:
        test_dl = accelerator.prepare(test_dl)

    total_steps = len(rl_dl) * epochs // gradient_accumulate_every
    logger.info(f"Total steps: {total_steps}, Warmup: {warmup_steps}")

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    if ref_model is not None:
        ref_model = accelerator.prepare(ref_model)

    logger.info(f"Device: {device}, Params: {sum(p.numel() for p in model.parameters()):,}, "
                f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Select reward function
    if reward_type == "rank":
        reward_fn = lambda gen, tgt: compute_rank_reward(gen, tgt, num_codebooks)
    elif reward_type == "hierarchical":
        reward_fn = lambda gen, tgt: compute_hierarchical_reward(gen, tgt, num_codebooks)
    elif reward_type == "exact":
        reward_fn = compute_rule_reward
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}. Use 'rank', 'hierarchical', or 'exact'.")

    # Skip index2item during training: 50-token free generation is ~10x slower
    # than the constrained seqrec/item2index paths and not used for early stopping.
    _train_eval_tasks = {'seqrec', 'item2index'}

    def snapshot_trainable_state() -> Dict[str, torch.Tensor]:
        unwrapped = accelerator.unwrap_model(model)
        trainable_names = {
            name for name, parameter in unwrapped.named_parameters()
            if parameter.requires_grad
        }
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in accelerator.get_state_dict(model).items()
            if name in trainable_names
        }

    best_valid_recall10 = float("-inf")
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None

    # Pre-training eval: verify SFT checkpoint baseline
    if do_eval and eval_dl is not None:
        logger.info("=== Pre-training eval (SFT baseline) ===")
        metrics, topk = lcrec_evaluate(
            model, eval_dl, accelerator, tokenizer, eval_helper,
            num_codebooks, eval_beam_width, logger, -1, debug_logging,
            eval_tasks=_train_eval_tasks, item_mapper=item_mapper,
            prefix_trie=prefix_trie,
        )
        if accelerator.is_main_process:
            logger.info(f"--- SFT Baseline (Valid) ---")
            lcrec_log_metrics(metrics, topk, num_codebooks, -1, logger, wandb_logging)
        # The starting SFT policy is a real candidate.  If RL hurts validation,
        # final test must fall back to this baseline rather than selecting an
        # inferior GRPO epoch merely because it was evaluated later.
        best_valid_recall10 = topk.get("Recall@10", 0.0)
        best_state = snapshot_trainable_state()
        model.train()

    # Training loop
    pbar = tqdm(total=total_steps, dynamic_ncols=True) if accelerator.is_main_process else None
    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_stats = defaultdict(float)

        for data in rl_dl:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            target_sem_ids = data['target_sem_ids']  # [B, C]
            B = input_ids.size(0)
            L = input_ids.size(1)

            # Step 1: Generate G completions per prompt. Rollouts run under
            # no_grad, so the unwrapped module is fine here (skips DDP overhead).
            unwrapped_model = accelerator.unwrap_model(model)
            model.eval()
            with torch.no_grad():
                generated_ids, gen_log_probs_old = unwrapped_model.generate_for_grpo(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_generations=num_generations,
                    max_new_tokens=num_codebooks + 1,  # +1 for EOS
                    temperature=temperature,
                    prefix_allowed_tokens_fn=prefix_fn,
                    use_beam_search=use_beam_search,
                )
            # generated_ids: [B*G, L+T], gen_log_probs_old: [B*G, T]

            # GT-injected rows: last candidate of each group.
            gt_rows = [b * num_generations + (num_generations - 1) for b in range(B)]

            # Step 1.5: Inject ground truth into last position of each group (add_gt)
            if add_gt:
                T = generated_ids.size(1) - L
                for b in range(B):
                    target = target_sem_ids[b].tolist()
                    # Ground truth token ids <C0_X><C1_Y><C2_Z><eos> via the
                    # precomputed map — no tokenizer calls in the hot loop.
                    gt_tokens = prefix_trie.sem_ids_to_token_ids(target)
                    gt_tokens.append(tokenizer.eos_token_id)
                    gt_tokens += [tokenizer.pad_token_id] * (T - len(gt_tokens))
                    generated_ids[gt_rows[b], L:] = torch.tensor(
                        gt_tokens[:T], device=device, dtype=generated_ids.dtype)
                # GT injection rewrote those rows, so the recorded old log-probs no
                # longer match the sequences. Recompute behavior log-probs for just
                # the injected rows so the importance ratio scores each sequence
                # (incl. the GT) against itself, not the original sampled candidate.
                with torch.no_grad():
                    gen_log_probs_old[gt_rows] = unwrapped_model.compute_log_probs(
                        generated_ids[gt_rows], prompt_len=L,
                    )

            # Mark GT-injected rows. They are off-policy: the policy may assign
            # them ~0 probability, which makes the k3 KL estimator explode (exp of
            # a large log-ratio). We keep them in the PG term (their large positive
            # advantage is the whole point of add_gt) but exclude them from the KL
            # penalty below.
            is_gt_row = torch.zeros(B * num_generations, device=device)
            if add_gt:
                is_gt_row[gt_rows] = 1.0

            # Step 2: Extract generated sem_ids and compute rewards
            all_rewards = []
            for b in range(B):
                target = target_sem_ids[b].tolist()
                gen_sem_ids = []
                for g in range(num_generations):
                    idx = b * num_generations + g
                    gen_tokens = tokenizer.decode(generated_ids[idx, L:], skip_special_tokens=False)
                    matches = pattern.findall(gen_tokens)
                    if len(matches) >= num_codebooks:
                        sem = [int(matches[c][1]) for c in range(num_codebooks)]
                    else:
                        sem = [0] * num_codebooks
                    gen_sem_ids.append(sem)

                rewards = reward_fn(gen_sem_ids, target)
                all_rewards.extend(rewards)

            rewards_tensor = torch.tensor(all_rewards, device=device, dtype=torch.float32)

            # Step 3: Group-normalize rewards to get advantages (MiniOneRec style)
            rewards_grouped = rewards_tensor.view(B, num_generations)
            group_mean = rewards_grouped.mean(dim=1, keepdim=True)
            group_std = rewards_grouped.std(dim=1, keepdim=True)
            # Soft normalization: no hard mask, let near-zero advantages naturally have small gradients
            advantages = ((rewards_grouped - group_mean) / (group_std + 1e-4)).view(-1)  # [B*G]

            # Track zero-reward groups for diagnostics
            zero_reward_groups = (rewards_grouped.sum(dim=1) == 0).float().mean().item()
            uniform_reward_groups = (group_std.squeeze(1) < 1e-4).float().mean().item()

            # Step 4: Compute policy log probs. This is the gradient-bearing
            # forward, so it MUST go through the prepared model — DDP only arms
            # its gradient-sync hooks inside DDP.forward; calling a method on the
            # unwrapped module would silently train each rank independently.
            model.train()
            with accelerator.accumulate(model):
                # Build attention mask for generated sequences
                gen_attn_mask = (generated_ids != tokenizer.pad_token_id).long()
                policy_log_probs = model(
                    generated_ids.to(device), prompt_len=L,
                    attention_mask=gen_attn_mask.to(device),
                )  # [B*G, T]

                # Step 5: Compute ref model log probs for KL (skip when beta=0)
                if beta > 0:
                    with torch.no_grad():
                        if ref_model is not None:
                            # Separate ref model path
                            ref_log_probs = accelerator.unwrap_model(ref_model).compute_log_probs(
                                generated_ids.to(device), prompt_len=L,
                                attention_mask=gen_attn_mask.to(device),
                            )  # [B*G, T]
                        else:
                            # LoRA path: disable adapters to get base (ref) model behavior
                            unwrapped_model.model.disable_adapter_layers()
                            unwrapped_model.eval()
                            ref_log_probs = unwrapped_model.compute_log_probs(
                                generated_ids.to(device), prompt_len=L,
                                attention_mask=gen_attn_mask.to(device),
                            )  # [B*G, T]
                            unwrapped_model.model.enable_adapter_layers()
                            unwrapped_model.train()

                # Per-token loss and KL (aligned with MiniOneRec/OpenOneRec)
                # Mask for valid generated tokens (non-padding)
                gen_tokens = generated_ids[:, L:].to(device)
                completion_mask = (gen_tokens != tokenizer.pad_token_id).float()  # [B*G, T]

                # Per-token KL (only when beta > 0)
                if beta > 0:
                    per_token_log_ratio = ref_log_probs - policy_log_probs  # [B*G, T]
                    per_token_log_ratio = per_token_log_ratio.clamp(-20, 20)
                    per_token_kl = (per_token_log_ratio.exp() - per_token_log_ratio - 1)  # [B*G, T]
                    # Exclude off-policy GT-injected rows from the KL penalty: the
                    # policy can assign them near-zero prob, blowing up the estimator.
                    per_token_kl = per_token_kl * (1.0 - is_gt_row).unsqueeze(1)
                else:
                    per_token_kl = torch.zeros_like(policy_log_probs)

                # Per-token PPO objective with a real importance ratio against the
                # behavior policy (old log-probs recorded at generation time, line ~484).
                # With one update per rollout the ratio is near 1, but using the true
                # ratio (a) is correct under dropout (gen runs eval, training runs with
                # dropout, so old != new), and (b) stays valid if we ever do multiple
                # inner updates / off-policy reuse — the old `exp(p - p.detach())` was
                # identically 1 and would have silently broken that.
                old_log_probs = gen_log_probs_old.detach().to(device)  # [B*G, T]
                per_token_ratio = torch.exp(policy_log_probs - old_log_probs)  # [B*G, T]
                adv = advantages.detach().unsqueeze(1)  # [B*G, 1]
                unclipped = per_token_ratio * adv
                clipped = torch.clamp(per_token_ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                per_token_pg = -torch.min(unclipped, clipped)  # PPO clipped surrogate
                per_token_loss = per_token_pg + beta * per_token_kl

                # Sequence-mean then batch-mean
                seq_loss = (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)
                loss = seq_loss.mean()

                # For logging: scalar KL, true mean PG loss, and PPO clip fraction
                kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
                policy_loss = ((per_token_pg * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
                with torch.no_grad():
                    clip_frac = ((((per_token_ratio < 1.0 - clip_eps) | (per_token_ratio > 1.0 + clip_eps)).float()
                                 * completion_mask).sum() / completion_mask.sum().clamp(min=1)).item()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    epoch_stats['loss'] += loss.item()
                    epoch_stats['policy_loss'] += policy_loss.item()
                    epoch_stats['kl'] += kl.item()
                    epoch_stats['mean_reward'] += rewards_tensor.mean().item()
                    epoch_stats['steps'] += 1

                    if pbar:
                        pbar.set_description(
                            f'Epoch {epoch} | loss: {loss.item():.4f} | '
                            f'reward: {rewards_tensor.mean().item():.3f} | '
                            f'kl: {kl.item():.4f}'
                        )
                        pbar.update(1)

                    if wandb_logging and accelerator.is_main_process and global_step % wandb_log_interval == 0:
                        wandb.log({
                            "global_step": global_step,
                            "train/loss": loss.item(),
                            "train/policy_loss": policy_loss.item(),
                            "train/kl": kl.item(),
                            "train/clip_frac": clip_frac,
                            "train/mean_reward": rewards_tensor.mean().item(),
                            "train/zero_reward_groups": zero_reward_groups,
                            "train/uniform_reward_groups": uniform_reward_groups,
                            "train/lr": lr_scheduler.get_last_lr()[0],
                        })

            accelerator.wait_for_everyone()

        # Epoch summary
        n_steps = max(epoch_stats['steps'], 1)
        logger.info(
            f"Epoch {epoch} - loss: {epoch_stats['loss']/n_steps:.4f}, "
            f"policy_loss: {epoch_stats['policy_loss']/n_steps:.4f}, "
            f"kl: {epoch_stats['kl']/n_steps:.4f}, "
            f"mean_reward: {epoch_stats['mean_reward']/n_steps:.4f}"
        )

        # Evaluation
        if do_eval and eval_dl is not None and (epoch + 1) % eval_every_epoch == 0:
            metrics, topk = lcrec_evaluate(
                model, eval_dl, accelerator, tokenizer, eval_helper,
                num_codebooks, eval_beam_width, logger, epoch, debug_logging,
                eval_tasks=_train_eval_tasks, item_mapper=item_mapper,
                prefix_trie=prefix_trie,
            )
            current_recall10 = topk.get("Recall@10", 0.0)
            is_best = current_recall10 > best_valid_recall10
            if is_best:
                best_valid_recall10 = current_recall10
                best_epoch = epoch
                best_state = snapshot_trainable_state()

            if accelerator.is_main_process:
                logger.info(f"--- Valid ---")
                lcrec_log_metrics(metrics, topk, num_codebooks, epoch, logger, wandb_logging)

            model.train()

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

    last_test_topk: Dict[str, float] = {}
    last_test_sid_topk: Dict[str, float] = {}
    last_test_task_exact: Dict[str, float] = {}
    if test_dl is not None:
        accelerator.wait_for_everyone()
        if best_state is not None:
            accelerator.unwrap_model(model).load_state_dict(best_state, strict=False)
        test_metrics, test_topk = lcrec_evaluate(
            model, test_dl, accelerator, tokenizer, eval_helper,
            num_codebooks, eval_beam_width, logger, best_epoch, debug_logging,
            eval_tasks=_train_eval_tasks, item_mapper=item_mapper,
            prefix_trie=prefix_trie,
        )
        last_test_topk = dict(test_topk)
        last_test_sid_topk = dict(test_metrics.get('sid_topk', {}))
        last_test_task_exact = {
            task: (test_metrics[task]['exact'] / test_metrics[task]['total'])
            for task in ('seqrec', 'item2index')
            if task in test_metrics and test_metrics[task].get('total', 0) > 0
        }
        if accelerator.is_main_process:
            logger.info("--- Test (one-shot after validation selection) ---")
            lcrec_log_metrics(
                test_metrics, test_topk, num_codebooks, best_epoch, logger,
                wandb_log=False,
            )

    if wandb_logging and accelerator.is_main_process:
        if last_test_topk:
            wandb.log({
                "epoch": best_epoch,
                **{f"test/{task}_exact": value for task, value in last_test_task_exact.items()},
                **{f"test/seqrec_{key}": value for key, value in last_test_topk.items()},
            })
        wandb.finish()

    if accelerator.is_main_process and last_test_topk:
        save_run_results(
            save_dir=save_dir_root,
            model="onerec_grpo",
            split=get_run_split(),
            seed=seed,
            metrics=last_test_topk,
            config=_run_config,
            extra={
                "best_epoch": best_epoch,
                "best_valid_recall@10": best_valid_recall10,
                "evaluation_level": "item",
                "sid_resolution": "most_popular_training_only",
                "sid_reference_metrics": last_test_sid_topk,
                "task_exact": last_test_task_exact,
                "reward_type": reward_type,
                "test_policy": "one_shot_after_validation_selection",
            },
        )

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parse_config()
    train_grpo()
