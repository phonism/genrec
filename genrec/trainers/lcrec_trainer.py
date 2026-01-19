"""
LCRec Trainer - Adapting LLMs by Integrating Collaborative Semantics for Recommendation
"""
import os
import re
import gin
import torch
import wandb
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Callable, Tuple

from accelerate import Accelerator
from genrec.models.lcrec import LCRec
from genrec.modules.utils import parse_config
from genrec.modules.metrics import TopKAccumulator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup


def setup_logger(save_dir: str, name: str = "lcrec") -> logging.Logger:
    """Setup logger to write to both file and console."""
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(save_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def lcrec_collate_fn(batch, tokenizer, max_length=512, num_codebooks=5, is_eval=False):
    """Collate function for LCRec dataset."""
    prompts = [s['prompt'] for s in batch]
    responses = [s['response'] for s in batch]
    tasks = [s['task'] for s in batch]

    if is_eval:
        # Left padding for decoder-only models during generation
        orig_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'
        encoded = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        tokenizer.padding_side = orig_side
        labels = None
    else:
        full_texts = [p + r + tokenizer.eos_token for p, r in zip(prompts, responses)]
        encoded = tokenizer(full_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        labels = encoded['input_ids'].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        for i, prompt in enumerate(prompts):
            labels[i, :len(tokenizer(prompt, add_special_tokens=False).input_ids)] = -100

    # Extract target info
    default_sem_ids = [0] * num_codebooks
    target_sem_ids, target_items = [], []
    for s in batch:
        if s['task'] in ['seqrec', 'item2index']:
            target_sem_ids.append(s.get('target_sem_ids', default_sem_ids))
            target_items.append(s.get('target_item', -1))
        else:
            target_sem_ids.append(default_sem_ids)
            target_items.append(-1)

    result = {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'tasks': tasks,
        'target_sem_ids': torch.tensor(target_sem_ids, dtype=torch.long),
        'target_items': torch.tensor(target_items, dtype=torch.long),
    }
    if labels is not None:
        result['labels'] = labels
    return result


@dataclass
class ConstrainedDecodingHelper:
    """Helper for constrained decoding with codebook tokens."""
    num_codebooks: int
    codebook_size: int
    tokenizer: Any
    allowed_tokens: Dict[int, Set[int]] = field(default_factory=dict)
    all_codebook_tokens: Set[int] = field(default_factory=set)
    response_marker_ids: List[int] = field(default_factory=list)
    pattern: re.Pattern = field(default=None)

    def __post_init__(self):
        for c in range(self.num_codebooks):
            self.allowed_tokens[c] = set()
            for code in range(self.codebook_size):
                ids = self.tokenizer(f"<C{c}_{code}>", add_special_tokens=False).input_ids
                if len(ids) == 1:
                    self.allowed_tokens[c].add(ids[0])
                    self.all_codebook_tokens.add(ids[0])
        self.allowed_tokens[self.num_codebooks] = {self.tokenizer.eos_token_id}
        self.response_marker_ids = self.tokenizer("### Response:", add_special_tokens=False).input_ids
        self.pattern = re.compile(r'<C(\d+)_(\d+)>')

    def get_prefix_allowed_tokens_fn(self) -> Callable:
        def fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
            input_list = input_ids.tolist()
            marker_len = len(self.response_marker_ids)
            response_start = -1
            for i in range(len(input_list) - marker_len + 1):
                if input_list[i:i + marker_len] == self.response_marker_ids:
                    response_start = i + marker_len
                    break
            if response_start == -1:
                num_gen = sum(1 for t in reversed(input_list) if t in self.all_codebook_tokens)
            else:
                num_gen = len(input_list) - response_start
            return list(self.allowed_tokens[min(num_gen, self.num_codebooks)])
        return fn

    def extract_sem_ids(self, text: str) -> Optional[List[int]]:
        matches = self.pattern.findall(text)
        return [int(matches[i][1]) for i in range(self.num_codebooks)] if len(matches) >= self.num_codebooks else None


def evaluate(model, dataloader, accelerator, tokenizer, helper, num_codebooks, beam_width=10, logger=None, epoch=0, debug=False):
    """Run evaluation on dataloader."""
    model.eval()
    device = accelerator.device
    metrics = {k: {'correct': [0]*num_codebooks, 'total': 0, 'exact': 0} for k in ['seqrec', 'item2index']}
    metrics['index2item'] = {'total': 0, 'exact': 0}
    topk_acc = TopKAccumulator(ks=[1, 5, 10])
    prefix_fn = helper.get_prefix_allowed_tokens_fn()
    debug_count = 0

    def generate(input_ids, attn_mask, max_new, use_beam=False, constrained=True):
        kwargs = dict(
            input_ids=input_ids, attention_mask=attn_mask, max_new_tokens=max_new,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, early_stopping=True, use_cache=True,
        )
        if use_beam:
            kwargs.update(num_beams=beam_width, num_return_sequences=beam_width)
        if constrained:
            kwargs['prefix_allowed_tokens_fn'] = prefix_fn
        return accelerator.unwrap_model(model).model.generate(**kwargs)

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_main_process):
            tasks = data['tasks']

            # SeqRec evaluation
            mask = torch.tensor([t == 'seqrec' for t in tasks])
            if mask.any():
                inp, attn, tgt = data["input_ids"][mask].to(device), data["attention_mask"][mask].to(device), data["target_sem_ids"][mask]
                gen = generate(inp, attn, num_codebooks + 1, use_beam=True)
                inp_len = inp.size(1)

                for i in range(inp.size(0)):
                    target = tgt[i].tolist()
                    preds = []
                    for k in range(beam_width):
                        idx = i * beam_width + k
                        if idx < gen.size(0):
                            sem = helper.extract_sem_ids(tokenizer.decode(gen[idx, inp_len:], skip_special_tokens=False))
                            if sem:
                                preds.append(sem)
                    while len(preds) < beam_width:
                        preds.append([0] * num_codebooks)

                    if debug and debug_count < 3 and accelerator.is_main_process and logger:
                        logger.debug(f"[Epoch {epoch}] Sample {debug_count}: Target={target}, Pred={preds[0]}")
                        debug_count += 1

                    pred = preds[0]
                    exact = all(pred[c] == target[c] for c in range(num_codebooks))
                    for c in range(num_codebooks):
                        if pred[c] == target[c]:
                            metrics['seqrec']['correct'][c] += 1
                    metrics['seqrec']['total'] += 1
                    if exact:
                        metrics['seqrec']['exact'] += 1
                    topk_acc.accumulate(torch.tensor([target], device=device), torch.tensor([preds], device=device))

            # item2index evaluation
            mask = torch.tensor([t == 'item2index' for t in tasks])
            if mask.any():
                inp, attn, tgt = data["input_ids"][mask].to(device), data["attention_mask"][mask].to(device), data["target_sem_ids"][mask]
                gen = generate(inp, attn, num_codebooks + 1)
                inp_len = inp.size(1)

                for i in range(inp.size(0)):
                    sem = helper.extract_sem_ids(tokenizer.decode(gen[i, inp_len:], skip_special_tokens=False))
                    if sem:
                        target = tgt[i].tolist()
                        exact = all(sem[c] == target[c] for c in range(num_codebooks))
                        for c in range(num_codebooks):
                            if sem[c] == target[c]:
                                metrics['item2index']['correct'][c] += 1
                        metrics['item2index']['total'] += 1
                        if exact:
                            metrics['item2index']['exact'] += 1

            # index2item evaluation
            mask = torch.tensor([t == 'index2item' for t in tasks])
            if mask.any():
                inp, attn = data["input_ids"][mask].to(device), data["attention_mask"][mask].to(device)
                labels = data.get("labels")
                labels = labels[mask] if labels is not None else None
                gen = generate(inp, attn, 50, constrained=False)

                for i in range(inp.size(0)):
                    tgt_text = tokenizer.decode(labels[i][labels[i] != -100], skip_special_tokens=True).strip().lower() if labels is not None else ""
                    gen_text = tokenizer.decode(gen[i, inp.size(1):], skip_special_tokens=True).strip().lower()
                    metrics['index2item']['total'] += 1
                    if tgt_text and gen_text and tgt_text in gen_text:
                        metrics['index2item']['exact'] += 1

    topk_metrics = topk_acc.reduce()

    # Gather from all GPUs
    def gather(v):
        t = torch.tensor([v], device=device, dtype=torch.float32)
        return int(accelerator.reduce(t, reduction="sum").item())

    for task in ['seqrec', 'item2index']:
        for c in range(num_codebooks):
            metrics[task]['correct'][c] = gather(metrics[task]['correct'][c])
        metrics[task]['total'] = gather(metrics[task]['total'])
        metrics[task]['exact'] = gather(metrics[task]['exact'])
    metrics['index2item']['total'] = gather(metrics['index2item']['total'])
    metrics['index2item']['exact'] = gather(metrics['index2item']['exact'])

    return metrics, topk_metrics


def log_metrics(metrics, topk_metrics, num_codebooks, epoch, logger, wandb_log=False):
    """Log evaluation metrics."""
    log_dict = {"epoch": epoch}

    for task in ['seqrec', 'item2index']:
        total = metrics[task]['total']
        if total > 0:
            c_accs = [metrics[task]['correct'][c] / total for c in range(num_codebooks)]
            exact = metrics[task]['exact'] / total
            logger.info(f"Epoch {epoch} - {task}: exact={exact:.4f}, " + ", ".join([f"c{c}={c_accs[c]:.4f}" for c in range(num_codebooks)]))
            log_dict[f"eval/{task}_exact"] = exact
            for c in range(num_codebooks):
                log_dict[f"eval/{task}_c{c}"] = c_accs[c]

    if metrics['seqrec']['total'] > 0:
        logger.info(f"Epoch {epoch} - seqrec TopK: " + ", ".join([f"{k}={v:.4f}" for k, v in topk_metrics.items()]))
        for k, v in topk_metrics.items():
            log_dict[f"eval/seqrec_{k}"] = v

    if metrics['index2item']['total'] > 0:
        rate = metrics['index2item']['exact'] / metrics['index2item']['total']
        logger.info(f"Epoch {epoch} - index2item: match={rate:.4f}")
        log_dict["eval/index2item_match"] = rate

    if wandb_log:
        wandb.log(log_dict)


@gin.configurable
def train(
    epochs=4, batch_size=8, learning_rate=5e-5, weight_decay=0.01, warmup_ratio=0.01,
    gradient_accumulate_every=2, max_length=512,
    pretrained_path="Qwen/Qwen2.5-1.5B", use_lora=True,
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    num_codebooks=5, codebook_size=256,
    dataset=None, dataset_folder="dataset/amazon", max_seq_len=20, max_text_len=128,
    pretrained_rqvae_path="./out/lcrec/amazon/beauty/rqvae/checkpoint.pt",
    do_eval=True, eval_every_epoch=1, eval_batch_size=64, eval_beam_width=10,
    save_dir_root="out/lcrec/amazon/beauty", save_every_epoch=1,
    wandb_logging=False, wandb_project="lcrec_training", wandb_run_name=None, wandb_log_interval=10,
    split_batches=True, amp=True, mixed_precision_type="bf16",
    max_train_samples=0, max_eval_samples=0, debug_logging=False,
    eval_only=False, checkpoint_path=None,
):
    """Train an LCRec model."""
    logger = setup_logger(save_dir_root)
    accelerator = Accelerator(
        split_batches=split_batches,
        gradient_accumulation_steps=gradient_accumulate_every,
        mixed_precision=mixed_precision_type if amp else "no",
    )
    device = accelerator.device

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        wandb.init(project=wandb_project, name=wandb_run_name, config=locals())
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("eval/*", step_metric="epoch")

    # Model setup
    model = LCRec(pretrained_path=pretrained_path)
    model.add_codebook_tokens(num_codebooks=num_codebooks, codebook_size=codebook_size)

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
    helper = ConstrainedDecodingHelper(num_codebooks, codebook_size, tokenizer)

    # Dataset setup
    ds_kwargs = dict(root=dataset_folder, max_seq_len=max_seq_len, max_text_len=max_text_len, pretrained_rqvae_path=pretrained_rqvae_path)
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

    collate_train = lambda x: lcrec_collate_fn(x, tokenizer, max_length, num_codebooks, is_eval=False)
    collate_eval = lambda x: lcrec_collate_fn(x, tokenizer, max_length, num_codebooks, is_eval=True)

    train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_train)
    valid_dl = DataLoader(valid_ds, batch_size=eval_batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_eval)
    test_dl = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_eval)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_dl, valid_dl, test_dl = accelerator.prepare(train_dl, valid_dl, test_dl)

    total_steps = len(train_dl) * epochs // gradient_accumulate_every
    num_warmup = int(total_steps * warmup_ratio)
    logger.info(f"Total steps: {total_steps}, Warmup: {num_warmup}")

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup, total_steps)
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    logger.info(f"Device: {device}, Params: {sum(p.numel() for p in model.parameters()):,}, Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if checkpoint_path:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        accelerator.unwrap_model(model).load_pretrained(checkpoint_path)

    # Eval only mode
    if eval_only:
        logger.info("Running eval-only mode...")
        metrics, topk = evaluate(model, valid_dl, accelerator, tokenizer, helper, num_codebooks, eval_beam_width, logger, 0, debug_logging)
        if accelerator.is_main_process:
            log_metrics(metrics, topk, num_codebooks, 0, logger)
        accelerator.wait_for_everyone()
        return

    # Training loop
    pbar = tqdm(total=total_steps, dynamic_ncols=True) if accelerator.is_main_process else None
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_steps = 0.0, 0

        for data in train_dl:
            with accelerator.accumulate(model):
                outputs = model(input_ids=data["input_ids"], attention_mask=data["attention_mask"], labels=data["labels"])
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
                        log_dict = {"global_step": global_step, "train/lr": lr_scheduler.get_last_lr()[0], "train/loss": loss.item()}
                        # item2index accuracy
                        tasks = data['tasks']
                        mask = torch.tensor([t == 'item2index' for t in tasks], device=device)
                        if mask.any():
                            with torch.no_grad():
                                logits, labels_m = outputs.logits[mask], data["labels"][mask]
                                valid = labels_m[:, 1:] != -100
                                if valid.any():
                                    acc = ((logits[:, :-1].argmax(-1) == labels_m[:, 1:]) & valid).sum().float() / valid.sum()
                                    log_dict["train/acc_item2index"] = acc.item()
                        wandb.log(log_dict)

            accelerator.wait_for_everyone()

        logger.info(f"Epoch {epoch} - avg_loss: {epoch_loss / max(epoch_steps, 1):.4f}")

        # Evaluation
        if do_eval and (epoch + 1) % eval_every_epoch == 0:
            metrics, topk = evaluate(model, valid_dl, accelerator, tokenizer, helper, num_codebooks, eval_beam_width, logger, epoch, debug_logging)
            if accelerator.is_main_process:
                log_metrics(metrics, topk, num_codebooks, epoch, logger, wandb_logging)
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

    if wandb_logging and accelerator.is_main_process:
        wandb.finish()
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parse_config()
    train()
