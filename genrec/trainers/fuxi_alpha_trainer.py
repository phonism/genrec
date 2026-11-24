"""FuXi-alpha training in GenRec, independently of the upstream trainer.

The native_fixed_final_epoch protocol deliberately has no validation-based
selection; its input CSV and catalogue match the original public ML-1M run.
The optional leave_one_out protocol has a separate validation split/selection.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import gin
import torch
from torch.utils.data import DataLoader

from genrec.data.sequential_csv import SequentialCSVDataset
from genrec.models.fuxi_alpha import FuXiAlpha
from genrec.models.fuxi_beta import FuXiBeta
from genrec.models.fuxi_gamma import FuXiGamma
from genrec.modules.utils import parse_config, setup_logger
from genrec.trainers.trainer_utils import (
    file_sha256, save_checkpoint, save_run_results, set_seed, setup_accelerator,
)


@torch.no_grad()
def evaluate(model, loader, accelerator, candidate_ids, top_ks=(1, 10, 50)):
    model.eval()
    totals = {f"{name}@{k}": 0.0 for name in ("Recall", "NDCG") for k in top_ks}
    totals["MRR"] = 0.0
    count = 0
    candidates = candidate_ids.to(accelerator.device)
    # Match the public evaluator, including its rank-2501 fallback for MRR.
    maxk = min(max(2500, max(top_ks)), len(candidates))
    for batch in loader:
        logits, _ = model(batch["input_ids"], batch["timestamps"], last_only=True)
        scores = logits.index_select(-1, candidates)
        # The source evaluator removes historical items before full ranking.
        seen = batch["input_ids"]
        mask = torch.zeros(len(logits), logits.shape[-1], dtype=torch.bool, device=logits.device)
        mask.scatter_(1, seen, True)
        scores = scores.masked_fill(mask.index_select(-1, candidates), float("-inf"))
        ranked = candidates[scores.topk(maxk, dim=-1).indices]
        ranked, targets = accelerator.gather_for_metrics((ranked, batch["target_items"]))
        matches = ranked.eq(targets[:, None])
        for k in top_ks:
            selected = matches[:, :k]
            discount = torch.log2(torch.arange(2, selected.shape[1] + 2, device=ranked.device).float())
            totals[f"Recall@{k}"] += selected.any(-1).float().sum().item()
            totals[f"NDCG@{k}"] += (selected.float() / discount).sum().item()
        matched_rank = matches.long().argmax(-1) + 1
        matched_rank = torch.where(matches.any(-1), matched_rank, 2501)
        totals["MRR"] += matched_rank.float().reciprocal().sum().item()
        count += len(targets)
    if not count:
        raise ValueError("empty sequence evaluation loader")
    return {name: value / count for name, value in totals.items()}


@gin.configurable
def train(dataset_path="dataset/ml-1m/sasrec_format.csv", catalog_path=None,
          save_dir_root="out/framework/fuxi_alpha/ml1m/seed{seed}", seed=42,
          evaluation_protocol="native_fixed_final_epoch", epochs=101,
          batch_size=128, eval_batch_size=128, learning_rate=1e-3, weight_decay=0.0,
          max_seq_len=200, max_output_len=11, embed_dim=50, num_blocks=2,
          num_heads=1, dqk=50, dv=50, dropout=0.2, ffn_multiply=1,
          ffn_single_stage=False, num_negatives=128, temperature=0.05,
          patience=10, num_workers=4, max_rows=None, amp=False,
          mixed_precision_type="bf16", enable_tf32=True, interaction_bf16=True,
          log_interval=100, model_name="fuxi_alpha"):
    config = dict(locals())
    classes = {"fuxi_alpha": FuXiAlpha, "fuxi_beta": FuXiBeta, "fuxi_gamma": FuXiGamma}
    references = {"fuxi_alpha": "USTC-StarTeam/FuXi-alpha@69b46a9ddff8f6fa40c057a02b9f86e7bf38983e",
                  "fuxi_beta": "USTC-StarTeam/FuXi-beta@4fc73bc6b3a4d66d6a37e4178d567ca83edb7a73",
                  "fuxi_gamma": "Yeedzhi/FuXi-gamma@b434e5daab49cc973740af7b4d219f7133b9d84b"}
    if model_name not in classes:
        raise ValueError("unknown FuXi model")
    implementation = classes[model_name].__module__
    config["implementation"] = implementation
    if min(epochs, batch_size, eval_batch_size, num_negatives, log_interval) < 1:
        raise ValueError("training counts must be positive")
    if evaluation_protocol == "native_fixed_final_epoch" and catalog_path is None:
        raise ValueError("the native ML-1M comparison requires its processed item catalogue CSV")
    output = Path(save_dir_root.format(seed=seed))
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(output), name=model_name)
    set_seed(seed)
    accelerator = setup_accelerator(amp=amp, mixed_precision_type=mixed_precision_type)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    if accelerator.is_main_process:
        (output / "progress.json").write_text(json.dumps({"stage": "loading_data", "seed": seed}) + "\n")
    ds_args = dict(path=dataset_path, max_seq_len=max_seq_len, max_output_len=max_output_len,
                   evaluation_protocol=evaluation_protocol, max_rows=max_rows, catalog_path=catalog_path)
    train_ds = SequentialCSVDataset(phase="train", **ds_args)
    test_ds = SequentialCSVDataset(phase="test", **ds_args)
    config["train_samples"] = len(train_ds)
    config["test_samples"] = len(test_ds)
    kwargs = dict(num_items=train_ds.num_items, max_seq_len=max_seq_len,
                  max_output_len=max_output_len, embed_dim=embed_dim, num_blocks=num_blocks,
                  num_heads=num_heads, dqk=dqk, dv=dv, dropout=dropout,
                  ffn_multiply=ffn_multiply, ffn_single_stage=ffn_single_stage,
                  temperature=temperature, candidate_ids=train_ds.candidate_ids,
                  interaction_bf16=interaction_bf16)
    model = classes[model_name](**kwargs)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  betas=(0.9, 0.98), weight_decay=weight_decay)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)
    valid_dl = None
    if evaluation_protocol == "leave_one_out":
        valid_ds = SequentialCSVDataset(phase="valid", **ds_args)
        valid_dl = accelerator.prepare(DataLoader(valid_ds, batch_size=eval_batch_size,
                                                  shuffle=False, num_workers=num_workers))
    if accelerator.is_main_process:
        (output / "operative.gin").write_text(gin.operative_config_str())
        (output / "resolved-config.json").write_text(json.dumps(config, indent=2) + "\n")
    logger.info("GenRec %s: %s parameters, %s users, protocol=%s", model_name, params, len(train_ds), evaluation_protocol)
    history, best, best_epoch, wait = [], float("-inf"), -1, 0
    checkpoint = output / "checkpoint_best.pt"
    for epoch in range(epochs):
        model.train()
        total, steps = 0.0, 0
        epoch_start = time.time()
        for batch in train_dl:
            optimizer.zero_grad()
            _, loss = model(batch["input_ids"], batch["timestamps"], batch["targets"], num_negatives=num_negatives)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite FuXi loss")
            accelerator.backward(loss)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
            if steps % log_interval == 0 and accelerator.is_main_process:
                progress = {"stage": "training", "epoch": epoch, "batch": steps,
                            "batches": len(train_dl), "train_loss": total / steps}
                (output / "progress.json").write_text(json.dumps(progress) + "\n")
                logger.info("epoch=%s batch=%s/%s loss=%.6f", epoch, steps, len(train_dl), total / steps)
        row = {"epoch": epoch, "train_loss": total / max(1, steps), "epoch_seconds": time.time() - epoch_start}
        selected = evaluation_protocol == "native_fixed_final_epoch" and epoch == epochs - 1
        if valid_dl is not None:
            metrics = evaluate(model, valid_dl, accelerator, accelerator.unwrap_model(model).candidate_ids)
            row["validation"] = metrics
            if metrics["Recall@10"] > best:
                best, wait, selected = metrics["Recall@10"], 0, True
            else:
                wait += 1
        history.append(row)
        if selected:
            best_epoch = epoch
            if accelerator.is_main_process:
                save_checkpoint(str(checkpoint), accelerator.unwrap_model(model), optimizer,
                                epoch=epoch, extra={"model_name": model_name, "model_config": kwargs})
        logger.info("epoch=%s loss=%.6f selected=%s", epoch, row["train_loss"], selected)
        if accelerator.is_main_process:
            (output / "metrics-by-epoch.json").write_text(json.dumps(history, indent=2) + "\n")
            (output / "progress.json").write_text(json.dumps({"stage": "epoch_complete", **row}) + "\n")
        accelerator.wait_for_everyone()
        if valid_dl is not None and wait >= patience:
            break
    state = torch.load(checkpoint, map_location=accelerator.device, weights_only=False)
    accelerator.unwrap_model(model).load_state_dict(state["model"])
    test_dl = accelerator.prepare(DataLoader(test_ds, batch_size=eval_batch_size,
                                             shuffle=False, num_workers=num_workers))
    metrics = evaluate(model, test_dl, accelerator, accelerator.unwrap_model(model).candidate_ids)
    if accelerator.is_main_process:
        files = [dataset_path] + ([catalog_path] if catalog_path else [])
        save_run_results(str(output), model_name, "ml1m", seed, metrics, config, {
            "evaluation_level": "item", "implementation": implementation,
            "upstream_network_imported": False, "best_epoch": best_epoch,
            "selection": "fixed_final_epoch" if valid_dl is None else "validation_recall_at_10",
            "parameters_total": params, "data_sha256": {str(p): file_sha256(p) for p in files},
            "checkpoint_sha256": file_sha256(checkpoint), "elapsed_seconds": time.time() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "smoke": max_rows is not None,
            "protocol_note": "source CSV/catalogue and target timestamps; separate from Amazon matrix",
            "mrr_definition": "public evaluator: top-2500 rank, absent target assigned rank 2501",
            "upstream_reference": references[model_name],
            "mechanism_scope": "released dense causal base; diagonal-sparse kernels not claimed" if model_name == "fuxi_gamma" else "published default base",
        })
        (output / "progress.json").write_text(json.dumps({"stage": "completed", "test": metrics}) + "\n")
        logger.info("test=%s", metrics)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    parse_config()
    train()
