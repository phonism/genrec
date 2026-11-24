"""GenRec training/evaluation for its own RankMixer and RankElastor modules.

Uses the shared Gin CLI, Accelerator, checkpoint and results contract. It
does not import FuxiCTR, an upstream network, or an upstream training loop.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import gin
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss, roc_auc_score

from genrec.data.criteo_ranking import CriteoRankingDataset
from genrec.models.rankmixer import RankElastor, RankMixer
from genrec.models.tokenmixer_large import TokenMixerLarge, SparsePerTokenMoE, PAPER
from genrec.models.wukong import Wukong
from genrec.modules.utils import parse_config, setup_logger
from genrec.trainers.trainer_utils import (
    file_sha256, save_checkpoint, save_run_results, set_seed, setup_accelerator,
)


@torch.no_grad()
def evaluate(model, loader, accelerator):
    model.eval()
    labels, predictions = [], []
    for batch in loader:
        logits, _ = model(batch["features"])
        y, p = accelerator.gather_for_metrics((batch["labels"].reshape(-1), logits.sigmoid().reshape(-1)))
        labels.append(y.float().cpu().numpy())
        predictions.append(p.float().cpu().numpy())
    if not labels:
        raise ValueError("empty CTR evaluation loader")
    y, p = np.concatenate(labels), np.concatenate(predictions).astype(np.float64)
    return {"AUC": float(roc_auc_score(y, p)),
            "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))}


@gin.configurable
def train(model_name="rankmixer", dataset_folder="dataset/criteo_x1/criteo_x1_7b681156",
          save_dir_root="out/framework/rankmixer/criteo_x1/seed{seed}", seed=42,
          epochs=100, batch_size=4096, eval_batch_size=4096, learning_rate=1e-3,
          embedding_dim=20, token_num=15, token_dim=30, num_blocks=2,
          ffn_hidden=30, tokenizer_hidden=500, tokenizer_dropout=0.1,
          expansion_rate=3, embedding_regularizer=1e-5, patience=2,
          min_delta=1e-6, reduce_lr_on_plateau=True, max_gradient_norm=10.0,
          num_workers=3, max_train_rows=None, max_eval_rows=None,
          amp=False, mixed_precision_type="bf16", log_interval=100,
          optimizer_name="adam", embedding_learning_rate=None, tokenmixer_params=None,
          wukong_params=None):
    run_config = dict(locals())
    run_config["evaluation_protocol"] = "criteo_x1_validation_logloss"
    if model_name not in {"rankmixer", "rankelastor", "tokenmixer_large", "wukong"}:
        raise ValueError("unknown ranking model")
    if optimizer_name not in {"adam", "adagrad"}:
        raise ValueError("optimizer_name must be adam or adagrad")
    if min(epochs, batch_size, eval_batch_size, patience, log_interval) < 1:
        raise ValueError("epochs, batch sizes and patience must be positive")
    output = Path(save_dir_root.format(seed=seed))
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(output), name=model_name)
    set_seed(seed)
    accelerator = setup_accelerator(amp=amp, mixed_precision_type=mixed_precision_type)
    started = time.time()
    if accelerator.is_main_process:
        (output / "progress.json").write_text(json.dumps({"stage": "loading_data", "seed": seed}) + "\n")
    train_ds = CriteoRankingDataset(dataset_folder, "train", max_train_rows)
    valid_ds = CriteoRankingDataset(dataset_folder, "valid", max_eval_rows)
    # Test features are loaded only after validation selection has finished.
    run_config["train_samples"] = len(train_ds)
    run_config["valid_samples"] = len(valid_ds)
    kwargs = dict(feature_map=train_ds.feature_map, embedding_dim=embedding_dim,
                  token_num=token_num, token_dim=token_dim, num_blocks=num_blocks,
                  tokenizer_hidden=tokenizer_hidden,
                  tokenizer_dropout=tokenizer_dropout, embedding_regularizer=embedding_regularizer)
    model_cls = {"rankmixer": RankMixer, "rankelastor": RankElastor, "tokenmixer_large": TokenMixerLarge, "wukong": Wukong}[model_name]
    if model_name == "wukong":
        kwargs = dict(feature_map=train_ds.feature_map, embedding_dim=embedding_dim,
                      num_blocks=num_blocks, embedding_regularizer=embedding_regularizer)
        kwargs.update(wukong_params or {})
    elif model_name == "tokenmixer_large":
        kwargs.update(tokenmixer_params or {})
    else:
        kwargs["ffn_hidden"] = ffn_hidden
    if model_name == "rankelastor":
        kwargs["expansion_rate"] = expansion_rate
    model = model_cls(**kwargs)
    implementation = type(model).__module__
    run_config["implementation"] = implementation
    model_info = {
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "parameters_embedding": sum(m.weight.numel() for m in model.modules() if isinstance(m, torch.nn.Embedding)),
    }
    model_info["parameters_non_embedding"] = model_info["parameters_total"] - model_info["parameters_embedding"]
    inactive_expert_parameters = sum(
        m.slots * (m.num_experts - m.top_k) * 3 * m.width * m.hidden
        for m in model.modules() if isinstance(m, SparsePerTokenMoE)
    )
    model_info["parameters_active_non_embedding_per_example_training"] = model_info["parameters_non_embedding"] - inactive_expert_parameters
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        sparse_parameters = [m.weight for m in model.modules() if isinstance(m, torch.nn.Embedding)]
        sparse_ids = {id(p) for p in sparse_parameters}
        dense_parameters = [p for p in model.parameters() if id(p) not in sparse_ids]
        optimizer = torch.optim.Adagrad([
            {"params": dense_parameters, "lr": learning_rate},
            {"params": sparse_parameters, "lr": embedding_learning_rate or learning_rate},
        ])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available())
    valid_dl = DataLoader(valid_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
    model, optimizer, train_dl, valid_dl = accelerator.prepare(model, optimizer, train_dl, valid_dl)
    logger.info("GenRec %s: %s; %s train samples", model_name, model_info, len(train_ds))
    if accelerator.is_main_process:
        (output / "operative.gin").write_text(gin.operative_config_str())
        (output / "resolved-config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    best, wait, best_epoch, history = float("inf"), 0, -1, []
    checkpoint = output / "checkpoint_best.pt"
    for epoch in range(epochs):
        model.train()
        epoch_start, total_loss, batches = time.time(), 0.0, 0
        for batch in train_dl:
            optimizer.zero_grad()
            _, loss = model(batch["features"], batch["labels"])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite CTR training loss")
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_gradient_norm)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
            if batches % log_interval == 0 and accelerator.is_main_process:
                progress = {"stage": "training", "epoch": epoch, "batch": batches,
                            "batches": len(train_dl), "train_loss": total_loss / batches}
                (output / "progress.json").write_text(json.dumps(progress) + "\n")
                logger.info("epoch=%s batch=%s/%s loss=%.6f", epoch, batches, len(train_dl), total_loss / batches)
        metrics = evaluate(model, valid_dl, accelerator)
        row = {"epoch": epoch, "train_loss": total_loss / max(1, batches),
               "validation": metrics, "lr": optimizer.param_groups[0]["lr"],
               "epoch_seconds": time.time() - epoch_start}
        history.append(row)
        if metrics["logloss"] <= best - min_delta:
            best, wait, best_epoch = metrics["logloss"], 0, epoch
            if accelerator.is_main_process:
                save_checkpoint(str(checkpoint), accelerator.unwrap_model(model), optimizer,
                                epoch=epoch, extra={"model_name": model_name, "model_config": kwargs,
                                                    "valid_metrics": metrics})
        else:
            wait += 1
            if reduce_lr_on_plateau:
                for group in optimizer.param_groups:
                    group["lr"] = max(group["lr"] * 0.1, 1e-6)
        logger.info("epoch=%s valid=%s best_epoch=%s wait=%s", epoch, metrics, best_epoch, wait)
        if accelerator.is_main_process:
            (output / "metrics-by-epoch.json").write_text(json.dumps(history, indent=2) + "\n")
            (output / "progress.json").write_text(json.dumps({"stage": "epoch_complete", **row}) + "\n")
            unwrapped = accelerator.unwrap_model(model)
            if hasattr(unwrapped, "routing_summary"):
                (output / "routing-stats.json").write_text(json.dumps(unwrapped.routing_summary(), indent=2) + "\n")
        accelerator.wait_for_everyone()
        if wait >= patience:
            break
    if best_epoch < 0:
        raise RuntimeError("no validation-selected checkpoint was saved")
    state = torch.load(checkpoint, map_location=accelerator.device, weights_only=False)
    accelerator.unwrap_model(model).load_state_dict(state["model"])
    test_ds = CriteoRankingDataset(dataset_folder, "test", max_eval_rows)
    test_dl = accelerator.prepare(DataLoader(test_ds, batch_size=eval_batch_size,
                                            shuffle=False, num_workers=num_workers))
    test = evaluate(model, test_dl, accelerator)
    if accelerator.is_main_process:
        input_files = [train_ds.feature_map_path, train_ds.parquet_path, valid_ds.parquet_path, test_ds.parquet_path]
        hashes = {str(p): file_sha256(p) for p in input_files}
        extra = {"evaluation_level": "exposure", "task": "binary_CTR", "model_info": model_info,
                 "implementation": implementation, "trainer": __name__,
                 "upstream_network_imported": False, "best_epoch": best_epoch,
                 "best_valid_logloss": best, "data_sha256": hashes,
                 "checkpoint_sha256": file_sha256(checkpoint),
                 "elapsed_seconds": time.time() - started,
                 "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                 "smoke": max_train_rows is not None or max_eval_rows is not None,
                 "paper_scope": "independent_paper_public_CTR_adaptation" if model_name in {"tokenmixer_large", "wukong"} else "public_CTR_configuration_adaptation",
                 "code_origin": "paper_based_backbone_with_framework_feature_adapter" if model_name in {"tokenmixer_large", "wukong"} else "MIT_public_source_port",
                 "paper_reference": ("https://arxiv.org/abs/2403.02545v4" if model_name == "wukong" else PAPER) if model_name in {"tokenmixer_large", "wukong"} else None,
                 "upstream_reference": None if model_name in {"tokenmixer_large", "wukong"} else "vasile-paskardlgm/RankElastor@68e5f29c9147b0c2263a1d28216b500940c057fe",
                 "optimizer": optimizer_name}
        save_run_results(str(output), model_name, "criteo_x1", seed, test, run_config, extra)
        (output / "progress.json").write_text(json.dumps({"stage": "completed", "test": test}) + "\n")
        logger.info("test=%s", test)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    parse_config()
    train()
