#!/usr/bin/env python3
"""One-shot validation/test evaluation for a frozen TIGER checkpoint.

The training search can keep ``train.test_during_training=False`` and call this
script only after the configuration and checkpoint-selection rule are frozen.
Headline metrics always compare catalog item IDs; exact SID metrics are emitted
only as a collision/decoding diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gin
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from genrec.data.amazon import AmazonSeqDataset
from genrec.models.tiger import Tiger
from genrec.modules.metrics import SemanticIdItemMapper, TopKAccumulator
from genrec.trainers.tiger_trainer import t5_collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="beauty")
    parser.add_argument("--eval-split", choices=("valid", "test"), default="valid")
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--infer-batch-size", type=int)
    parser.add_argument("--output-json")
    parser.add_argument("--gin", action="append", default=[])
    return parser.parse_args()


def query(name: str):
    return gin.query_parameter(f"train.{name}")


def load_config(args: argparse.Namespace) -> None:
    content = Path(args.config_path).read_text().replace("{split}", args.split)
    gin.parse_config(content)
    if args.gin:
        gin.parse_config([value.replace("{split}", args.split) for value in args.gin])


def main() -> None:
    args = parse_args()
    load_config(args)

    codebook_size = int(query("codebook_size"))
    sem_id_dim = int(query("sem_id_dim"))
    max_seq_len = int(query("max_seq_len"))
    include_user_token = bool(query("include_user_token"))
    num_user_tokens = int(query("num_user_tokens"))
    padding_side = str(query("padding_side"))
    rqvae_path = str(query("pretrained_rqvae_path"))
    beam_size = args.beam_size or int(query("beam_size"))
    infer_batch_size = args.infer_batch_size or int(query("infer_batch_size"))

    train_dataset = AmazonSeqDataset(
        root=str(query("dataset_folder")),
        split=args.split,
        train_test_split="train",
        max_seq_len=max_seq_len,
        pretrained_rqvae_path=rqvae_path,
    )
    eval_dataset = AmazonSeqDataset(
        root=str(query("dataset_folder")),
        split=args.split,
        train_test_split=args.eval_split,
        max_seq_len=max_seq_len,
        pretrained_rqvae_path=rqvae_path,
    )
    observed_sid_dim = len(eval_dataset.sem_ids_list[0])
    if observed_sid_dim != sem_id_dim:
        raise ValueError(
            f"dataset SID width {observed_sid_dim} != model width {sem_id_dim}"
        )

    mapper = SemanticIdItemMapper.from_sequences(
        train_dataset.sem_ids_list, train_dataset.sequences
    )
    raw_mapper = SemanticIdItemMapper(
        getattr(train_dataset, "raw_sem_ids_list", train_dataset.sem_ids_list)
    )
    max_history_tokens = max_seq_len * sem_id_dim
    collate_fn = lambda batch: t5_collate(
        batch,
        codebook_size=codebook_size,
        sem_id_dim=sem_id_dim,
        max_history_tokens=max_history_tokens,
        include_user_token=include_user_token,
        num_user_tokens=num_user_tokens,
        padding_side=padding_side,
    )
    loader = DataLoader(
        eval_dataset,
        batch_size=infer_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model_config = {
        "num_layers": int(query("num_layers")),
        "num_decoder_layers": int(query("num_decoder_layers")),
        "d_model": int(query("d_model")),
        "d_ff": int(query("d_ff")),
        "num_heads": int(query("num_heads")),
        "d_kv": int(query("d_kv")),
        "dropout_rate": float(query("dropout_rate")),
        "vocab_size": (
            codebook_size * sem_id_dim
            + (num_user_tokens if include_user_token else 0)
            + 1
        ),
        "pad_token_id": 0,
        "eos_token_id": 0,
        "feed_forward_proj": str(query("feed_forward_proj")),
        "sem_id_dim": sem_id_dim,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Tiger(model_config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
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
        for batch in tqdm(loader, desc=f"TIGER {args.eval_split}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            targets = batch["target_items"]
            batch_size = input_ids.size(0)
            predictions = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=beam_size,
            )[:, 1:].reshape(batch_size, beam_size, sem_id_dim).cpu()
            sid_acc.accumulate(labels, predictions)

            raw_predictions = predictions.clone()
            for codebook_idx in range(sem_id_dim):
                raw_predictions[:, :, codebook_idx] -= (
                    codebook_idx * codebook_size + 1
                )
            predicted_items = mapper.map_predictions(raw_predictions)
            item_acc.accumulate(targets.unsqueeze(-1), predicted_items.unsqueeze(-1))

            raw_rows = raw_predictions.tolist()
            total_generated += batch_size * beam_size
            valid_generated += sum(
                tuple(int(value) for value in sid) in mapper.sid_to_item
                for row in raw_rows
                for sid in row
            )
            valid_per_row = predicted_items.ne(mapper.padding_item_id).sum(1)
            coverage_at_5 += valid_per_row.ge(5).sum().item()
            coverage_at_10 += valid_per_row.ge(10).sum().item()
            total_examples += batch_size

    metrics = item_acc.reduce()
    sid_metrics = sid_acc.reduce()
    diagnostics = {
        "invalid_sid_rate": 1.0 - valid_generated / max(total_generated, 1),
        "valid_candidates_at_least_5_rate": coverage_at_5 / max(total_examples, 1),
        "valid_candidates_at_least_10_rate": coverage_at_10 / max(total_examples, 1),
    }
    result = {
        "model": "tiger",
        "split": args.split,
        "eval_split": args.eval_split,
        "protocol": train_dataset.evaluation_protocol,
        "evaluation_level": "item",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "n_examples": len(eval_dataset),
        "n_items": mapper.num_items,
        "raw_sid_collision_rate": raw_mapper.collision_rate,
        "final_sid_collision_rate": mapper.collision_rate,
        "beam_size": beam_size,
        "metrics": metrics,
        "sid_reference_metrics": sid_metrics,
        "generation_diagnostics": diagnostics,
        "model_config": model_config,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

