#!/usr/bin/env python3
"""Bounded real-backbone smoke test for the NoteLLM benchmark path."""

import argparse
import math

import torch

from genrec.data.amazon_notellm import (
    AmazonNoteLLMDataset,
    format_item_csft_prompt,
    format_item_prompt,
)
from genrec.models.notellm import Query2Embedding


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="beauty")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("NoteLLM real-backbone smoke requires CUDA")

    dataset = AmazonNoteLLMDataset(
        root=args.dataset_root,
        split=args.split,
        train_test_split="test",
        evaluation_protocol="leave_one_out",
    )
    model = Query2Embedding(
        pretrained_path=args.pretrained_path,
        gradient_checkpointing=True,
    )

    from peft import LoraConfig, TaskType, get_peft_model

    model.model = get_peft_model(
        model.model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    model.model.enable_input_require_grads()
    model.cuda()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if trainable_count <= 0 or trainable_count >= 100_000_000:
        raise RuntimeError(f"unexpected trainable parameter count: {trainable_count}")

    pair_prompts = []
    pair_targets = []
    for sample in dataset.samples[:2]:
        pair_ids = [sample["query"], sample["target"]]
        pair_prompts.extend([
            format_item_csft_prompt(dataset.item_texts[item_id])
            for item_id in pair_ids
        ])
        pair_targets.extend([
            dataset.item_categories[item_id] for item_id in pair_ids
        ])
    tokens = {
        key: value.cuda()
        for key, value in model.tokenize_with_targets(
            pair_prompts, pair_targets, max_length=128
        ).items()
    }
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**tokens, return_loss=True)
    loss = output["loss"]
    if not math.isfinite(float(loss.detach())):
        raise RuntimeError(f"non-finite NoteLLM loss: {loss}")
    loss.backward()
    if not any(parameter.grad is not None for parameter in trainable):
        raise RuntimeError("NoteLLM smoke produced no trainable gradients")

    model.eval()
    embedding_prompts = [
        format_item_prompt(dataset.item_texts[item_id])
        for sample in dataset.samples[:2]
        for item_id in [sample["query"], sample["target"]]
    ]
    embedding_tokens = {
        key: value.cuda()
        for key, value in model.tokenize(
            embedding_prompts, truncation=True, max_length=128
        ).items()
    }
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        embeddings = model(
            **embedding_tokens, return_loss=False
        )["sentence_embedding"]
    metrics = Query2Embedding.compute_full_catalog_metrics(
        query_embeddings=embeddings[::2],
        catalog_embeddings=embeddings,
        target_item_ids=torch.tensor([11, 21], device=embeddings.device),
        catalog_item_ids=torch.tensor([10, 11, 20, 21], device=embeddings.device),
        excluded_item_ids=torch.tensor([10, 20], device=embeddings.device),
        topks=(1, 2),
    )
    if not all(0.0 <= value <= 1.0 for value in metrics.values()):
        raise RuntimeError(f"invalid retrieval metrics: {metrics}")
    print(
        {
            "status": "ok",
            "split": args.split,
            "items": dataset.num_items,
            "queries": len(dataset),
            "trainable_params": trainable_count,
            "loss": float(loss.detach()),
            "embedding_shape": list(embeddings.shape),
            "metrics": metrics,
        }
    )


if __name__ == "__main__":
    main()
