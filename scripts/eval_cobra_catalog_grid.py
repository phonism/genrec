"""Validation-only COBRA catalog-constraint and BeamFusion grid.

This script deliberately never loads the test split.  It reuses one expensive
catalog-constrained beam-search pass for every ``tau``/``psi`` pair, allowing
inference hyperparameters to be selected without touching test labels.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from genrec.data.amazon_cobra import AmazonCobraDataset
from genrec.models.cobra import Cobra
from genrec.modules.metrics import TopKAccumulator
from genrec.trainers.cobra_trainer import cobra_collate_fn


ID_VOCAB = 32
N_CODEBOOKS = 3
D_MODEL = 768


def parse_pair(value: str) -> tuple[float, float]:
    try:
        tau, psi = value.split(",", maxsplit=1)
        return float(tau), float(psi)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("grid values must be TAU,PSI") from exc


def make_dataset(args) -> AmazonCobraDataset:
    return AmazonCobraDataset(
        root="dataset/amazon",
        split=args.split,
        train_test_split="valid",
        max_seq_len=args.max_seq_len,
        pretrained_rqvae_path=args.rqvae,
        encoder_model_name=args.rqvae_encoder,
        max_text_len=128,
        rqvae_input_dim=768,
        rqvae_embed_dim=32,
        rqvae_hidden_dims=[512, 256, 128, 64],
        rqvae_codebook_size=ID_VOCAB,
        rqvae_n_layers=N_CODEBOOKS,
        item_text_format=args.item_text_format,
        rqvae_item_text_format=args.rqvae_item_text_format,
        min_raw_codebook_usage=args.min_raw_codebook_usage,
        evaluation_protocol="leave_one_out",
    )


@torch.no_grad()
def encode_catalog(model, token_ids, device, batch_size=256):
    vectors = []
    for start in range(0, len(token_ids), batch_size):
        tokens = token_ids[start:start + batch_size].to(device)
        vectors.append(model.encoder(tokens.unsqueeze(1)).squeeze(1).cpu())
    return F.normalize(torch.cat(vectors), dim=-1).to(device)


def metrics_dict(accumulator: TopKAccumulator) -> dict[str, float]:
    return {key: float(value) for key, value in accumulator.reduce().items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--rqvae", required=True)
    parser.add_argument("--split", default="beauty")
    parser.add_argument("--rqvae-encoder", required=True)
    parser.add_argument("--model-encoder", default="/root/workspace/models_hub/sentence-t5-base")
    parser.add_argument("--encoder-type", choices=("light", "pretrained"), default="light")
    parser.add_argument("--encoder-pooling", choices=("mean", "cls"), default="cls")
    parser.add_argument("--item-text-format", default="paper_attributes")
    parser.add_argument("--rqvae-item-text-format", default="cobra_paper")
    parser.add_argument("--min-raw-codebook-usage", type=float, default=0.8)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--n-beam", type=int, default=64)
    parser.add_argument(
        "--grid",
        action="append",
        type=parse_pair,
        default=None,
        help="BeamFusion TAU,PSI pair; repeat for a grid",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.grid is None:
        args.grid = [(1.0, 0.0), (1.0, 4.0), (1.0, 8.0), (1.0, 16.0),
                     (1.0, 32.0), (0.5, 16.0), (2.0, 16.0)]

    device = torch.device("cuda")
    dataset = make_dataset(args)
    pad_id = ID_VOCAB * N_CODEBOOKS
    model = Cobra(
        id_vocab_size=ID_VOCAB,
        n_codebooks=N_CODEBOOKS,
        d_model=D_MODEL,
        max_len=1024,
        temperature=0.07,
        decoder_n_layers=2,
        decoder_num_heads=6,
        encoder_type=args.encoder_type,
        encoder_model_name=args.model_encoder,
        encoder_n_layers=1,
        encoder_hidden_dim=D_MODEL,
        encoder_num_heads=6,
        encoder_pooling=args.encoder_pooling,
        dense_negative_scope="all_batch",
    ).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint/model mismatch: missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )
    model.eval()
    print(f"loaded {args.ckpt} exactly")

    item_sem_ids = torch.tensor(dataset.sem_ids_list, dtype=torch.long, device=device)
    sem_id_to_items = defaultdict(list)
    for item_id, semantic_id in enumerate(dataset.sem_ids_list):
        sem_id_to_items[tuple(semantic_id)].append(item_id)
    sem_id_to_items = dict(sem_id_to_items)
    item_dense_vecs = encode_catalog(model, dataset.item_token_ids, device)
    print(
        f"catalog items={len(dataset.sem_ids_list)} unique_sids={len(sem_id_to_items)} "
        f"grid={args.grid}"
    )

    item_metrics = {pair: TopKAccumulator() for pair in args.grid}
    sid_metrics = TopKAccumulator()
    target_beam_hits = 0
    examples = 0
    max_examples = len(dataset)
    if args.max_batches is not None:
        max_examples = min(max_examples, args.max_batches * args.batch_size)

    for start in range(0, max_examples, args.batch_size):
        batch = [dataset[i] for i in range(start, min(start + args.batch_size, max_examples))]
        data = cobra_collate_fn(
            batch, pad_id=pad_id, n_codebooks=N_CODEBOOKS, is_train=False
        )
        input_ids = data["input_ids"].to(device)
        encoder_input_ids = data["encoder_input_ids"].to(device)
        targets = data["target_item_ids"].to(device)
        target_sids = data["target_sem_ids"].to(device)

        generated = model.generate(
            input_ids=input_ids,
            encoder_input_ids=encoder_input_ids,
            n_candidates=args.n_beam,
            valid_sem_ids=item_sem_ids,
        )
        sid_metrics.accumulate(target_sids, generated.sem_ids[:, :10])
        target_beam_hits += (
            generated.sem_ids == target_sids.unsqueeze(1)
        ).all(-1).any(-1).sum().item()

        for pair in args.grid:
            tau, psi = pair
            fused = model.fuse_generated(
                gen_output=generated,
                item_dense_vecs=item_dense_vecs,
                item_sem_ids=item_sem_ids,
                sem_id_to_items=sem_id_to_items,
                n_candidates=10,
                tau=tau,
                psi=psi,
            )
            item_metrics[pair].accumulate(
                targets.unsqueeze(-1), fused.item_ids.unsqueeze(-1)
            )
        examples += len(batch)
        if (start // args.batch_size + 1) % 50 == 0:
            print(f"progress {examples}/{max_examples}", flush=True)

    results = {
        "split": "valid",
        "protocol": "leave_one_out",
        "checkpoint": args.ckpt,
        "catalog_constrained": True,
        "n_examples": examples,
        "n_beam": args.n_beam,
        "target_sid_in_beam": target_beam_hits / max(examples, 1),
        "sid_top10": metrics_dict(sid_metrics),
        "grid": {
            f"tau={tau:g},psi={psi:g}": metrics_dict(item_metrics[(tau, psi)])
            for tau, psi in args.grid
        },
    }
    best_name, best_metrics = max(
        results["grid"].items(), key=lambda entry: (entry[1]["Recall@10"], entry[1]["NDCG@10"])
    )
    results["best_by_recall10"] = {"setting": best_name, **best_metrics}
    print(json.dumps(results, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
