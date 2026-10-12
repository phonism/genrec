"""Evaluate a frozen RPG checkpoint with direct and graph decoding.

Validation is the safe default.  After graph hyperparameters are frozen, pass
``--gin "eval_rpg_graph.eval_split='test'"`` for the one-shot test evaluation.
"""
import hashlib
import json
import os

import gin
import torch
from torch.utils.data import DataLoader

from genrec.models.rpg import RPGModel
from genrec.data.amazon_rpg import AmazonRPGDataset, rpg_eval_collate_fn
from genrec.modules.utils import parse_config
from genrec.trainers.rpg_trainer import evaluate
from genrec.trainers.trainer_utils import setup_accelerator, set_seed


@gin.configurable
def eval_rpg_graph(
    n_embd=448,
    n_layer=2,
    n_head=4,
    n_inner=1024,
    n_codebook=32,
    codebook_size=256,
    temperature=0.07,
    embd_pdrop=0.5,
    attn_pdrop=0.5,
    resid_pdrop=0.0,
    num_beams=50,
    n_edges=50,
    propagation_steps=3,
    chunk_size=1024,
    max_seq_len=50,
    dataset_folder="dataset/amazon",
    split="beauty",
    encoder_model_name="sentence-transformers/sentence-t5-xl",
    sent_emb_pca=0,
    eval_batch_size=32,
    eval_split="valid",
    checkpoint_path=None,
    output_json=None,
    save_dir_root="out/rpg/amazon/{split}",
    amp=False,
    mixed_precision_type="no",
    seed=42,
):
    save_dir_root = save_dir_root.replace("{split}", split)
    if eval_split not in {"valid", "test"}:
        raise ValueError(f"eval_split must be 'valid' or 'test', got {eval_split!r}")
    set_seed(seed)
    accelerator = setup_accelerator(amp=amp, mixed_precision_type=mixed_precision_type)
    device = accelerator.device

    # The tokenizer is fit from protocol-visible training items regardless of
    # which held-out split is evaluated.
    ds_kwargs = dict(
        root=dataset_folder,
        split=split,
        max_seq_len=max_seq_len,
        n_codebook=n_codebook,
        codebook_size=codebook_size,
        encoder_model_name=encoder_model_name,
        sent_emb_pca=sent_emb_pca,
    )
    eval_ds = AmazonRPGDataset(train_test_split=eval_split, **ds_kwargs)
    collate_eval = lambda x: rpg_eval_collate_fn(x, max_seq_len)
    eval_dl = DataLoader(
        eval_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_eval,
    )

    # Load model
    item_id2tokens = eval_ds.item_id2tokens.to(device)
    model = RPGModel(
        n_items=eval_ds.num_items,
        n_codebook=n_codebook,
        codebook_size=codebook_size,
        item_id2tokens=item_id2tokens,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_inner,
        max_seq_len=max_seq_len,
        temperature=temperature,
        embd_pdrop=embd_pdrop,
        attn_pdrop=attn_pdrop,
        resid_pdrop=resid_pdrop,
        num_beams=num_beams,
        n_edges=n_edges,
        propagation_steps=propagation_steps,
        chunk_size=chunk_size,
    )

    ckpt_path = checkpoint_path or f"{save_dir_root}/best_model.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    eval_dl = accelerator.prepare(eval_dl)
    model = accelerator.prepare(model)

    # Direct scoring
    print("\n=== Direct Scoring ===")
    direct_metrics = evaluate(model, eval_dl, accelerator)
    for k, v in direct_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Graph-constrained decoding
    print("\n=== Graph-Constrained Decoding ===")
    graph_metrics = evaluate(model, eval_dl, accelerator, use_graph=True)
    for k, v in graph_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Comparison
    print("\n=== Comparison (NDCG@10) ===")
    d = direct_metrics.get("NDCG@10", 0)
    g = graph_metrics.get("NDCG@10", 0)
    print(f"  Direct: {d:.4f}")
    print(f"  Graph:  {g:.4f}")
    if d > 0:
        print(f"  Delta:  {g - d:+.4f} ({(g - d) / d * 100:+.1f}%)")

    if accelerator.is_main_process:
        artifact_path = output_json or os.path.join(
            save_dir_root, f"eval_{eval_split}_graph.json"
        )
        artifact_path = artifact_path.replace("{split}", split)
        os.makedirs(os.path.dirname(artifact_path) or ".", exist_ok=True)
        with open(ckpt_path, "rb") as stream:
            checkpoint_digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checkpoint_digest.update(chunk)
            checkpoint_sha256 = checkpoint_digest.hexdigest()
        payload = {
            "model": "rpg",
            "split": split,
            "eval_split": eval_split,
            "evaluation_protocol": eval_ds.evaluation_protocol,
            "seed": seed,
            "checkpoint": ckpt_path,
            "checkpoint_sha256": checkpoint_sha256,
            "direct_metrics": dict(direct_metrics),
            "graph_metrics": dict(graph_metrics),
            "graph_config": {
                "num_beams": num_beams,
                "n_edges": n_edges,
                "propagation_steps": propagation_steps,
            },
            "source": {
                "cell_id": os.environ.get("GENREC_CELL_ID"),
                "bundle": os.environ.get("GENREC_SOURCE_BUNDLE_NAME"),
                "bundle_sha256": os.environ.get("GENREC_SOURCE_BUNDLE_SHA256"),
                "git_sha": os.environ.get("GENREC_SOURCE_GIT_SHA"),
            },
        }
        temporary = f"{artifact_path}.part.{os.getpid()}"
        try:
            with open(temporary, "w") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
            os.replace(temporary, artifact_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(f"Saved evaluation artifact: {artifact_path}")


if __name__ == "__main__":
    parse_config()
    eval_rpg_graph()
