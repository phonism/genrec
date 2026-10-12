#!/usr/bin/env python3
"""Build one atomic item-embedding cache for each requested Amazon split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from genrec.data.amazon import AmazonItemDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument(
        "--splits", nargs="+", default=("beauty", "sports", "toys")
    )
    parser.add_argument("--root", default="dataset/amazon")
    parser.add_argument(
        "--item-text-format",
        choices=("legacy", "tiger_paper"),
        default="legacy",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(cache: Path, *, split: str, encoder: str,
                   item_text_format: str, items: int,
                   dim: int, digest: str) -> Path:
    manifest = cache.with_suffix(cache.suffix + ".manifest.json")
    payload = {
        "cache": str(cache),
        "bytes": cache.stat().st_size,
        "sha256": digest,
        "split": split,
        "encoder": encoder,
        "item_text_format": item_text_format,
        "items": items,
        "dim": dim,
        "source_bundle": os.environ.get("GENREC_SOURCE_BUNDLE_NAME"),
        "source_bundle_sha256": os.environ.get("GENREC_SOURCE_BUNDLE_SHA256"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    temporary = manifest.with_name(f"{manifest.name}.part.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest)
    return manifest


def main() -> None:
    args = parse_args()
    for split in args.splits:
        dataset = AmazonItemDataset(
            root=args.root,
            split=split,
            train_test_split="all",
            encoder_model_name=args.encoder,
            item_text_format=args.item_text_format,
        )
        cache = Path(dataset.parquet_path)
        digest = sha256(cache)
        manifest = write_manifest(
            cache,
            split=split,
            encoder=args.encoder,
            item_text_format=args.item_text_format,
            items=len(dataset),
            dim=dataset.dim,
            digest=digest,
        )
        print(
            f"READY split={split} items={len(dataset)} dim={dataset.dim} "
            f"bytes={cache.stat().st_size} sha256={digest} path={cache} "
            f"manifest={manifest}",
            flush=True,
        )


if __name__ == "__main__":
    main()
