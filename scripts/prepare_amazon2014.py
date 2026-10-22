#!/usr/bin/env python3
"""Atomically download and validate Amazon Reviews 2014 raw files.

This is deliberately separate from the dataset constructors: it serializes
downloads across Codex sessions, never exposes a partial file at the final
path, and emits a checksum manifest for reproducible remote runs.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request


BASE_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
DATASETS = {
    "beauty": ("reviews_Beauty_5.json.gz", "meta_Beauty.json.gz"),
    "sports": (
        "reviews_Sports_and_Outdoors_5.json.gz",
        "meta_Sports_and_Outdoors.json.gz",
    ),
    "toys": ("reviews_Toys_and_Games_5.json.gz", "meta_Toys_and_Games.json.gz"),
    "home": ("reviews_Home_and_Kitchen_5.json.gz", "meta_Home_and_Kitchen.json.gz"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("dataset/amazon"))
    parser.add_argument(
        "--splits", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS)
    )
    parser.add_argument(
        "--kinds", nargs="+", choices=("reviews", "meta"), default=("reviews", "meta")
    )
    return parser.parse_args()


def decode_first_record(line: str, path: Path) -> None:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        record = ast.literal_eval(line)
    if not isinstance(record, dict):
        raise ValueError(f"first record in {path} is not a mapping")


def validate_gzip(path: Path) -> tuple[int, str]:
    """Read through EOF (including CRC) and validate the first record."""
    digest = hashlib.sha256()
    with path.open("rb") as raw:
        while chunk := raw.read(1024 * 1024):
            digest.update(chunk)

    first_line = ""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not first_line and line.strip():
                first_line = line.strip()
            # Exhaust the stream so gzip verifies its trailer/CRC.
    if not first_line:
        raise ValueError(f"empty gzip stream: {path}")
    decode_first_record(first_line, path)
    return path.stat().st_size, digest.hexdigest()


def download_atomic(url: str, destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(destination.name + ".download.lock")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process is preparing {destination}") from exc

        if destination.exists():
            size, digest = validate_gzip(destination)
            print(f"VALID {destination} bytes={size} sha256={digest}", flush=True)
            return size, digest

        temporary = destination.with_name(f".{destination.name}.part.{os.getpid()}")
        try:
            print(f"GET   {url}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "genrec-repro/1"})
            with urllib.request.urlopen(request, timeout=120) as response:
                with temporary.open("xb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
            size, digest = validate_gzip(temporary)
            os.replace(temporary, destination)
            print(f"READY {destination} bytes={size} sha256={digest}", flush=True)
            return size, digest
        finally:
            if temporary.exists():
                temporary.unlink()


def main() -> int:
    args = parse_args()
    manifest: list[str] = []
    kind_index = {"reviews": 0, "meta": 1}
    try:
        for split in args.splits:
            for kind in args.kinds:
                filename = DATASETS[split][kind_index[kind]]
                destination = args.root / "raw" / split / filename
                size, digest = download_atomic(f"{BASE_URL}/{filename}", destination)
                manifest.append(f"{digest}  {size}  {destination}")
    except Exception as exc:
        print(f"ERROR {exc}", file=sys.stderr, flush=True)
        return 1

    manifest_key = "-".join(sorted(args.splits)) + "." + "-".join(sorted(args.kinds))
    manifest_path = args.root / "raw" / f"SHA256SUMS.amazon2014.{manifest_key}.txt"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.part.{os.getpid()}")
    manifest_tmp.write_text("\n".join(manifest) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    print(f"MANIFEST {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
