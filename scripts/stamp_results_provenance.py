#!/usr/bin/env python3
"""Backfill an exact source-bundle identity into existing results.json files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--bundle-name", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--git-sha")
    parser.add_argument("--cell-id")
    parser.add_argument("--physical-gpus")
    parser.add_argument("--task-id", type=int)
    return parser.parse_args()


def result_files(paths: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for path in paths:
        if path.is_file():
            if path.name != "results.json":
                raise ValueError(f"expected results.json, got {path}")
            found.add(path)
        elif path.is_dir():
            found.update(path.rglob("results.json"))
        else:
            raise FileNotFoundError(path)
    return sorted(found)


def stamp(
    path: Path,
    bundle_name: str,
    sha256: str,
    git_sha: str | None,
    cell_id: str | None = None,
    physical_gpus: str | None = None,
    task_id: int | None = None,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    current_sha = payload.get("source_bundle_sha256")
    if current_sha not in (None, "", sha256):
        raise ValueError(f"{path} already belongs to different bundle {current_sha}")
    current_name = payload.get("source_bundle")
    if current_name not in (None, "", bundle_name):
        raise ValueError(f"{path} already belongs to different bundle {current_name}")

    payload["source_bundle"] = bundle_name
    payload["source_bundle_sha256"] = sha256
    if git_sha is not None:
        current_git_sha = payload.get("git_sha")
        if current_git_sha not in (None, "", "unknown", git_sha):
            raise ValueError(
                f"{path} already records different git SHA {current_git_sha}"
            )
        payload["git_sha"] = git_sha
    supplied_execution = {
        "cell_id": cell_id,
        "physical_gpus": physical_gpus,
        "task_id": task_id,
    }
    supplied_execution = {
        key: value for key, value in supplied_execution.items() if value is not None
    }
    if supplied_execution:
        execution = payload.setdefault("execution", {})
        if not isinstance(execution, dict):
            raise ValueError(f"{path} execution provenance is not an object")
        for key, value in supplied_execution.items():
            current = execution.get(key)
            if current not in (None, "", value):
                raise ValueError(
                    f"{path} already records different execution {key}={current!r}"
                )
            execution[key] = value
    temporary = path.with_name(f".{path.name}.part.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    print(f"STAMPED {path}")


def main() -> int:
    args = parse_args()
    if not SHA256_RE.fullmatch(args.sha256):
        print("--sha256 must be 64 lowercase hex characters", file=sys.stderr)
        return 2
    if args.git_sha is not None and not re.fullmatch(r"[0-9a-f]{7,40}", args.git_sha):
        print("--git-sha must be 7-40 lowercase hex characters", file=sys.stderr)
        return 2
    try:
        files = result_files(args.paths)
        if not files:
            raise ValueError("no results.json files found")
        for path in files:
            stamp(
                path,
                args.bundle_name,
                args.sha256,
                args.git_sha,
                cell_id=args.cell_id,
                physical_gpus=args.physical_gpus,
                task_id=args.task_id,
            )
    except Exception as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
