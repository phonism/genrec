#!/usr/bin/env python3
"""Trigger 179's cooperative ``hold.py`` workers, then exec a real GPU job.

This helper must be used *inside* ``run-locked-179.sh`` with
``GENREC_ALLOW_OCCUPIED_GPU=1``.  A tiny CUDA allocation makes the real job
visible to the external hold service; after that service yields its memory, the
requested command replaces this process without releasing the cooperative GPU
lock.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--min-free-gib", type=float, default=16.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # 64 MiB is deliberately small enough to fit beside a high-water hold.py
    # worker while still registering this process as real CUDA work.
    trigger = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    trigger.zero_()
    torch.cuda.synchronize()
    print("CUDA trigger allocated; waiting for hold.py to yield", flush=True)

    required_free = int(args.min_free_gib * 1024**3)
    deadline = time.monotonic() + args.timeout
    while True:
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes >= required_free:
            break
        if time.monotonic() >= deadline:
            free_gib = free_bytes / 1024**3
            raise RuntimeError(
                f"hold.py did not yield within {args.timeout:g}s "
                f"(free={free_gib:.2f} GiB)"
            )
        time.sleep(0.5)

    del trigger
    torch.cuda.empty_cache()
    free_gib = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"hold.py yielded; free={free_gib:.2f} GiB; starting command", flush=True)
    os.execvp(command[0], command)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"trigger-hold-yield: {exc}", file=sys.stderr, flush=True)
        raise
