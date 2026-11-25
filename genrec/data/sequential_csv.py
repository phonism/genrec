"""Chronological sequence CSV adapter for exact public ML-1M input reuse.

This is separate from the repository's five-core Amazon/ML-1M recipes.
It preserves the source CSV IDs, temporal order, catalogue and held-out target.
The default native_fixed_final_epoch protocol trains without the last item,
then evaluates that item only after the fixed training endpoint.
"""
from __future__ import annotations

import ast
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset


def _int_sequence(value):
    parsed = ast.literal_eval(value)
    return [int(parsed)] if isinstance(parsed, (int, float)) else [int(x) for x in parsed]


class SequentialCSVDataset(Dataset):
    def __init__(self, path, phase="train", max_seq_len=200, max_output_len=11,
                 evaluation_protocol="native_fixed_final_epoch", max_rows=None,
                 catalog_path=None):
        if phase not in {"train", "valid", "test"}:
            raise ValueError("phase must be train, valid or test")
        if evaluation_protocol not in {"native_fixed_final_epoch", "leave_one_out"}:
            raise ValueError("unsupported sequence evaluation protocol")
        if phase == "valid" and evaluation_protocol == "native_fixed_final_epoch":
            raise ValueError("native_fixed_final_epoch has no validation selection")
        self.phase, self.max_seq_len = phase, max_seq_len
        self.padded_length = max_seq_len + max_output_len
        if self.padded_length <= max_seq_len:
            raise ValueError("CSV parity recipe reserves at least one output position")
        self.evaluation_protocol = evaluation_protocol
        self.path = Path(path)
        self.rows = []
        maximum = 0
        with self.path.open() as stream:
            for row in csv.DictReader(stream):
                items = _int_sequence(row["sequence_item_ids"])
                times = _int_sequence(row["sequence_timestamps"])
                if len(items) != len(times) or min(items, default=0) < 1:
                    raise ValueError("sequence IDs/timestamps are invalid or misaligned")
                maximum = max(maximum, max(items))
                if len(items) < 3:
                    continue
                # Native train/eval use ignore_last_n=1/0. The explicit LOO
                # alternative holds two items out and selects only on valid.
                holdout = (2 if phase == "train" else 1 if phase == "valid" else 0) \
                    if evaluation_protocol == "leave_one_out" else (1 if phase == "train" else 0)
                selected_items = items[:-holdout] if holdout else items
                if len(selected_items) < 2:
                    continue
                selected_times = times[:-holdout] if holdout else times
                self.rows.append((selected_items[-max_seq_len-1:], selected_times[-max_seq_len-1:]))
                if max_rows is not None and len(self.rows) >= max_rows:
                    break
        if not self.rows:
            raise ValueError("sequence CSV produced no samples")
        self.candidate_ids = None
        if catalog_path is not None:
            with Path(catalog_path).open() as stream:
                reader = csv.DictReader(stream)
                name = "movie_id" if "movie_id" in reader.fieldnames else "item_id"
                self.candidate_ids = list(dict.fromkeys(int(row[name]) for row in reader))
            maximum = max(maximum, max(self.candidate_ids))
        self.num_items = maximum

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        items, times = self.rows[index]
        length = len(items) - 1
        ids = torch.zeros(self.padded_length, dtype=torch.long)
        stamps = torch.zeros_like(ids)
        targets = torch.zeros_like(ids)
        ids[:length] = torch.tensor(items[:-1])
        stamps[:length + 1] = torch.tensor(times)
        if self.phase == "train":
            targets[:length] = torch.tensor(items[1:])
        return {"input_ids": ids, "timestamps": stamps, "targets": targets,
                "lengths": torch.tensor(length), "target_items": torch.tensor(items[-1])}
