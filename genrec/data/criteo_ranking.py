"""Training-vocabulary encoded Criteo parquet data for GenRec CTR training.

Reads the frozen feature_map.json and split parquet files from preprocessing;
never fits vocabulary or normalization on validation/test. FuxiCTR is not a
runtime dependency. Use the existing native preparation entry to reproduce
the public Criteo_x1 encoding and retain its data-manifest.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from genrec.models.rankmixer import ranking_feature_specs


class CriteoRankingDataset(Dataset):
    def __init__(self, data_folder, phase="train", max_rows=None):
        import pyarrow as pa
        import pyarrow.parquet as pq

        if phase not in {"train", "valid", "test"}:
            raise ValueError("phase must be train, valid or test")
        if max_rows is not None and max_rows < 1:
            raise ValueError("max_rows must be positive")
        root = Path(data_folder)
        self.feature_map_path = root / "feature_map.json"
        self.feature_map = json.loads(self.feature_map_path.read_text())
        self.feature_specs = ranking_feature_specs(self.feature_map)
        self.feature_names = list(self.feature_specs)
        labels = self.feature_map.get("labels", ["label"])
        if len(labels) != 1:
            raise ValueError("CTR currently requires exactly one binary label")
        self.label_name = labels[0]
        self.parquet_path = root / f"{phase}.parquet"
        columns = self.feature_names + [self.label_name]
        parquet = pq.ParquetFile(self.parquet_path)
        if max_rows is None:
            table = parquet.read(columns=columns)
        else:
            batches, count = [], 0
            for batch in parquet.iter_batches(batch_size=min(max_rows, 65536), columns=columns):
                batches.append(batch)
                count += batch.num_rows
                if count >= max_rows:
                    break
            table = pa.Table.from_batches(batches).slice(0, max_rows)
        self.features = np.column_stack([
            table[name].to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            for name in self.feature_names
        ])
        self.labels = table[self.label_name].to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        if not len(self.labels) or not np.isfinite(self.features).all():
            raise ValueError("empty or non-finite CTR features")
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError("CTR labels must be binary")
        for i, (name, spec) in enumerate(self.feature_specs.items()):
            if spec["type"] == "categorical":
                values = self.features[:, i]
                if (values < 0).any() or (values >= spec["vocab_size"]).any() or (values != np.floor(values)).any():
                    raise ValueError(f"encoded IDs do not match training vocabulary for {name}")
        self.num_rows = len(self.labels)

    def __len__(self):
        return self.num_rows

    def __getitem__(self, index):
        return {
            "features": torch.from_numpy(self.features[index]),
            "labels": torch.tensor(self.labels[index]),
        }
