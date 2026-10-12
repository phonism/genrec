"""Amazon 2014 next-item data adapter for NoteLLM.

The adapter shares the repository's leave-one-out/global-temporal splitter and
raw item-ID mapping.  It adds deterministic text prompts while keeping the
headline target as the exact item ID, so evaluation can rank the full catalog.
"""

from __future__ import annotations

import fcntl
import json
import os
from typing import Dict, Iterable, List

import gin
from tqdm import tqdm

from genrec.data.amazon import DATASET_CONFIGS, parse_gzip_json
from genrec.data.amazon_sasrec import AmazonSASRecDataset


def _flatten_text(value) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        flattened: List[str] = []
        for child in value:
            flattened.extend(_flatten_text(child))
        return flattened
    if isinstance(value, dict):
        return [f"{key}: {child}" for key, child in value.items()]
    text = str(value).strip()
    return [text] if text else []


def build_item_text(
    meta: dict,
    fallback: str,
    include_categories: bool = True,
) -> str:
    """Build a compact, deterministic textual item representation."""
    fields = []
    source_fields = [
        ("title", "title"),
        ("brand", "brand"),
        ("description", "description"),
    ]
    if include_categories:
        source_fields.insert(2, ("category", "categories"))
    for label, key in source_fields:
        values = list(_flatten_text(meta.get(key)))
        if values:
            fields.append(f"{label}: {'; '.join(values)}")
    return " | ".join(fields)[:4000] if fields else f"item: {fallback}"


def format_item_prompt(item_text: str, item_token: str = "[EMB]") -> str:
    clean = item_text.replace(item_token, " ").strip()
    return f"Represent this catalog item for retrieval: {clean} {item_token}"


def format_item_csft_prompt(item_text: str, item_token: str = "[EMB]") -> str:
    """Joint compression/category-generation prompt used by NoteLLM CSFT."""
    return format_item_prompt(item_text, item_token) + "\nGenerate its category: "


@gin.configurable
class AmazonNoteLLMDataset(AmazonSASRecDataset):
    """Protocol-aware NoteLLM I2I pairs with exact item-ID targets.

    NoteLLM is natively an item-to-item retriever, not a sequence encoder.  An
    Amazon transition therefore uses the immediately preceding item as the
    query item and the next item as the positive.  This keeps the adaptation
    faithful while using the same protocol targets as the sequential models.
    """

    def __init__(
        self,
        root: str = "dataset/amazon",
        split: str = "beauty",
        train_test_split: str = "train",
        max_seq_len: int = 50,
        min_seq_len: int = 5,
        evaluation_protocol: str = "leave_one_out",
        temporal_train_ratio: float = 0.8,
        temporal_valid_ratio: float = 0.1,
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            train_test_split=train_test_split,
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            evaluation_protocol=evaluation_protocol,
            temporal_train_ratio=temporal_train_ratio,
            temporal_valid_ratio=temporal_valid_ratio,
        )
        self.item_texts = self._load_item_texts()

    def _generate_samples(self) -> None:
        self.samples = []
        if self.train_test_split == "train":
            for full_seq in tqdm(self.sequences, desc="Generating NoteLLM train samples"):
                train_seq = full_seq[:-2]
                for target_index in range(1, len(train_seq)):
                    self.samples.append(
                        {
                            "query": train_seq[target_index - 1],
                            "target": train_seq[target_index],
                        }
                    )
            return

        for full_seq in self.sequences:
            if self.train_test_split == "valid":
                sequence = full_seq[:-1]
            else:
                sequence = full_seq
            if len(sequence) < 2:
                continue
            self.samples.append(
                {
                    "query": sequence[-2],
                    "target": sequence[-1],
                }
            )

    def __getitem__(self, index: int) -> dict:
        """Return the native I2I schema instead of SASRec's history schema."""
        sample = self.samples[index]
        return {
            "query": int(sample["query"]),
            "target": int(sample["target"]),
        }

    def _load_item_texts(self) -> Dict[int, str]:
        processed_dir = os.path.join(self.root, "processed", self.split)
        os.makedirs(processed_dir, exist_ok=True)
        # Versioned because the CSFT target (category) must not be copied into
        # its own input prompt.  Older caches included categories in item text.
        cache_path = os.path.join(processed_dir, "notellm_item_texts_csft_v2.json")
        category_cache_path = os.path.join(
            processed_dir, "notellm_item_categories.json"
        )

        with open(cache_path + ".lock", "a+") as cache_lock:
            fcntl.flock(cache_lock, fcntl.LOCK_EX)
            if os.path.exists(cache_path) and os.path.exists(category_cache_path):
                with open(cache_path) as stream:
                    asin_texts = json.load(stream)
                with open(category_cache_path) as stream:
                    asin_categories = json.load(stream)
            else:
                config = DATASET_CONFIGS[self.split]
                meta_path = os.path.join(
                    self.root, "raw", self.split, config["meta"]
                )
                wanted = set(self.item_id_mapping)
                asin_texts = {}
                asin_categories = {}
                for meta in tqdm(
                    parse_gzip_json(meta_path), desc="Loading NoteLLM item text"
                ):
                    asin = meta.get("asin")
                    if asin in wanted:
                        asin_texts[asin] = build_item_text(
                            meta, fallback=asin, include_categories=False
                        )
                        categories = list(_flatten_text(meta.get("categories")))
                        # Amazon has no hashtags.  Its category path is the
                        # closest available CSFT target and is kept separate
                        # from the input text for explicit response masking.
                        asin_categories[asin] = " > ".join(categories[:8])

                temporary = f"{cache_path}.part.{os.getpid()}"
                category_temporary = (
                    f"{category_cache_path}.part.{os.getpid()}"
                )
                try:
                    with open(temporary, "w") as stream:
                        json.dump(asin_texts, stream, ensure_ascii=False)
                    with open(category_temporary, "w") as stream:
                        json.dump(asin_categories, stream, ensure_ascii=False)
                    os.replace(temporary, cache_path)
                    os.replace(category_temporary, category_cache_path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
                    if os.path.exists(category_temporary):
                        os.unlink(category_temporary)

        self.item_categories = {
            item_id: asin_categories.get(asin) or "unknown category"
            for asin, item_id in self.item_id_mapping.items()
        }

        return {
            item_id: asin_texts.get(asin, f"item: {asin}")
            for asin, item_id in self.item_id_mapping.items()
        }


def notellm_collate_fn(batch: List[dict]) -> dict:
    return {
        "queries": [sample["query"] for sample in batch],
        "targets": [sample["target"] for sample in batch],
    }
