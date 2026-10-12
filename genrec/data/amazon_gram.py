"""
Amazon GRAM Dataset for GRAM model training.

Loads pre-processed data files from GRAM's rec_datasets/ directory:
- user_sequence.txt: User interaction sequences (original item IDs)
- item_plain_text.txt: Item text descriptions
- similar_item_sasrec.txt: SASRec-precomputed similar items
- item_generative_indexing_*.txt: Pre-computed hierarchical lexical IDs

Constructs multi-granularity inputs for the FiD encoder:
- Passage 0 (user prompt): Coarse-grained user history using lexical IDs
- Passage 1..N (item text): Fine-grained item descriptions + similar item lexical IDs

Reference: GRAM (ACL 2025)
"""
import os
import gin
import torch
import logging

from torch.utils.data import Dataset
from typing import Dict, List


logger = logging.getLogger(__name__)


def read_lines(path: str) -> List[str]:
    """Read lines from a file, stripping trailing newlines."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r") as f:
        return [line.rstrip("\n") for line in f]


def parse_dict_from_lines(lines: List[str]) -> Dict[str, str]:
    """Parse 'key value' lines into a dict."""
    result = {}
    for line in lines:
        parts = line.split(" ", 1)
        if len(parts) == 2:
            result[parts[0]] = parts[1]
    return result


def load_similar_items(path: str, top_k: int = 10) -> Dict[str, List[str]]:
    """Load similar items from file. First line is header."""
    result = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("anchor"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                item = parts[0]
                similar = parts[1:top_k + 1]
                result[item] = similar
    return result


@gin.configurable
class AmazonGRAMDataset(Dataset):
    """
    Amazon dataset for GRAM training/validation/test.

    Loads pre-processed data files and constructs multi-granularity inputs:
    - User prompt with lexical ID history (coarse-grained)
    - Per-item text descriptions with similar item lexical IDs (fine-grained)
    """

    def __init__(
        self,
        root: str = "dataset/gram",
        split: str = "Beauty",
        train_test_split: str = "train",
        max_seq_len: int = 20,
        max_text_len: int = 128,
        # GRAM-specific
        item_id_file: str = "item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt",
        item_prompt: str = "all_text",
        top_k_similar_item: int = 10,
        cf_model: str = "sasrec",
        his_sep: str = " ; ",
        skip_empty_his: bool = True,
        reverse_history: bool = True,
    ) -> None:
        self.root = root
        self.split = split
        self.train_test_split = train_test_split
        self.max_his = max_seq_len
        self.max_text_len = max_text_len
        self.his_sep = his_sep
        self.skip_empty_his = skip_empty_his
        self.reverse_history = reverse_history

        data_dir = os.path.join(root, split)

        # Load user sequences: {user_id: [item_id1, item_id2, ...]}
        user_seq_lines = read_lines(os.path.join(data_dir, "user_sequence.txt"))
        self.user_seq_dict = {}
        for line in user_seq_lines:
            parts = line.split()
            if len(parts) >= 2:
                self.user_seq_dict[parts[0]] = parts[1:]

        # Load lexical IDs: {item_id: lexical_id_string}
        item_id_path = os.path.join(data_dir, item_id_file)
        item_id_lines = read_lines(item_id_path)
        self.item2lexid = parse_dict_from_lines(item_id_lines)

        # Load item text descriptions: {item_id: text}
        item_text_path = os.path.join(data_dir, "item_plain_text.txt")
        item_text_lines = read_lines(item_text_path)
        item_text_dict = parse_dict_from_lines(item_text_lines)

        # Load similar items from CF model
        cf_path = os.path.join(data_dir, f"similar_item_{cf_model}.txt")
        cf_item_dict = load_similar_items(cf_path, top_k=top_k_similar_item)

        # Build item2input: maps item_id to its multi-granularity text representation
        # Format: "similar items: <lexid1>, <lexid2>, ...; <item_text>"
        self.item2input = {}
        for item_id, text in item_text_dict.items():
            similar_items = cf_item_dict.get(item_id, [])
            cf_verb = []
            for sim_item in similar_items:
                if sim_item in self.item2lexid:
                    cf_verb.append(self.item2lexid[sim_item])
            if cf_verb:
                self.item2input[item_id] = f"similar items: {', '.join(cf_verb)}; {text}"
            else:
                self.item2input[item_id] = text

        # All valid lexical IDs (for Trie construction)
        self.all_items = list(self.item2lexid.values())

        # Load data splits
        self.data_samples = []
        if train_test_split == "train":
            self._load_train()
        elif train_test_split == "valid":
            self._load_validation()
        elif train_test_split == "test":
            self._load_test()
        else:
            raise ValueError(f"Unknown train_test_split: {train_test_split}")

        # Construct input/output pairs
        self._construct_sentences()

        logger.info(f"GRAM {split} {train_test_split}: {len(self.data)} samples, "
                    f"{len(self.all_items)} items")

    def _load_train(self):
        """Load training data with sliding window augmentation."""
        for user, items in self.user_seq_dict.items():
            seq = items[:-2]  # Leave last 2 for valid/test
            for i in range(len(seq)):
                if i == 0 and self.skip_empty_his:
                    continue

                sample = {
                    "user_id": user,
                    "target": seq[i],
                    "target_lex_id": self.item2lexid.get(seq[i], ""),
                }

                history = seq[:i]
                if self.max_his > 0:
                    history = history[-self.max_his:]

                # Item text for each history item (fine-grained passages)
                history_input = [self.item2input.get(h, "") for h in history]

                if self.reverse_history:
                    history = history[::-1]
                    history_input = history_input[::-1]

                # Lexical IDs for history items (coarse-grained, reversed)
                history_lex_ids = [self.item2lexid.get(h, "") for h in history]

                sample["history_input"] = history_input
                sample["history_lex_id"] = self.his_sep.join(history_lex_ids)

                self.data_samples.append(sample)

    def _load_validation(self):
        """Load validation data (2nd-to-last item as target)."""
        for user, items in self.user_seq_dict.items():
            if len(items) < 3:
                continue
            target = items[-2]
            history = items[:-2]
            if self.max_his > 0:
                history = history[-self.max_his:]

            history_input = [self.item2input.get(h, "") for h in history]

            if self.reverse_history:
                history = history[::-1]
                history_input = history_input[::-1]

            history_lex_ids = [self.item2lexid.get(h, "") for h in history]

            self.data_samples.append({
                "user_id": user,
                "target": target,
                "target_lex_id": self.item2lexid.get(target, ""),
                "history_input": history_input,
                "history_lex_id": self.his_sep.join(history_lex_ids),
            })

    def _load_test(self):
        """Load test data (last item as target)."""
        for user, items in self.user_seq_dict.items():
            if len(items) < 2:
                continue
            target = items[-1]
            history = items[:-1]
            if self.max_his > 0:
                history = history[-self.max_his:]

            history_input = [self.item2input.get(h, "") for h in history]

            if self.reverse_history:
                history = history[::-1]
                history_input = history_input[::-1]

            history_lex_ids = [self.item2lexid.get(h, "") for h in history]

            self.data_samples.append({
                "user_id": user,
                "target": target,
                "target_lex_id": self.item2lexid.get(target, ""),
                "history_input": history_input,
                "history_lex_id": self.his_sep.join(history_lex_ids),
            })

    def _construct_sentences(self):
        """
        Construct multi-granularity input/output pairs.

        Input is a list of passages:
        - [0]: User prompt with coarse-grained lexical ID history
        - [1..N]: Fine-grained per-item text descriptions

        Output is the target item's lexical ID.
        """
        self.data = []
        for sample in self.data_samples:
            # Passage 0: user prompt with lexical ID history
            user_sentence = f"What would user purchase after {sample['history_lex_id']} ?"
            input_passages = [user_sentence] + sample["history_input"]

            self.data.append({
                "input": input_passages,
                "output": sample["target_lex_id"],
                "user_id": sample["user_id"],
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GRAMCollator:
    """
    Collator for GRAM multi-granularity inputs.

    Handles variable-length passages (user prompt + item texts) and pads them
    to create tensors of shape B x (N+1) x L for the FiD encoder.

    Args:
        tokenizer: T5 tokenizer
        item_prompt_max_len: Max token length per passage
        target_max_len: Max token length for target
        max_item_num: Max number of history items (N)
    """

    def __init__(self, tokenizer, item_prompt_max_len=128, target_max_len=50, max_item_num=20):
        self.tokenizer = tokenizer
        self.item_prompt_max_len = item_prompt_max_len
        self.target_max_len = target_max_len
        self.max_item_num = max_item_num

    def __call__(self, batch):
        input_texts = [item["input"] for item in batch]
        output_texts = [item["output"] for item in batch]
        user_ids = [item["user_id"] for item in batch]

        # Encode target
        target = self.tokenizer.batch_encode_plus(
            output_texts,
            max_length=self.target_max_len if self.target_max_len > 0 else None,
            padding="longest",
            return_tensors="pt",
            truncation=True if self.target_max_len > 0 else False,
        )
        target_ids = target["input_ids"]
        target_masks = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_masks, -100)

        # Encode multi-granularity inputs: B x (N+1) x L
        item_text_ids, item_text_masks = self._encode_passages(input_texts)

        return {
            "target_ids": target_ids,
            "target_masks": target_masks,
            "item_text_ids": item_text_ids,
            "item_text_masks": item_text_masks,
            "user_ids": user_ids,
        }

    def _encode_passages(self, batch_item_texts):
        """Encode all passages per sample, pad to uniform shape."""
        item_text_ids, item_text_masks = [], []

        max_item_batch = max(len(texts) for texts in batch_item_texts)
        max_item_num = min(max_item_batch, self.max_item_num + 1)  # +1 for user prompt

        for text_passages in batch_item_texts:
            # Truncate to max passages
            passages = text_passages[:max_item_num]

            p = self.tokenizer.batch_encode_plus(
                passages,
                max_length=self.item_prompt_max_len,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )

            cur_ids = p["input_ids"]
            cur_mask = p["attention_mask"]

            # Pad to uniform number of passages
            if len(passages) < max_item_num:
                pad_count = max_item_num - len(passages)
                cur_ids = torch.cat([
                    cur_ids,
                    torch.zeros(pad_count, self.item_prompt_max_len, dtype=torch.long),
                ], dim=0)
                cur_mask = torch.cat([
                    cur_mask,
                    torch.zeros(pad_count, self.item_prompt_max_len, dtype=torch.long),
                ], dim=0)

            item_text_ids.append(cur_ids.unsqueeze(0))
            item_text_masks.append(cur_mask.unsqueeze(0))

        item_text_ids = torch.cat(item_text_ids, dim=0)    # B x N x L
        item_text_masks = torch.cat(item_text_masks, dim=0)  # B x N x L

        # Trim to actual max text length (remove trailing padding columns)
        max_text_len = item_text_masks.sum(-1).max().item()
        if max_text_len > 0:
            max_text_len = int(max_text_len)
            item_text_ids = item_text_ids[:, :, :max_text_len]
            item_text_masks = item_text_masks[:, :, :max_text_len]

        return item_text_ids, item_text_masks.bool()
