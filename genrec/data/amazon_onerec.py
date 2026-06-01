"""
Amazon OneRec Dataset for OneRec model training.

OneRec uses 4 training tasks (simplified from LCRec's 6):
- sidsft: Given history SIDs, predict next item's SID (= LCRec's seqrec)
- title2sid: Given item title, predict SID (= LCRec's item2index_title)
- sid2title: Given SID, predict item title (= LCRec's index2item_title)
- fusionseqrec: Given history SIDs, predict next item's title

Uses 3 codebooks (vs LCRec's 5).
"""
import os
import gin
import torch
import random

from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional, Set

from genrec.data.amazon import (
    AmazonItemDataset,
    DATASET_CONFIGS,
    parse_gzip_json,
)


# SFT prompt format (same as LCRec)
SFT_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

HISTORY_SEP = ", "
ADD_PREFIX = True


# Prompt templates for OneRec tasks (5 per task, simplified from LCRec)
ONEREC_PROMPT_TEMPLATES = {
    "sidsft": [
        "User interaction history: {history}\nPredict the next item:",
        "Given the user's past interactions: {history}\nWhat item will they interact with next?",
        "The user has interacted with: {history}\nRecommend the next item:",
        "Based on history: {history}\nNext item prediction:",
        "A user has the following purchase history: {history}\nCan you predict the next possible item?",
    ],
    "title2sid": [
        "Item title: {title}\nItem index:",
        "Given the item titled \"{title}\", what is its index?",
        "Find the index for item: {title}\nIndex:",
        "What is the index of the item with title: {title}?",
        "Item named \"{title}\" has index:",
    ],
    "sid2title": [
        "Item index: {index}\nItem title:",
        "Given index {index}, what is the item's title?",
        "Find the title for item with index: {index}",
        "What item has index {index}? Title:",
        "Index {index} corresponds to item titled:",
    ],
    "fusionseqrec": [
        "User interaction history: {history}\nPredict the next item title:",
        "Given history: {history}\nRecommend next item with its name:",
        "Based on: {history}\nNext item title:",
        "History: {history}\nPredict next item's name:",
        "User has interacted with: {history}\nNext item title:",
    ],
}


@gin.configurable
class AmazonOneRecSFTDataset(Dataset):
    """
    Amazon Dataset for OneRec SFT training.

    4 tasks: sidsft, title2sid, sid2title, fusionseqrec.
    Uses 3 codebooks by default.
    """

    def __init__(
        self,
        root: str = "dataset/amazon2023",
        split: str = "arts_crafts_and_sewing",
        train_test_split: str = "train",
        max_seq_len: int = 20,
        max_text_len: int = 128,
        # Tokenizer type: "rqvae" or "rqkmeans"
        tokenizer_type: str = "rqvae",
        # RQ-VAE config (used when tokenizer_type="rqvae")
        pretrained_rqvae_path: str = "./out/onerec/amazon2023/{split}/rqvae/checkpoint.pt",
        encoder_model_name: str = "./models_hub/sentence-t5-xl",
        rqvae_input_dim: int = 768,
        rqvae_embed_dim: int = 64,
        rqvae_hidden_dims: List[int] = [512, 256, 128],
        rqvae_codebook_size: int = 256,
        rqvae_n_layers: int = 3,
        # RQ-KMeans config (used when tokenizer_type="rqkmeans")
        pretrained_rqkmeans_path: str = "",
        rqkmeans_codebook_size: int = 8192,
        rqkmeans_n_layers: int = 3,
        # Task configuration
        enabled_tasks: List[str] = None,
        task_sample_weights: Dict[str, float] = None,
    ) -> None:
        self.root = root
        self.split = split.lower()
        self.train_test_split = train_test_split
        self._max_seq_len = max_seq_len
        self.max_text_len = max_text_len

        self.enabled_tasks: Set[str] = set(enabled_tasks or [
            "sidsft", "title2sid", "sid2title", "fusionseqrec"
        ])

        self.task_sample_weights = task_sample_weights or {
            "sidsft": 1.0,
            "title2sid": 0.5,
            "sid2title": 0.5,
            "fusionseqrec": 0.5,
        }

        # Load item dataset for embeddings
        item_dataset = AmazonItemDataset(
            root=root,
            split=split,
            train_test_split="all",
            encoder_model_name=encoder_model_name,
        )
        self.item_embeddings = torch.tensor(item_dataset.embeddings, dtype=torch.float32)

        # Generate semantic IDs based on tokenizer type
        if tokenizer_type == "rqkmeans":
            from genrec.models.rqkmeans import RqKmeans
            self.n_codebooks = rqkmeans_n_layers
            self.codebook_size = rqkmeans_codebook_size
            path = pretrained_rqkmeans_path.format(split=self.split)
            rqkmeans = RqKmeans.load(path)
            output = rqkmeans.assign(item_dataset.embeddings.astype('float32'))
            self.sem_ids_list = output.sem_ids.tolist()
        else:
            from genrec.models.rqvae import RqVae
            self.n_codebooks = rqvae_n_layers
            self.codebook_size = rqvae_codebook_size
            pretrained_rqvae_path = pretrained_rqvae_path.format(split=self.split)
            rqvae = RqVae(
                input_dim=rqvae_input_dim,
                embed_dim=rqvae_embed_dim,
                hidden_dims=rqvae_hidden_dims,
                codebook_size=rqvae_codebook_size,
                codebook_kmeans_init=False,
                codebook_normalize=False,
                codebook_sim_vq=False,
                n_layers=rqvae_n_layers,
                n_cat_features=0,
                commitment_weight=0.25,
            )
            rqvae.load_pretrained(pretrained_rqvae_path)
            rqvae.eval()
            with torch.no_grad():
                self.sem_ids_list = rqvae.get_semantic_ids(self.item_embeddings).sem_ids.tolist()

        self._load_item_metadata()
        self._load_sequences()
        self._generate_samples()

    def _load_item_metadata(self) -> None:
        """Load item metadata for text generation."""
        config = DATASET_CONFIGS[self.split]
        meta_path = os.path.join(self.root, "raw", self.split, config["meta"])
        reviews_path = os.path.join(self.root, "raw", self.split, config["reviews"])

        # Build item mapping from reviews
        item_id_mapping: Dict[str, int] = {}
        for review in parse_gzip_json(reviews_path):
            asin = review.get('asin')
            if asin and asin not in item_id_mapping:
                item_id_mapping[asin] = len(item_id_mapping)

        # Load metadata
        self.item_titles: Dict[int, str] = {}
        for meta in parse_gzip_json(meta_path):
            asin = meta.get('asin')
            if asin in item_id_mapping:
                item_id = item_id_mapping[asin]
                title = meta.get('title', '')
                self.item_titles[item_id] = title.strip() or f"item_{item_id}"

        # Fill missing items
        for i in range(len(item_id_mapping)):
            if i not in self.item_titles:
                self.item_titles[i] = f"item_{i}"

        self.num_items = len(item_id_mapping)

    def _load_sequences(self) -> None:
        """Load user interaction sequences from reviews."""
        config = DATASET_CONFIGS[self.split]
        reviews_path = os.path.join(self.root, "raw", self.split, config["reviews"])

        user_sequences: Dict[str, List[tuple]] = {}
        item_id_mapping: Dict[str, int] = {}

        for review in parse_gzip_json(reviews_path):
            asin = review.get('asin')
            user_id = review.get('reviewerID')
            timestamp = review.get('unixReviewTime', 0)

            if asin and user_id:
                if asin not in item_id_mapping:
                    item_id_mapping[asin] = len(item_id_mapping)

                item_id = item_id_mapping[asin]
                if user_id not in user_sequences:
                    user_sequences[user_id] = []
                user_sequences[user_id].append((timestamp, item_id))

        self.sequences = []
        self.user_ids = []
        for uid, seq in user_sequences.items():
            seq.sort(key=lambda x: x[0])
            items = [x[1] for x in seq]
            if len(items) >= 5:
                self.sequences.append(items)
                self.user_ids.append(uid)

        print(f"Loaded {len(self.sequences)} user sequences for OneRec SFT")

    def _generate_samples(self) -> None:
        """Generate training/evaluation samples."""
        # Fix random seed to ensure all DDP ranks generate identical samples.
        # Without this, random.random() in fusionseqrec sampling produces different
        # sample counts per rank, causing NCCL deadlock at epoch boundaries.
        rng_state = random.getstate()
        random.seed(42)

        self.samples = []

        if self.train_test_split == "train":
            self._generate_train_samples()
        else:
            self._generate_eval_samples()

        print(f"Generated {len(self.samples)} OneRec SFT samples for {self.train_test_split}")
        task_counts = {}
        for s in self.samples:
            task_counts[s['task']] = task_counts.get(s['task'], 0) + 1
        print(f"Task distribution: {task_counts}")

        # Restore random state so downstream randomness (e.g. DataLoader shuffling) is unaffected
        random.setstate(rng_state)

    def _generate_train_samples(self) -> None:
        """Generate training samples for all enabled tasks.

        Sliding window: for each position i in user sequence, create a sample
        with history[max(0, i-window):i] → target[i]. This generates N-1
        samples per user (where N = sequence length), aligned with MiniOneRec.
        """
        window = self._max_seq_len  # sliding window size

        for user_idx, full_seq in enumerate(self.sequences):
            seq = full_seq[:-2]  # Leave last 2 for valid/test
            if len(seq) < 2:
                continue

            # Sliding window: each position generates a sample
            for i in range(1, len(seq)):
                st = max(0, i - window)
                history = seq[st:i]
                target = seq[i]

                if "sidsft" in self.enabled_tasks:
                    self.samples.append({
                        'task': 'sidsft',
                        'history': history,
                        'target': target,
                    })

                if "fusionseqrec" in self.enabled_tasks:
                    self.samples.append({
                        'task': 'fusionseqrec',
                        'history': history,
                        'target': target,
                    })

        # Item-based tasks (all items, bidirectional SID↔title alignment)
        if "title2sid" in self.enabled_tasks:
            for item_id in range(self.num_items):
                if item_id < len(self.sem_ids_list):
                    self.samples.append({
                        'task': 'title2sid',
                        'item_id': item_id,
                    })

        if "sid2title" in self.enabled_tasks:
            for item_id in range(self.num_items):
                if item_id < len(self.sem_ids_list):
                    self.samples.append({
                        'task': 'sid2title',
                        'item_id': item_id,
                    })

    def _generate_eval_samples(self) -> None:
        """Generate evaluation samples (sidsft only for fair comparison)."""
        if self.train_test_split == "valid":
            for full_seq in self.sequences:
                seq = full_seq[:-1]
                if len(seq) >= 2:
                    history = seq[max(0, len(seq) - 1 - self._max_seq_len):-1]
                    target = seq[-1]
                    self.samples.append({
                        'task': 'sidsft',
                        'history': history,
                        'target': target,
                    })
        else:  # test
            for full_seq in self.sequences:
                if len(full_seq) >= 2:
                    history = full_seq[max(0, len(full_seq) - 1 - self._max_seq_len):-1]
                    target = full_seq[-1]
                    self.samples.append({
                        'task': 'sidsft',
                        'history': history,
                        'target': target,
                    })

    def _sem_ids_to_tokens(self, sem_ids: List[int]) -> str:
        """Convert semantic IDs to codebook token string."""
        return "".join(f"<C{c}_{code}>" for c, code in enumerate(sem_ids))

    def _history_to_tokens(self, history: List[int]) -> str:
        """Convert history item IDs to token string."""
        tokens = []
        for idx, item_id in enumerate(history):
            if item_id < len(self.sem_ids_list):
                item_token = self._sem_ids_to_tokens(self.sem_ids_list[item_id])
            else:
                item_token = "<UNK>"
            if ADD_PREFIX:
                tokens.append(f"{idx + 1}. {item_token}")
            else:
                tokens.append(item_token)
        return HISTORY_SEP.join(tokens)

    def _get_random_prompt(self, task: str) -> str:
        """Get a random prompt template for the task."""
        templates = ONEREC_PROMPT_TEMPLATES.get(task, ONEREC_PROMPT_TEMPLATES["sidsft"])
        return random.choice(templates)

    def _format_sidsft(self, history: List[int], target: int) -> Dict[str, str]:
        """Format sidsft task."""
        history_str = self._history_to_tokens(history)
        target_tokens = self._sem_ids_to_tokens(
            self.sem_ids_list[target] if target < len(self.sem_ids_list) else [0] * self.n_codebooks
        )
        instruction = self._get_random_prompt("sidsft").format(history=history_str)
        prompt = SFT_PROMPT.format(instruction=instruction)
        return {"prompt": prompt, "response": target_tokens}

    def _format_title2sid(self, item_id: int) -> Dict[str, str]:
        """Format title2sid task."""
        title = self.item_titles.get(item_id, f"item_{item_id}")
        sem_ids = self.sem_ids_list[item_id] if item_id < len(self.sem_ids_list) else [0] * self.n_codebooks
        instruction = self._get_random_prompt("title2sid").format(title=title)
        prompt = SFT_PROMPT.format(instruction=instruction)
        return {"prompt": prompt, "response": self._sem_ids_to_tokens(sem_ids)}

    def _format_sid2title(self, item_id: int) -> Dict[str, str]:
        """Format sid2title task."""
        title = self.item_titles.get(item_id, f"item_{item_id}")
        sem_ids = self.sem_ids_list[item_id] if item_id < len(self.sem_ids_list) else [0] * self.n_codebooks
        index_str = self._sem_ids_to_tokens(sem_ids)
        instruction = self._get_random_prompt("sid2title").format(index=index_str)
        prompt = SFT_PROMPT.format(instruction=instruction)
        return {"prompt": prompt, "response": title}

    def _format_fusionseqrec(self, history: List[int], target: int) -> Dict[str, str]:
        """Format fusionseqrec task (history -> next item title)."""
        history_str = self._history_to_tokens(history)
        target_title = self.item_titles.get(target, f"item_{target}")
        instruction = self._get_random_prompt("fusionseqrec").format(history=history_str)
        prompt = SFT_PROMPT.format(instruction=instruction)
        return {"prompt": prompt, "response": target_title}

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        task = sample['task']

        if task == 'sidsft':
            formatted = self._format_sidsft(sample['history'], sample['target'])
            target = sample['target']
            return {
                'task': task,
                'prompt': formatted['prompt'],
                'response': formatted['response'],
                'target_item': target,
                'target_sem_ids': self.sem_ids_list[target] if target < len(self.sem_ids_list) else [0] * self.n_codebooks,
            }
        elif task == 'title2sid':
            formatted = self._format_title2sid(sample['item_id'])
            item_id = sample['item_id']
            return {
                'task': task,
                'prompt': formatted['prompt'],
                'response': formatted['response'],
                'target_item': item_id,
                'target_sem_ids': self.sem_ids_list[item_id] if item_id < len(self.sem_ids_list) else [0] * self.n_codebooks,
            }
        elif task == 'sid2title':
            formatted = self._format_sid2title(sample['item_id'])
            item_id = sample['item_id']
            return {
                'task': task,
                'prompt': formatted['prompt'],
                'response': formatted['response'],
                'target_item': item_id,
                'target_sem_ids': self.sem_ids_list[item_id] if item_id < len(self.sem_ids_list) else [0] * self.n_codebooks,
            }
        elif task == 'fusionseqrec':
            formatted = self._format_fusionseqrec(sample['history'], sample['target'])
            target = sample['target']
            return {
                'task': task,
                'prompt': formatted['prompt'],
                'response': formatted['response'],
                'target_item': target,
                'target_sem_ids': self.sem_ids_list[target] if target < len(self.sem_ids_list) else [0] * self.n_codebooks,
            }
        else:
            raise ValueError(f"Unknown task: {task}")


@gin.configurable
class AmazonOneRecRLDataset(Dataset):
    """
    Amazon Dataset for OneRec GRPO training.

    Returns prompt-only data (no response) for RL generation.
    3 RL data types:
    - sidsft: SID history -> next SID (main task)
    - title2sid: item title -> SID (alignment task)
    - title_seqrec: title history -> next SID (cross-modal, limited to max_cross_modal samples)
    """

    def __init__(
        self,
        root: str = "dataset/amazon2023",
        split: str = "arts_crafts_and_sewing",
        max_seq_len: int = 20,
        # Tokenizer type: "rqvae" or "rqkmeans"
        tokenizer_type: str = "rqvae",
        # RQ-VAE config (used when tokenizer_type="rqvae")
        pretrained_rqvae_path: str = "./out/onerec/amazon2023/{split}/rqvae/checkpoint.pt",
        encoder_model_name: str = "./models_hub/sentence-t5-xl",
        rqvae_input_dim: int = 768,
        rqvae_embed_dim: int = 64,
        rqvae_hidden_dims: List[int] = [512, 256, 128],
        rqvae_codebook_size: int = 256,
        rqvae_n_layers: int = 3,
        # RQ-KMeans config (used when tokenizer_type="rqkmeans")
        pretrained_rqkmeans_path: str = "",
        rqkmeans_codebook_size: int = 8192,
        rqkmeans_n_layers: int = 3,
        # RL-specific
        max_cross_modal: int = 10000,
        rl_tasks: List[str] = None,
    ) -> None:
        self.root = root
        self.split = split.lower()
        self._max_seq_len = max_seq_len
        self.max_cross_modal = max_cross_modal

        self.rl_tasks = set(rl_tasks or ["sidsft", "title2sid", "title_seqrec"])

        # Load item dataset for embeddings
        item_dataset = AmazonItemDataset(
            root=root,
            split=split,
            train_test_split="all",
            encoder_model_name=encoder_model_name,
        )
        self.item_embeddings = torch.tensor(item_dataset.embeddings, dtype=torch.float32)

        # Generate semantic IDs based on tokenizer type
        if tokenizer_type == "rqkmeans":
            from genrec.models.rqkmeans import RqKmeans
            self.n_codebooks = rqkmeans_n_layers
            self.codebook_size = rqkmeans_codebook_size
            path = pretrained_rqkmeans_path.format(split=self.split)
            rqkmeans = RqKmeans.load(path)
            output = rqkmeans.assign(item_dataset.embeddings.astype('float32'))
            self.sem_ids_list = output.sem_ids.tolist()
        else:
            from genrec.models.rqvae import RqVae
            self.n_codebooks = rqvae_n_layers
            self.codebook_size = rqvae_codebook_size
            pretrained_rqvae_path = pretrained_rqvae_path.format(split=self.split)
            rqvae = RqVae(
                input_dim=rqvae_input_dim,
                embed_dim=rqvae_embed_dim,
                hidden_dims=rqvae_hidden_dims,
                codebook_size=rqvae_codebook_size,
                codebook_kmeans_init=False,
                codebook_normalize=False,
                codebook_sim_vq=False,
                n_layers=rqvae_n_layers,
                n_cat_features=0,
                commitment_weight=0.25,
            )
            rqvae.load_pretrained(pretrained_rqvae_path)
            rqvae.eval()
            with torch.no_grad():
                self.sem_ids_list = rqvae.get_semantic_ids(self.item_embeddings).sem_ids.tolist()

        self._load_item_metadata()
        self._load_sequences()
        self._generate_rl_samples()

    def _load_item_metadata(self) -> None:
        """Load item titles."""
        config = DATASET_CONFIGS[self.split]
        meta_path = os.path.join(self.root, "raw", self.split, config["meta"])
        reviews_path = os.path.join(self.root, "raw", self.split, config["reviews"])

        item_id_mapping: Dict[str, int] = {}
        for review in parse_gzip_json(reviews_path):
            asin = review.get('asin')
            if asin and asin not in item_id_mapping:
                item_id_mapping[asin] = len(item_id_mapping)

        self.item_titles: Dict[int, str] = {}
        for meta in parse_gzip_json(meta_path):
            asin = meta.get('asin')
            if asin in item_id_mapping:
                item_id = item_id_mapping[asin]
                title = meta.get('title', '')
                self.item_titles[item_id] = title.strip() or f"item_{item_id}"

        for i in range(len(item_id_mapping)):
            if i not in self.item_titles:
                self.item_titles[i] = f"item_{i}"

        self.num_items = len(item_id_mapping)

    def _load_sequences(self) -> None:
        """Load user interaction sequences."""
        config = DATASET_CONFIGS[self.split]
        reviews_path = os.path.join(self.root, "raw", self.split, config["reviews"])

        user_sequences: Dict[str, List[tuple]] = {}
        item_id_mapping: Dict[str, int] = {}

        for review in parse_gzip_json(reviews_path):
            asin = review.get('asin')
            user_id = review.get('reviewerID')
            timestamp = review.get('unixReviewTime', 0)

            if asin and user_id:
                if asin not in item_id_mapping:
                    item_id_mapping[asin] = len(item_id_mapping)

                item_id = item_id_mapping[asin]
                if user_id not in user_sequences:
                    user_sequences[user_id] = []
                user_sequences[user_id].append((timestamp, item_id))

        self.sequences = []
        for uid, seq in user_sequences.items():
            seq.sort(key=lambda x: x[0])
            items = [x[1] for x in seq]
            if len(items) >= 5:
                self.sequences.append(items)

        print(f"Loaded {len(self.sequences)} user sequences for OneRec RL")

    def _sem_ids_to_tokens(self, sem_ids: List[int]) -> str:
        return "".join(f"<C{c}_{code}>" for c, code in enumerate(sem_ids))

    def _history_to_tokens(self, history: List[int]) -> str:
        tokens = []
        for idx, item_id in enumerate(history):
            if item_id < len(self.sem_ids_list):
                item_token = self._sem_ids_to_tokens(self.sem_ids_list[item_id])
            else:
                item_token = "<UNK>"
            tokens.append(f"{idx + 1}. {item_token}")
        return HISTORY_SEP.join(tokens)

    def _history_to_titles(self, history: List[int]) -> str:
        """Convert history item IDs to title string."""
        tokens = []
        for idx, item_id in enumerate(history):
            title = self.item_titles.get(item_id, f"item_{item_id}")
            tokens.append(f"{idx + 1}. {title}")
        return HISTORY_SEP.join(tokens)

    def _generate_rl_samples(self) -> None:
        """Generate RL training samples."""
        self.samples = []

        # sidsft: SID history -> next SID
        if "sidsft" in self.rl_tasks:
            for full_seq in self.sequences:
                seq = full_seq[:-2]
                if len(seq) < 2:
                    continue
                # Use last transition for RL (not sliding window)
                history = seq[max(0, len(seq) - 1 - self._max_seq_len):-1]
                target = seq[-1]
                instruction = random.choice(ONEREC_PROMPT_TEMPLATES["sidsft"]).format(
                    history=self._history_to_tokens(history)
                )
                self.samples.append({
                    'task': 'sidsft',
                    'prompt': SFT_PROMPT.format(instruction=instruction),
                    'target_sem_ids': self.sem_ids_list[target] if target < len(self.sem_ids_list) else [0] * self.n_codebooks,
                    'target_item': target,
                })

        # title2sid: item title -> SID
        if "title2sid" in self.rl_tasks:
            for item_id in range(self.num_items):
                if item_id < len(self.sem_ids_list):
                    title = self.item_titles.get(item_id, f"item_{item_id}")
                    instruction = random.choice(ONEREC_PROMPT_TEMPLATES["title2sid"]).format(title=title)
                    self.samples.append({
                        'task': 'title2sid',
                        'prompt': SFT_PROMPT.format(instruction=instruction),
                        'target_sem_ids': self.sem_ids_list[item_id],
                        'target_item': item_id,
                    })

        # title_seqrec: title history -> next SID (cross-modal)
        if "title_seqrec" in self.rl_tasks:
            cross_modal_samples = []
            for full_seq in self.sequences:
                seq = full_seq[:-2]
                if len(seq) < 2:
                    continue
                history = seq[max(0, len(seq) - 1 - self._max_seq_len):-1]
                target = seq[-1]
                history_str = self._history_to_titles(history)
                instruction = f"User has interacted with: {history_str}\nPredict the next item index:"
                cross_modal_samples.append({
                    'task': 'title_seqrec',
                    'prompt': SFT_PROMPT.format(instruction=instruction),
                    'target_sem_ids': self.sem_ids_list[target] if target < len(self.sem_ids_list) else [0] * self.n_codebooks,
                    'target_item': target,
                })
            # Limit cross-modal samples
            if len(cross_modal_samples) > self.max_cross_modal:
                cross_modal_samples = random.sample(cross_modal_samples, self.max_cross_modal)
            self.samples.extend(cross_modal_samples)

        random.shuffle(self.samples)
        print(f"Generated {len(self.samples)} OneRec RL samples")
        task_counts = {}
        for s in self.samples:
            task_counts[s['task']] = task_counts.get(s['task'], 0) + 1
        print(f"RL task distribution: {task_counts}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
