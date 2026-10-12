"""
Amazon COBRA Dataset for COBRA model training.
Provides semantic IDs and tokenized text inputs for the SentenceT5 encoder.
"""
import os
import gin
import torch

from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any

from genrec.data.amazon import (
    AmazonItemDataset,
    DATASET_CONFIGS,
    parse_gzip_json,
    semantic_id_layer_usage,
)
from genrec.data.evaluation_protocol import select_protocol_sequences


@gin.configurable
class AmazonCobraDataset(Dataset):
    """
    Amazon Sequence Dataset for COBRA training.

    Loads user interaction sequences and provides both semantic IDs and
    tokenized text inputs for the SentenceT5 encoder.

    Training uses sliding window strategy:
    - For each user sequence, generate N-1 training samples
    - Each item (except first) serves as a target once

    Valid/Test uses leave-one-out:
    - Valid: history = seq[:-2], target = seq[-2]
    - Test: history = seq[:-1], target = seq[-1]
    """

    def __init__(
        self,
        root: str = "dataset/amazon",
        split: str = "beauty",
        train_test_split: str = "train",
        max_seq_len: int = 20,
        pretrained_rqvae_path: str = "./out/rqvae/amazon/{split}/checkpoint.pt",
        encoder_model_name: str = "./models_hub/sentence-t5-xl",
        max_text_len: int = 128,
        # RQVAE config - should match pretrained model
        rqvae_input_dim: int = 768,
        rqvae_embed_dim: int = 32,
        rqvae_hidden_dims: List[int] = [512, 256, 128, 64],
        rqvae_codebook_size: int = 256,
        rqvae_n_layers: int = 3,
        evaluation_protocol: str = "leave_one_out",
        temporal_train_ratio: float = 0.8,
        temporal_valid_ratio: float = 0.1,
        item_text_format: str = "title_brand",
        rqvae_item_text_format: str = "legacy",
        min_raw_codebook_usage: float = 0.0,
    ) -> None:
        from genrec.models.rqvae import RqVae

        self.root = root
        self.split = split.lower()
        self.train_test_split = train_test_split
        self._max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.n_codebooks = rqvae_n_layers
        self.codebook_size = rqvae_codebook_size
        self.evaluation_protocol = evaluation_protocol
        self.temporal_train_ratio = temporal_train_ratio
        self.temporal_valid_ratio = temporal_valid_ratio
        if item_text_format not in {
            "title_brand", "flattened_attributes", "paper_attributes"
        }:
            raise ValueError(
                "item_text_format must be 'title_brand', "
                "'flattened_attributes' or 'paper_attributes', "
                f"got {item_text_format!r}"
            )
        self.item_text_format = item_text_format
        self.rqvae_item_text_format = rqvae_item_text_format
        if not 0.0 <= min_raw_codebook_usage <= 1.0:
            raise ValueError("min_raw_codebook_usage must be in [0, 1]")

        # Replace {split} placeholder in rqvae path
        pretrained_rqvae_path = pretrained_rqvae_path.format(split=self.split)

        # Load SentenceT5 tokenizer
        st_model = SentenceTransformer(encoder_model_name)
        self.tokenizer = st_model.tokenizer

        # Load item dataset for embeddings
        item_dataset = AmazonItemDataset(
            root=root,
            split=split,
            train_test_split="all",
            encoder_model_name=encoder_model_name,
            item_text_format=rqvae_item_text_format,
        )
        self.item_embeddings = torch.tensor(item_dataset.embeddings, dtype=torch.float32)

        # Load pretrained RQVAE and generate semantic IDs
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
            semantic_ids = rqvae.get_semantic_ids(self.item_embeddings).sem_ids
            layer_usage = semantic_id_layer_usage(
                semantic_ids, rqvae_codebook_size
            )
            if min(layer_usage, default=0.0) < min_raw_codebook_usage:
                raise RuntimeError(
                    "Refusing to build COBRA data from an unhealthy RQ-VAE: "
                    f"layer usage={layer_usage}, required minimum="
                    f"{min_raw_codebook_usage:.2%}."
                )
            self.sem_ids_list = semantic_ids.tolist()
        print(
            "COBRA RQ-VAE raw layer usage: "
            f"{[round(value, 4) for value in layer_usage]}"
        )

        # Load item metadata for text
        self._load_item_metadata()

        # Pre-tokenize all item texts (avoids per-sample tokenization in __getitem__)
        self._precompute_item_tokens()

        # Load user sequences and generate samples
        self._load_sequences()
        self._generate_samples()

    def _format_item_text(self, meta: Dict[str, Any], item_id: int) -> str:
        if self.item_text_format == "paper_attributes":
            text = (
                f"title: {meta.get('title', '')}\n"
                f"price: {meta.get('price', '')}\n"
                f"category: {meta.get('categories', '')}\n"
                f"description: {meta.get('description', '')}"
            )
        elif self.item_text_format == "flattened_attributes":
            text = (
                f"'title':{meta.get('title', '')}\n"
                f" 'price':{meta.get('price', '')}\n"
                f" 'salesRank':{meta.get('salesRank', '')}\n"
                f" 'brand':{meta.get('brand', '')}\n"
                f" 'categories':{meta.get('categories', '')}"
            )
        else:
            text = f"{meta.get('title', '')} {meta.get('brand', '')}".strip()
        return text.strip() or f"item_{item_id}"

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
        self.item_texts: Dict[int, str] = {}
        for meta in parse_gzip_json(meta_path):
            asin = meta.get('asin')
            if asin in item_id_mapping:
                item_id = item_id_mapping[asin]
                # Match the attribute flattening used to construct the
                # Sentence-T5-XL vectors consumed by COBRA's RQ-VAE.  This is
                # also the input representation described by the paper.
                self.item_texts[item_id] = self._format_item_text(meta, item_id)

        # Fill missing items
        for i in range(len(item_id_mapping)):
            if i not in self.item_texts:
                self.item_texts[i] = f"item_{i}"

    def _precompute_item_tokens(self) -> None:
        """Pre-tokenize all item texts to avoid repeated tokenization in __getitem__."""
        n_items = len(self.item_texts)
        print(f"Pre-tokenizing {n_items} item texts...")
        texts = [self.item_texts.get(i, f"item_{i}") for i in range(n_items)]
        # Batch tokenize all at once
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_tensors='pt'
        )
        self.item_token_ids = encoded['input_ids']  # (N, L)
        print(f"Pre-tokenized item tokens shape: {self.item_token_ids.shape}")

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

        selected, self.protocol_info = select_protocol_sequences(
            user_sequences,
            self.train_test_split,
            protocol=self.evaluation_protocol,
            min_seq_len=5,
            temporal_train_ratio=self.temporal_train_ratio,
            temporal_valid_ratio=self.temporal_valid_ratio,
        )
        self.user_ids = [uid for uid, _ in selected]
        self.sequences = [[item_id for _, item_id in seq] for _, seq in selected]

        print(f"Loaded {len(self.sequences)} user sequences for COBRA "
              f"(protocol={self.evaluation_protocol})")

    def _generate_samples(self) -> None:
        """Generate training/evaluation samples.

        COBRA uses teacher forcing, so we use full sequence training:
        - Each user = one sample (no sliding window)
        - Train: history = seq[:-2], target = seq[-2] (leave last for test)
        - Valid: history = seq[:-2], target = seq[-2]
        - Test: history = seq[:-1], target = seq[-1]
        """
        self.samples = []

        if self.train_test_split == "train":
            # Full sequence training: one sample per user
            for user_idx, full_seq in enumerate(self.sequences):
                user_id = hash(self.user_ids[user_idx]) % 10000
                # Leave last 2 items for valid/test, use rest for training
                seq = full_seq[:-2]
                if len(seq) >= 2:  # Need at least 2 items (1 history + 1 target)
                    self.samples.append({
                        'user_id': user_id,
                        'history': seq[:-1],  # All but last as history
                        'target': seq[-1],    # Last as target
                    })
        elif self.train_test_split == "valid":
            for user_idx, full_seq in enumerate(self.sequences):
                user_id = hash(self.user_ids[user_idx]) % 10000
                seq = full_seq[:-1]
                self.samples.append({
                    'user_id': user_id,
                    'history': seq[:-1],
                    'target': seq[-1],
                })
        else:  # test
            for user_idx, full_seq in enumerate(self.sequences):
                user_id = hash(self.user_ids[user_idx]) % 10000
                self.samples.append({
                    'user_id': user_id,
                    'history': full_seq[:-1],
                    'target': full_seq[-1],
                })

        print(f"Generated {len(self.samples)} COBRA samples for {self.train_test_split}")

    def _truncate_history(self, history: List[int]) -> List[int]:
        """Truncate history to max_seq_len (take last N items)."""
        if len(history) > self._max_seq_len:
            return history[-self._max_seq_len:]
        return history

    def _tokenize_items(self, item_ids: List[int]) -> torch.Tensor:
        """Tokenize item texts for SentenceT5 encoder."""
        texts = [self.item_texts.get(i, f"item_{i}") for i in item_ids]
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_tensors='pt'
        )
        return encoded['input_ids']  # (T, L)

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        history_items = self._truncate_history(sample['history'])
        target_item = sample['target']

        # Convert to semantic IDs (flattened)
        item_sem_ids = []
        for item_id in history_items:
            if item_id < len(self.sem_ids_list):
                item_sem_ids.extend(self.sem_ids_list[item_id])
            else:
                item_sem_ids.extend([0] * self.n_codebooks)

        # Look up pre-tokenized texts for history (no tokenization at runtime)
        n_items = self.item_token_ids.shape[0]
        valid_ids = [i if i < n_items else 0 for i in history_items]
        encoder_input_ids = self.item_token_ids[valid_ids]  # (T, L)

        # Target semantic IDs
        target_sem_ids = self.sem_ids_list[target_item] if target_item < len(self.sem_ids_list) else [0] * self.n_codebooks

        # Look up pre-tokenized target text
        target_idx = target_item if target_item < n_items else 0
        target_encoder_input_ids = self.item_token_ids[target_idx:target_idx+1]  # (1, L)

        return {
            'input_ids': item_sem_ids,  # List[int] of length T*C
            'history_item_ids': history_items,
            'encoder_input_ids': encoder_input_ids,  # (T, L)
            'target_sem_ids': target_sem_ids,  # List[int] of length C
            'target_encoder_input_ids': target_encoder_input_ids,  # (1, L)
            'target_item': int(target_item),  # ground-truth item id (for honest item-level eval)
        }


if __name__ == "__main__":
    # Test dataset creation
    dataset = AmazonCobraDataset(
        root="dataset/amazon",
        split="beauty",
        train_test_split="train",
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"First sample keys: {dataset[0].keys()}")
