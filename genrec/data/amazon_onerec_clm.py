"""
Amazon OneRec CLM Dataset - Causal Language Modeling format (OpenOneRec-style).

Instead of generating N sliding-window instruction-response samples per user,
we create ONE flat token stream per user sequence and train with standard
causal LM loss at every position.

Format: C0_a C1_b C2_c <|item_sep|> C0_d C1_e C2_f <|item_sep|> ... C0_x C1_y C2_z
"""
import os
import gin
import torch

from torch.utils.data import Dataset
from typing import Dict, List, Any

from genrec.data.amazon import (
    AmazonItemDataset,
    DATASET_CONFIGS,
    parse_gzip_json,
)
from genrec.data.evaluation_protocol import select_protocol_sequences


@gin.configurable
class AmazonOneRecCLMDataset(Dataset):
    """
    Amazon Dataset for OneRec CLM training.

    Training: one sample per user = flat codebook token stream.
    Eval: history token stream + target sem_ids for beam search.
    """

    def __init__(
        self,
        root: str = "dataset/amazon",
        split: str = "beauty",
        train_test_split: str = "train",
        max_seq_len: int = 50,
        # Tokenizer type: "rqvae" or "rqkmeans"
        tokenizer_type: str = "rqvae",
        # RQ-VAE config
        pretrained_rqvae_path: str = "",
        encoder_model_name: str = "./models_hub/sentence-t5-xl",
        rqvae_input_dim: int = 768,
        rqvae_embed_dim: int = 64,
        rqvae_hidden_dims: List[int] = [512, 256, 128],
        rqvae_codebook_size: int = 256,
        rqvae_n_layers: int = 3,
        # RQ-KMeans config
        pretrained_rqkmeans_path: str = "",
        rqkmeans_codebook_size: int = 8192,
        rqkmeans_n_layers: int = 3,
        # CLM-specific
        min_seq_len: int = 3,
        evaluation_protocol: str = "leave_one_out",
        temporal_train_ratio: float = 0.8,
        temporal_valid_ratio: float = 0.1,
    ) -> None:
        self.root = root
        self.split = split.lower()
        self.train_test_split = train_test_split
        self._max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.evaluation_protocol = evaluation_protocol
        self.temporal_train_ratio = temporal_train_ratio
        self.temporal_valid_ratio = temporal_valid_ratio

        # Load item dataset for embeddings
        item_dataset = AmazonItemDataset(
            root=root,
            split=split,
            train_test_split="all",
            encoder_model_name=encoder_model_name,
        )
        self.item_embeddings = torch.tensor(item_dataset.embeddings, dtype=torch.float32)

        # Generate semantic IDs
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

        self._load_sequences()
        self._generate_samples()

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
        self.sequences = [[item_id for _, item_id in seq] for _, seq in selected]

        self.num_items = len(item_id_mapping)
        print(f"Loaded {len(self.sequences)} user sequences for OneRec CLM "
              f"(protocol={self.evaluation_protocol})")

    def _generate_samples(self) -> None:
        """Generate samples: one per user.

        Histories are truncated to the last `max_seq_len` ITEMS here — the
        benchmark protocol is item-level, and truncating at the item level keeps
        the token stream aligned to item boundaries (the collate's token-level
        `[-max_length:]` slice could otherwise cut an item's codes in half).
        """
        self.samples = []
        keep = self._max_seq_len

        if self.train_test_split == "train":
            for full_seq in self.sequences:
                seq = full_seq[:-2][-keep:]  # Leave last 2 for valid/test
                if len(seq) >= self.min_seq_len:
                    self.samples.append({'item_seq': seq, 'task': 'clm'})
        elif self.train_test_split == "valid":
            for full_seq in self.sequences:
                seq = full_seq[:-1]
                if len(seq) >= 2:
                    history = seq[:-1][-keep:]
                    target = seq[-1]
                    self.samples.append({
                        'item_seq': history,
                        'target': target,
                        'target_sem_ids': self.sem_ids_list[target] if target < len(self.sem_ids_list) else [0] * self.n_codebooks,
                        'task': 'seqrec',
                    })
        else:  # test
            for full_seq in self.sequences:
                if len(full_seq) >= 2:
                    history = full_seq[:-1][-keep:]
                    target = full_seq[-1]
                    self.samples.append({
                        'item_seq': history,
                        'target': target,
                        'target_sem_ids': self.sem_ids_list[target] if target < len(self.sem_ids_list) else [0] * self.n_codebooks,
                        'task': 'seqrec',
                    })

        print(f"Generated {len(self.samples)} OneRec CLM samples for {self.train_test_split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
