"""
Amazon Dataset for RPG training.

Builds user interaction sequences from Amazon Review 2014 data and generates
semantic IDs using FAISS OPQ tokenization on sentence embeddings.

The OPQ tokenizer quantizes item text embeddings into n_codebook discrete
codes, which serve as the target vocabulary for the RPG model.
"""
import json
import math
import os
import gin
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict, List

from genrec.data.amazon import DATASET_CONFIGS, parse_gzip_json

logger = logging.getLogger(__name__)


def _build_meta_sentence(meta: dict) -> str:
    """Build a text description from item metadata."""
    parts = []
    for key in ["title", "price", "brand", "feature", "categories", "description"]:
        val = meta.get(key)
        if val is None:
            continue
        if isinstance(val, float):
            parts.append(str(val))
        elif isinstance(val, list):
            if len(val) > 0 and isinstance(val[0], list):
                flat = [str(v) for sub in val for v in sub]
                parts.append(", ".join(flat))
            else:
                parts.append(", ".join(str(v) for v in val))
        else:
            parts.append(str(val))
    return " ".join(parts)


def _encode_sentence_embeddings(
    sentences: List[str],
    model_name: str,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode sentences using SentenceTransformer (local model)."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading sentence embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    return embeddings.astype(np.float32)


def _generate_opq_semantic_ids(
    sent_embs: np.ndarray,
    train_mask: np.ndarray,
    n_codebook: int,
    codebook_size: int,
    pca_dim: int = 0,
    use_gpu: bool = False,
    gpu_id: int = 0,
    num_threads: int = 32,
) -> np.ndarray:
    """
    Generate semantic IDs using FAISS OPQ.

    Args:
        sent_embs: [N, D] sentence embeddings.
        train_mask: [N] boolean mask for training items.
        n_codebook: Number of codebooks.
        codebook_size: Size of each codebook (must be power of 2).
        pca_dim: If > 0, apply PCA whitening first.
        use_gpu: Use FAISS GPU.
        gpu_id: GPU device ID.
        num_threads: OMP threads for FAISS.

    Returns:
        pq_codes: [N, n_codebook] integer codes.
    """
    import faiss

    if pca_dim > 0:
        logger.info(f"Applying PCA (dim={pca_dim}) to embeddings...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_dim, whiten=True)
        sent_embs = pca.fit_transform(sent_embs).astype(np.float32)

    n_bits = int(math.log2(codebook_size))
    assert 2 ** n_bits == codebook_size, f"codebook_size must be power of 2, got {codebook_size}"

    index_factory_str = f"OPQ{n_codebook},IVF1,PQ{n_codebook}x{n_bits}"
    logger.info(f"FAISS index factory: {index_factory_str}")

    faiss.omp_set_num_threads(num_threads)
    index = faiss.index_factory(
        sent_embs.shape[1], index_factory_str, faiss.METRIC_INNER_PRODUCT
    )

    if use_gpu:
        res = faiss.StandardGpuResources()
        res.setTempMemory(1024 * 1024 * 512)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = n_codebook >= 56
        index = faiss.index_cpu_to_gpu(res, gpu_id, index, co)

    logger.info("Training FAISS index...")
    index.train(sent_embs[train_mask])
    index.add(sent_embs)

    if use_gpu:
        index = faiss.index_gpu_to_cpu(index)

    # Extract PQ codes
    ivf_index = faiss.downcast_index(index.index)
    invlists = faiss.extract_index_ivf(ivf_index).invlists
    ls = invlists.list_size(0)
    raw_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
    raw_codes = raw_codes.reshape(-1, invlists.code_size)

    # Decode PQ codes from bitstring
    pq_codes = []
    n_bytes = raw_codes.shape[1]
    for u8code in raw_codes:
        bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
        code = [bs.read(n_bits) for _ in range(n_codebook)]
        pq_codes.append(code)

    return np.array(pq_codes)


@gin.configurable
class AmazonRPGDataset(Dataset):
    """
    Amazon Sequence Dataset for RPG training.

    Loads user interaction sequences from Amazon Review 2014 5-core data.
    Generates semantic IDs using OPQ tokenization on text embeddings.

    Training: sliding window with all-position supervision.
    Valid/Test: leave-one-out (predict last item).

    Args:
        root: Root directory for data storage.
        split: Dataset category ("beauty", "sports", "toys", etc.).
        train_test_split: "train", "valid", or "test".
        max_seq_len: Maximum sequence length.
        min_seq_len: Minimum sequence length to keep a user.
        n_codebook: Number of OPQ codebooks.
        codebook_size: Size of each codebook.
        encoder_model_name: SentenceTransformer model name/path.
        sent_emb_pca: PCA dimension (0 = no PCA).
        sent_emb_batch_size: Batch size for sentence encoding.
        opq_use_gpu: Use FAISS GPU for OPQ.
        opq_gpu_id: GPU ID for FAISS.
        faiss_num_threads: Number of OMP threads.
    """

    def __init__(
        self,
        root: str = "dataset/amazon",
        split: str = "beauty",
        train_test_split: str = "train",
        max_seq_len: int = 50,
        min_seq_len: int = 3,
        n_codebook: int = 32,
        codebook_size: int = 256,
        encoder_model_name: str = "sentence-transformers/sentence-t5-xl",
        sent_emb_pca: int = 0,
        sent_emb_batch_size: int = 256,
        opq_use_gpu: bool = False,
        opq_gpu_id: int = 0,
        faiss_num_threads: int = 32,
    ) -> None:
        self.root = root
        self.split = split.lower()
        self.train_test_split = train_test_split
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.n_codebook = n_codebook
        self.codebook_size = codebook_size
        self.encoder_model_name = encoder_model_name
        self.sent_emb_pca = sent_emb_pca
        self.sent_emb_batch_size = sent_emb_batch_size
        self.opq_use_gpu = opq_use_gpu
        self.opq_gpu_id = opq_gpu_id
        self.faiss_num_threads = faiss_num_threads

        if self.split not in DATASET_CONFIGS:
            raise ValueError(f"Unknown split: {split}. Available: {list(DATASET_CONFIGS.keys())}")

        self.raw_dir = os.path.join(root, "raw", self.split)
        self.processed_dir = os.path.join(root, "processed", self.split)
        os.makedirs(self.processed_dir, exist_ok=True)

        # Load user sequences and item metadata
        self._load_sequences()

        # Generate or load semantic IDs
        self._init_semantic_ids()

        # Build item_id2tokens mapping (for model)
        self._build_item_id2tokens()

        # Generate train/valid/test samples
        self._generate_samples()

    def _load_sequences(self) -> None:
        """Load user interaction sequences and item metadata from reviews."""
        config = DATASET_CONFIGS[self.split]
        reviews_path = os.path.join(self.raw_dir, config["reviews"])
        meta_path = os.path.join(self.raw_dir, config["meta"])

        # Build user sequences
        user_sequences: Dict[str, List[tuple]] = {}
        self.item_id_mapping: Dict[str, int] = {}  # asin -> 1-based ID

        logger.info(f"Loading sequences from {reviews_path}...")
        for review in tqdm(parse_gzip_json(reviews_path), desc="Processing reviews"):
            asin = review.get("asin")
            user_id = review.get("reviewerID")
            timestamp = review.get("unixReviewTime", 0)

            if asin and user_id:
                if asin not in self.item_id_mapping:
                    self.item_id_mapping[asin] = len(self.item_id_mapping) + 1  # 1-based
                item_id = self.item_id_mapping[asin]
                if user_id not in user_sequences:
                    user_sequences[user_id] = []
                user_sequences[user_id].append((timestamp, item_id))

        # Sort by timestamp and filter
        self.sequences = []
        for uid, seq in user_sequences.items():
            seq.sort(key=lambda x: x[0])
            items = [x[1] for x in seq]
            if len(items) >= self.min_seq_len:
                self.sequences.append(items)

        self.num_items = len(self.item_id_mapping) + 1  # +1 for padding at index 0
        self.id2asin = {v: k for k, v in self.item_id_mapping.items()}
        logger.info(f"Loaded {len(self.sequences)} sequences, {self.num_items - 1} items")

        # Load metadata
        logger.info(f"Loading metadata from {meta_path}...")
        self.item_meta: Dict[str, str] = {}
        item_asins = set(self.item_id_mapping.keys())
        for meta in tqdm(parse_gzip_json(meta_path), desc="Processing metadata"):
            asin = meta.get("asin")
            if asin in item_asins:
                self.item_meta[asin] = _build_meta_sentence(meta)
        logger.info(f"Loaded metadata for {len(self.item_meta)}/{len(self.item_id_mapping)} items")

    def _init_semantic_ids(self) -> None:
        """Generate or load cached OPQ semantic IDs."""
        encoder_short = os.path.basename(self.encoder_model_name.rstrip("/"))
        n_bits = int(math.log2(self.codebook_size))
        factory_str = f"OPQ{self.n_codebook},IVF1,PQ{self.n_codebook}x{n_bits}"

        sem_ids_path = os.path.join(
            self.processed_dir, f"{encoder_short}_{factory_str}.sem_ids.json"
        )
        sent_emb_path = os.path.join(
            self.processed_dir, f"{encoder_short}.sent_emb.npy"
        )

        if os.path.exists(sem_ids_path):
            logger.info(f"Loading cached semantic IDs from {sem_ids_path}")
            with open(sem_ids_path, "r") as f:
                asin2sem_ids = json.load(f)
            self.asin2sem_ids = {k: tuple(v) for k, v in asin2sem_ids.items()}
            return

        # Encode sentence embeddings
        if os.path.exists(sent_emb_path):
            logger.info(f"Loading cached sentence embeddings from {sent_emb_path}")
            sent_embs = np.load(sent_emb_path)
        else:
            # Build sentences in item ID order (1-based)
            sentences = []
            for item_id in range(1, self.num_items):
                asin = self.id2asin[item_id]
                text = self.item_meta.get(asin, asin)
                sentences.append(text)

            sent_embs = _encode_sentence_embeddings(
                sentences, self.encoder_model_name, self.sent_emb_batch_size
            )
            np.save(sent_emb_path, sent_embs)
            logger.info(f"Saved sentence embeddings to {sent_emb_path}")

        logger.info(f"Sentence embeddings shape: {sent_embs.shape}")

        # Build training mask: items that appear in training sequences
        train_items = set()
        for seq in self.sequences:
            # Training uses seq[:-2]
            for item_id in seq[:-2]:
                train_items.add(item_id)

        train_mask = np.zeros(self.num_items - 1, dtype=bool)
        for item_id in train_items:
            train_mask[item_id - 1] = True
        logger.info(f"Training items: {train_mask.sum()} / {self.num_items - 1}")

        # Generate OPQ codes
        pq_codes = _generate_opq_semantic_ids(
            sent_embs,
            train_mask,
            self.n_codebook,
            self.codebook_size,
            pca_dim=self.sent_emb_pca,
            use_gpu=self.opq_use_gpu,
            gpu_id=self.opq_gpu_id,
            num_threads=self.faiss_num_threads,
        )

        # Map back to asin -> semantic IDs
        self.asin2sem_ids = {}
        for item_id in range(1, self.num_items):
            asin = self.id2asin[item_id]
            self.asin2sem_ids[asin] = tuple(pq_codes[item_id - 1].tolist())

        # Save
        with open(sem_ids_path, "w") as f:
            json.dump({k: list(v) for k, v in self.asin2sem_ids.items()}, f)
        logger.info(f"Saved semantic IDs to {sem_ids_path}")

    def _build_item_id2tokens(self) -> None:
        """Build item_id2tokens tensor: [num_items, n_codebook].

        Token encoding: codebook i, code c -> token_id = i * codebook_size + c + 1
        (0 is reserved for padding)
        """
        self.item_id2tokens = torch.zeros(
            (self.num_items, self.n_codebook), dtype=torch.long
        )
        for asin, sem_ids in self.asin2sem_ids.items():
            if asin not in self.item_id_mapping:
                continue
            item_id = self.item_id_mapping[asin]
            for digit in range(self.n_codebook):
                self.item_id2tokens[item_id, digit] = sem_ids[digit] + digit * self.codebook_size + 1

    def _generate_samples(self) -> None:
        """Generate training/evaluation samples.

        Training: for each user sequence (excluding last 2), use sliding window
                  with all-position supervision (first window) or last-position
                  supervision (subsequent windows).
        Valid: history=seq[:-2], target=seq[-2]
        Test: history=seq[:-1], target=seq[-1]
        """
        self.samples = []

        if self.train_test_split == "train":
            for full_seq in tqdm(self.sequences, desc="Generating train samples"):
                seq = full_seq[:-2]  # Exclude valid/test targets
                if len(seq) < 2:
                    continue

                # First window: all-position supervision
                window = seq[: min(len(seq), self.max_seq_len + 1)]
                self.samples.append({
                    "input_ids": window[:-1],
                    "labels": window[1:],  # All positions are targets
                })

                # Subsequent windows (if seq > max_seq_len + 1)
                for i in range(1, max(len(seq) - self.max_seq_len, 1)):
                    if i >= len(seq) - self.max_seq_len:
                        break
                    window = seq[i : i + self.max_seq_len + 1]
                    input_ids = window[:-1]
                    # Only last position is target
                    labels = [-100] * (len(input_ids) - 1) + [window[-1]]
                    self.samples.append({
                        "input_ids": input_ids,
                        "labels": labels,
                    })

        elif self.train_test_split == "valid":
            for full_seq in self.sequences:
                seq = full_seq[:-1]  # Exclude test target
                if len(seq) < 2:
                    continue
                input_ids = seq[max(0, len(seq) - 1 - self.max_seq_len) : -1]
                target = seq[-1]
                self.samples.append({
                    "input_ids": input_ids,
                    "labels": [target],
                    "seq_len": len(input_ids),
                })

        else:  # test
            for full_seq in self.sequences:
                if len(full_seq) < 2:
                    continue
                input_ids = full_seq[max(0, len(full_seq) - 1 - self.max_seq_len) : -1]
                target = full_seq[-1]
                self.samples.append({
                    "input_ids": input_ids,
                    "labels": [target],
                    "seq_len": len(input_ids),
                })

        logger.info(f"Generated {len(self.samples)} {self.train_test_split} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def rpg_train_collate_fn(batch: List[Dict], max_seq_len: int = 50):
    """
    Collate function for RPG training.

    Pads input_ids and labels to max length in batch.
    Padding: input_ids=0, labels=-100.
    """
    input_ids_list = [b["input_ids"] for b in batch]
    labels_list = [b["labels"] for b in batch]

    max_len = min(max(len(ids) for ids in input_ids_list), max_seq_len)

    padded_input_ids = []
    padded_labels = []
    attention_masks = []
    seq_lens_list = []

    for input_ids, labels in zip(input_ids_list, labels_list):
        # Truncate from left if too long
        if len(input_ids) > max_len:
            input_ids = input_ids[-max_len:]
        if len(labels) > max_len:
            labels = labels[-max_len:]

        seq_len = len(input_ids)
        pad_len = max_len - seq_len

        # Right padding (matching GPT-2 positional encoding)
        padded_input_ids.append(input_ids + [0] * pad_len)
        attention_masks.append([1] * seq_len + [0] * pad_len)

        # Labels: right pad with -100
        label_pad_len = max_len - len(labels)
        padded_labels.append(labels + [-100] * label_pad_len)
        seq_lens_list.append(seq_len)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "seq_lens": torch.tensor(seq_lens_list, dtype=torch.long),
    }


def rpg_eval_collate_fn(batch: List[Dict], max_seq_len: int = 50):
    """
    Collate function for RPG evaluation.

    Returns input_ids, attention_mask, seq_lens, and single-item labels.
    """
    input_ids_list = [b["input_ids"] for b in batch]
    labels_list = [b["labels"] for b in batch]  # Each is [target_item_id]

    max_len = min(max(len(ids) for ids in input_ids_list), max_seq_len)

    padded_input_ids = []
    attention_masks = []
    seq_lens_list = []

    for input_ids in input_ids_list:
        if len(input_ids) > max_len:
            input_ids = input_ids[-max_len:]
        seq_len = len(input_ids)
        pad_len = max_len - seq_len
        # Right padding
        padded_input_ids.append(input_ids + [0] * pad_len)
        attention_masks.append([1] * seq_len + [0] * pad_len)
        seq_lens_list.append(seq_len)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "seq_lens": torch.tensor(seq_lens_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),  # [B, 1]
    }
