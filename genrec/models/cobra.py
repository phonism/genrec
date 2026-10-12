"""
COBRA: Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations
https://arxiv.org/pdf/2503.02453
"""

import torch
from typing import NamedTuple, Optional, Tuple
from torch import nn
import torch.nn.functional as F
from genrec.modules.encoder import LightT5Encoder, SentenceT5Encoder

class CobraOutput(NamedTuple):
    """
    cobra output for training
    """
    loss: torch.Tensor
    loss_sparse: torch.Tensor
    loss_dense: torch.Tensor
    # For epoch accumulation
    acc_correct: torch.Tensor  # Codebook correct count
    acc_total: torch.Tensor  # Codebook total count
    recall_correct: torch.Tensor  # Item correct count (all 3 codebooks)
    recall_total: torch.Tensor  # Item total count
    # Additional metrics
    vec_cos_sim: torch.Tensor
    codebook_entropy: torch.Tensor


class CobraGenerationOutput(NamedTuple):
    """
    cobra output for generation
    """
    sem_ids: torch.Tensor      # (B, K, C) - top-K generated sparse IDs
    dense_vecs: torch.Tensor   # (B, K, D) - corresponding dense vectors
    scores: torch.Tensor       # (B, K) - scores for each candidate


class BeamFusionOutput(NamedTuple):
    """
    BeamFusion output - final recommendation with item IDs
    """
    item_ids: torch.Tensor     # (B, K) - top-K recommended item IDs
    sem_ids: torch.Tensor      # (B, K, C) - corresponding semantic IDs
    scores: torch.Tensor       # (B, K) - fused scores
    

class CobraEmbedding(nn.Module):
    """cobra embedding: e_t^1, e_t^2, ..., e_t^C, v_t
    ---------------------------------------
    ids  : (B, T*C)   # C = n_codebooks
    vecs : (B, T, Dv)
      -> out: (B, (C+1)*T, d_model)
    """
    def __init__(
        self,
        id_vocab_size: int,
        n_codebooks: int = 3,
        d_model: int = 768,
        max_len: int = 1024,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.C = n_codebooks
        self.pad_id = pad_id
        self.id_vocab_size = id_vocab_size
        # for each codebook, build a set of Embedding (parameters are not shared)
        self.id_embed = nn.Embedding(
            id_vocab_size * n_codebooks + 1, d_model, padding_idx=id_vocab_size * n_codebooks
        )
        # token-type: 0…C-1 (sparse), C (dense)
        self.type_embed = nn.Embedding(2, d_model)
        # absolute position: up to max_len
        self.pos_embed  = nn.Embedding(max_len, d_model)

    def forward(self, input_ids: torch.Tensor, input_vecs: torch.Tensor, mask: torch.Tensor, n_complete_items: int = None) -> torch.Tensor:
        """
        Forward Pass. Supports partial sequences for generation.

        Args:
            input_ids  : (B, L) where L = T*C for complete sequence, or T*C + partial for generation
            input_vecs : (B, T, Dv) - dense vectors for T complete items
            mask       : (B, L + T) - interleaved mask
            n_complete_items: number of complete items (with dense vecs). If None, computed as L // C
        Returns:
            (B, L + T, d_model) - embeddings with dense positions
        """
        B, L = input_ids.shape
        device = input_ids.device
        T_vecs = input_vecs.shape[1]  # Number of dense vectors

        if n_complete_items is None:
            n_complete_items = L // self.C

        # 1) Get embeddings for sparse tokens with codebook-specific offsets
        id_token_type_ids = (torch.arange(L, device=device) % self.C).unsqueeze(0).repeat(B, 1)
        emb_mask = (input_ids != self.pad_id)
        emb_input_ids = input_ids.clone()
        emb_input_ids[emb_mask] += id_token_type_ids[emb_mask] * self.id_vocab_size
        id_tok_list = self.id_embed(emb_input_ids)

        # 2) Build output sequence: insert dense vecs after every C sparse tokens (for complete items only)
        # For partial items (generated tokens), no dense vec is inserted
        n_complete_tokens = n_complete_items * self.C
        n_partial_tokens = L - n_complete_tokens

        output_chunks = []

        # Process complete items
        if n_complete_tokens > 0:
            complete_sparse = id_tok_list[:, :n_complete_tokens, :]  # (B, n_complete_tokens, D)
            chunks = complete_sparse.split(self.C, dim=1)  # list of (B, C, D) tensors

            for i, chunk in enumerate(chunks):
                output_chunks.append(chunk)
                if i < T_vecs:
                    output_chunks.append(input_vecs[:, i, :].unsqueeze(1))  # (B, 1, D)

        # Append partial item tokens (no dense vec after them)
        if n_partial_tokens > 0:
            partial_sparse = id_tok_list[:, n_complete_tokens:, :]  # (B, n_partial_tokens, D)
            output_chunks.append(partial_sparse)

        # 4) Concat all chunks
        h = torch.cat(output_chunks, dim=1)  # (B, L + n_complete_items, D)

        # 5) Add position & type embeddings
        out_len = h.shape[1]

        # Position: token-level position (0, 1, 2, 3, 4, 5, ...) for each token
        # Each token gets a unique position index
        pos_idx = torch.arange(out_len, device=device).unsqueeze(0).expand(B, -1)

        # Type: 0 for sparse, 1 for dense
        type_list = []
        for _ in range(n_complete_items):
            type_list.extend([0] * self.C + [1])  # C sparse (0) + 1 dense (1)
        # Partial item types (all sparse)
        if n_partial_tokens > 0:
            type_list.extend([0] * n_partial_tokens)

        type_idx = torch.tensor(type_list[:out_len], device=device).unsqueeze(0).expand(B, -1)

        mask = mask.unsqueeze(-1).float()
        h = h * mask
        h = h + self.pos_embed(pos_idx) * mask
        h = h + self.type_embed(type_idx) * mask
        return h 


class CobraDecoder(nn.Module):
    """
    TransformerDecoder
    """
    def __init__(
        self,
        hidden_dim: int = 768,
        n_layers: int = 6,
        n_heads: int = 12,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate upper-triangular bool mask, shape (L, L)

        Args:
            seq_len: sequence length
            device: device
        Returns:
            (L, L) bool mask
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )

    def forward(
        self,
        tgt: torch.Tensor,                 # (B, L, D)
        memory: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:                     # --> (B, L, D)
        """
        Forward Pass

        Args:
            tgt: (B, L, D)
            memory: (B, L, D)
        Returns:
            (B, L, D)
        """
        L = tgt.size(1)
        causal_mask = self._causal_mask(L, tgt.device)

        # when memory=None, PyTorch still requires a Tensor; use an empty Tensor as placeholder
        if memory is None:
            memory = torch.zeros(
                (tgt.size(0), 0, tgt.size(2)),
                dtype=tgt.dtype,
                device=tgt.device,
            )
        
        out = self.decoder(
            tgt,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return out


class Cobra(torch.nn.Module):
    """
    Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations
    https://arxiv.org/pdf/2503.02453
    """
    def __init__(
        self,
        encoder_n_layers: int = 1,
        encoder_hidden_dim: int = 768,
        encoder_num_heads: int = 8,
        encoder_vocab_size: int = 32128,
        id_vocab_size: int = 512,
        n_codebooks: int = 3,
        d_model: int = 768,
        max_len: int = 1024,
        temperature=0.2,
        queue_size=1024,
        # Decoder params (aligned with TIGER)
        decoder_n_layers: int = 8,
        decoder_num_heads: int = 6,
        decoder_dropout: float = 0.1,
        # Encoder type: "light" (random init) or "pretrained" (sentence-t5)
        encoder_type: str = "light",
        encoder_model_name: str = "./models_hub/sentence-t5-base",
        encoder_pooling: str = "mean",
        dense_negative_scope: str = "cross_sequence",
        use_dense_memory_queue: bool = False,
        # Paper's "w/o Dense" ablation: sparse-only training. Dense vector inputs
        # are replaced with zeros and the InfoNCE loss is skipped; the encoder is
        # kept (frozen) so eval-side BeamFusion still has an item table.
        use_dense: bool = True,
    ) -> None:
        super().__init__()
        self.C = n_codebooks
        self.d_model = d_model
        self.id_vocab_size = id_vocab_size
        self.pad_id = id_vocab_size * self.C
        self.use_dense = use_dense
        if dense_negative_scope not in {"cross_sequence", "all_batch"}:
            raise ValueError(
                "dense_negative_scope must be 'cross_sequence' or 'all_batch', "
                f"got {dense_negative_scope!r}"
            )
        self.dense_negative_scope = dense_negative_scope
        if use_dense_memory_queue and queue_size <= 0:
            raise ValueError("queue_size must be positive when the dense memory queue is enabled")
        self.use_dense_memory_queue = use_dense_memory_queue
        self._prefix_mask_cache_key = None
        self._prefix_mask_cache = None

        # Initialize encoder based on type
        if encoder_type not in {"light", "pretrained"}:
            raise ValueError(
                "encoder_type must be 'light' or 'pretrained', "
                f"got {encoder_type!r}"
            )
        if encoder_type == "pretrained":
            self.encoder = SentenceT5Encoder(
                model_name=encoder_model_name,
                output_dim=d_model,
            )
        else:
            self.encoder = LightT5Encoder(
                n_layers=encoder_n_layers,
                hidden_dim=encoder_hidden_dim,
                output_dim=d_model,
                num_heads=encoder_num_heads,
                vocab_size=encoder_vocab_size,
                pooling=encoder_pooling,
            )
        self.cobra_emb = CobraEmbedding(
            id_vocab_size=id_vocab_size,
            d_model=d_model,
            max_len=max_len,
            pad_id=self.pad_id
        )
        self.decoder = CobraDecoder(d_model, n_layers=decoder_n_layers, n_heads=decoder_num_heads, dropout=decoder_dropout)
        self.sparse_head = nn.ModuleList([
            nn.Linear(d_model, id_vocab_size) for _ in range(n_codebooks)
        ])
        self.temperature = temperature
       
        self.register_buffer(
            "feat_queue",
            torch.randn(queue_size, d_model)
        )
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size
        self.feat_queue = F.normalize(self.feat_queue, dim=-1)
        if use_dense_memory_queue:
            # These buffers are conditional so historical checkpoints (where
            # the registered feature queue was inert) remain strictly loadable
            # when the new research extension is disabled.
            self.register_buffer(
                "item_queue", torch.full((queue_size,), -1, dtype=torch.long)
            )
            self.register_buffer("queue_filled", torch.zeros(1, dtype=torch.long))

    def _encode_item_tokens(
        self,
        encoder_input_ids: torch.Tensor,
        item_mask: Optional[torch.Tensor] = None,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """Encode only real items, leaving padded item slots as exact zeros.

        COBRA batches are padded to the longest history in the batch.  Encoding
        every padded text row is both expensive (Amazon sequences are very
        uneven) and unsafe for encoders whose all-padding attention produces
        NaNs.  The returned tensor keeps the original ``(B, T, D)`` layout so
        the cascaded sparse/dense interleaving remains unchanged.
        """
        if encoder_input_ids.dim() != 3:
            return self.encoder(encoder_input_ids)

        B, T, text_len = encoder_input_ids.shape
        flat_tokens = encoder_input_ids.reshape(B * T, text_len)
        if item_mask is None:
            item_mask = flat_tokens.ne(0).any(dim=-1).view(B, T)
        if item_mask.shape != (B, T):
            raise ValueError(
                f"item_mask must have shape {(B, T)}, got {tuple(item_mask.shape)}"
            )

        valid_indices = item_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            return torch.zeros(
                B, T, self.d_model,
                device=encoder_input_ids.device,
                dtype=self.cobra_emb.id_embed.weight.dtype,
            )

        valid_tokens = flat_tokens.index_select(0, valid_indices)
        encoded_chunks = []
        for start in range(0, valid_tokens.size(0), chunk_size):
            encoded = self.encoder(valid_tokens[start:start + chunk_size].unsqueeze(1))
            encoded_chunks.append(encoded.squeeze(1))
        encoded_valid = torch.cat(encoded_chunks, dim=0)

        flat_vecs = encoded_valid.new_zeros(B * T, encoded_valid.size(-1))
        flat_vecs = flat_vecs.index_copy(0, valid_indices, encoded_valid)
        return flat_vecs.view(B, T, -1)

    def _decode_generated_prefix(
        self,
        input_ids: torch.Tensor,
        vecs: torch.Tensor,
        generated_prefix: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Return the hidden state after a generated target-code prefix.

        ``input_ids`` is right padded.  Appending generated codes to its tensor
        end (the old implementation) placed short samples *after the batch's
        padding gap*, unlike training where the target immediately follows each
        history.  Here every prefix is inserted into that sample's first free
        item slot, preserving the training-time position and causal context.
        """
        B, K, prefix_len = generated_prefix.shape
        if not 1 <= prefix_len <= self.C:
            raise ValueError(
                f"generated prefix length must be in [1, {self.C}], got {prefix_len}"
            )

        # One extra item slot is needed for samples already at the batch maximum.
        extended_ids = F.pad(input_ids, (0, self.C), value=self.pad_id)
        beam_ids = extended_ids.unsqueeze(1).expand(-1, K, -1).clone()
        offsets = torch.arange(prefix_len, device=input_ids.device)
        target_positions = history_lengths[:, None, None] * self.C + offsets
        beam_ids.scatter_(
            2,
            target_positions.expand(-1, K, -1),
            generated_prefix,
        )
        flat_ids = beam_ids.reshape(B * K, -1)

        extended_vecs = F.pad(vecs, (0, 0, 0, 1))
        flat_vecs = (
            extended_vecs.unsqueeze(1)
            .expand(-1, K, -1, -1)
            .reshape(B * K, extended_vecs.size(1), extended_vecs.size(2))
        )
        n_slots = extended_vecs.size(1)
        seq_mask = self.interleave_seq_mask(
            flat_ids.ne(self.pad_id), self.C, n_complete_items=n_slots
        )

        # A complete generated SID must condition the dense prediction, but its
        # (unknown) dense vector must never be exposed as an input token.
        flat_history_lengths = (
            history_lengths[:, None].expand(-1, K).reshape(-1)
        )
        target_dense_positions = flat_history_lengths * (self.C + 1) + self.C
        seq_mask[
            torch.arange(B * K, device=input_ids.device), target_dense_positions
        ] = False

        emb = self.cobra_emb(
            flat_ids, flat_vecs, seq_mask, n_complete_items=n_slots
        )
        hidden = self.decoder(emb, tgt_key_padding_mask=~seq_mask)
        target_sparse_positions = (
            flat_history_lengths * (self.C + 1) + prefix_len - 1
        )
        hidden_at_prefix = hidden[
            torch.arange(B * K, device=input_ids.device), target_sparse_positions
        ]
        return hidden_at_prefix.view(B, K, -1)

    def _get_catalog_prefix_masks(
        self,
        valid_sem_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Build cached masks for catalog-valid autoregressive SID prefixes."""
        if valid_sem_ids.dim() != 2 or valid_sem_ids.size(1) != self.C:
            raise ValueError(
                f"valid_sem_ids must have shape (N, {self.C}), "
                f"got {tuple(valid_sem_ids.shape)}"
            )
        if valid_sem_ids.numel() == 0:
            raise ValueError("valid_sem_ids cannot be empty")
        if valid_sem_ids.min() < 0 or valid_sem_ids.max() >= self.id_vocab_size:
            raise ValueError("valid_sem_ids contains a token outside the codebook")

        cache_key = (
            valid_sem_ids.device,
            valid_sem_ids.data_ptr(),
            tuple(valid_sem_ids.shape),
            valid_sem_ids._version,
        )
        if cache_key == self._prefix_mask_cache_key:
            return self._prefix_mask_cache

        masks = []
        prefix_index = torch.zeros(
            valid_sem_ids.size(0), device=valid_sem_ids.device, dtype=torch.long
        )
        rows = 1
        for codebook in range(self.C):
            mask = torch.zeros(
                rows,
                self.id_vocab_size,
                device=valid_sem_ids.device,
                dtype=torch.bool,
            )
            mask[prefix_index, valid_sem_ids[:, codebook]] = True
            masks.append(mask)
            prefix_index = (
                prefix_index * self.id_vocab_size + valid_sem_ids[:, codebook]
            )
            rows *= self.id_vocab_size

        self._prefix_mask_cache_key = cache_key
        self._prefix_mask_cache = tuple(masks)
        return self._prefix_mask_cache
    
    @torch.no_grad()
    def _dequeue_and_enqueue(
        self,
        new_feats: torch.Tensor,
        new_item_ids: torch.Tensor,
    ) -> None:
        """
        Enqueue new features, dequeue the oldest ones (circular queue).

        Args:
            new_feats: (n, D) already L2-normalized features
            new_item_ids: (n,) catalog item IDs used to mask false negatives
        """
        if not self.use_dense_memory_queue:
            return
        if new_feats.size(0) != new_item_ids.numel():
            raise ValueError("queue features and item IDs must have the same length")

        new_feats = new_feats.detach()
        new_item_ids = new_item_ids.detach().to(self.item_queue.device)
        n = new_feats.size(0)
        K = self.queue_size
        ptr = int(self.queue_ptr)

        # case 1: batch size >= queue size
        if n >= K:
            self.feat_queue.copy_(new_feats[-K:])
            self.item_queue.copy_(new_item_ids[-K:])
            self.queue_ptr[0] = 0
            self.queue_filled[0] = K
            return

        # case 2: no wrap-around, write directly [ptr : ptr+n)
        end_ptr = ptr + n
        if end_ptr <= K:
            self.feat_queue[ptr:end_ptr] = new_feats
            self.item_queue[ptr:end_ptr] = new_item_ids
        else:
            # case 3: need wrap-around, write in two segments
            first_len = K - ptr                 # how many can be written at the end of the queue
            self.feat_queue[ptr:] = new_feats[:first_len]
            self.feat_queue[:end_ptr - K] = new_feats[first_len:]
            self.item_queue[ptr:] = new_item_ids[:first_len]
            self.item_queue[:end_ptr - K] = new_item_ids[first_len:]

        # update pointer (guarantee 0 ≤ queue_ptr < K)
        self.queue_ptr[0] = end_ptr % K
        self.queue_filled[0] = min(K, int(self.queue_filled) + n)

    @staticmethod
    def _mask_duplicate_item_negatives(
        similarities: torch.Tensor,
        target_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Remove repeated catalog items from in-batch negative columns.

        Popular products can occur as targets in several user sequences in the
        same batch.  Their encoded target vector is the same positive, so
        treating the off-diagonal copies as negatives gives contradictory
        gradients and disproportionately hurts popular test items.
        """
        if similarities.ndim != 2 or similarities.shape[0] != similarities.shape[1]:
            raise ValueError("in-batch similarities must be a square matrix")
        if target_item_ids.numel() != similarities.shape[0]:
            raise ValueError("target_item_ids must align with similarity rows")
        duplicate = target_item_ids.reshape(-1, 1).eq(
            target_item_ids.reshape(1, -1)
        )
        duplicate.fill_diagonal_(False)
        return similarities.masked_fill(duplicate, -1e9)


    def interleave_seq_mask(self, seq_mask: torch.Tensor, n: int, n_complete_items: int = None) -> torch.Tensor:
        """
        Interleave sequence mask - insert dense positions after every n sparse positions.
        Supports partial sequences (length not divisible by n) for generation.

        Args:
            seq_mask: (B, L) - mask for sparse tokens
            n: number of codebooks (C)
            n_complete_items: number of complete items (with dense vectors).
                              If None, computed as L // n
        Returns:
            (B, L + n_complete_items) - mask with dense positions inserted
        """
        B, L = seq_mask.shape
        device = seq_mask.device
        dtype = seq_mask.dtype

        if n_complete_items is None:
            n_complete_items = L // n

        # Calculate new positions for original tokens
        # For each token at position i, new position = i + (i // n) for complete items part
        # For partial item tokens (after n_complete_items * n), new position = i + n_complete_items
        orig_pos = torch.arange(L, device=device)
        complete_part = orig_pos < n_complete_items * n
        new_pos = torch.where(
            complete_part,
            orig_pos + orig_pos // n,
            orig_pos + n_complete_items
        )

        # Dense positions to insert (after each complete item)
        g = torch.arange(n_complete_items, device=device)
        ins_pos = g * (n + 1) + n
        prev_idx = g * n + (n - 1)
        prev_idx = prev_idx.clamp(max=L-1)  # Safety clamp
        ins_value = seq_mask[:, prev_idx]

        # New length: original + number of dense positions
        new_len = L + n_complete_items
        out_mask = seq_mask.new_zeros(B, new_len, dtype=dtype)

        out_mask.scatter_(
            dim=1,
            index=new_pos.expand(B, -1),
            src=seq_mask
        )

        if n_complete_items > 0:
            out_mask.scatter_(
                dim=1,
                index=ins_pos.expand(B, -1),
                src=ins_value
            )
        return out_mask
    
    def forward(self,
        input_ids: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        item_ids: Optional[torch.Tensor] = None,
        mask=None
    ) -> CobraOutput:
        """
        Forward Pass

        Args:
            input_ids: (B, T*C) - semantic IDs
            encoder_input_ids: (B, T, L) - tokenized text for encoder
            item_ids: optional (B, T) catalog IDs. Required only when the
                cross-batch dense memory queue is enabled.
        Returns:
            CobraOutput
        """

        item_mask = input_ids[:, ::self.C].ne(self.pad_id)
        # Encode in mini-batches to avoid OOM (B*T items can be very large).
        # Padded item rows are skipped entirely by _encode_item_tokens().
        if not self.use_dense:
            B_enc = encoder_input_ids.shape[0]
            T_enc = encoder_input_ids.shape[1] if encoder_input_ids.dim() == 3 else input_ids.shape[1] // self.C
            vecs = torch.zeros(B_enc, T_enc, self.d_model, device=input_ids.device)
        else:
            vecs = self._encode_item_tokens(encoder_input_ids, item_mask=item_mask)

        B, T = input_ids.shape
        n_items = T // self.C
        if item_ids is not None and item_ids.shape != (B, n_items):
            raise ValueError(
                f"item_ids must have shape {(B, n_items)}, got {tuple(item_ids.shape)}"
            )
        if self.use_dense_memory_queue and item_ids is None:
            raise ValueError("item_ids are required when the dense memory queue is enabled")

        seq_mask = (input_ids != self.pad_id)
        
        seq_mask = self.interleave_seq_mask(seq_mask, self.C)
        # print("seq_mask:", seq_mask)
        # ---------- ① Decoder ----------
        emb = self.cobra_emb(input_ids, vecs, seq_mask)
        
        h = self.decoder(emb, tgt_key_padding_mask=~seq_mask)       # (B,L,D)
        L = h.size(1)

        T = n_items
        n_positions = T - 1  # Number of prediction positions
        # ---------- ② Sparse ID Loss ----------
        loss_sparse = 0.0
        total_correct, total_top5, total_tokens = 0, 0, 0
        # Track all positions: (B, T-1) - whether all codebooks correct at each position
        all_item_correct = torch.ones(B, n_positions, dtype=torch.bool, device=h.device)
        all_valid_mask = None  # Will be set in loop

        for c in range(self.C):
            if c == 0:
                # e_{t+1}^0  ←  v_t
                #  logits come from v_t (last step has no target)
                pos_c = torch.arange(0, T - 1, device=h.device) * (self.C + 1) + self.C
                logits = self.sparse_head[0](h[:, pos_c, :])        # (B, T-1, V)
                target_pos = torch.arange(1, T, device=h.device) * self.C
                target = input_ids[:, target_pos]                              # (B, T-1)
            else:
                # e_{t+1}^c  ←  e_{t+1}^{c-1}   (same item, previous codebook)
                pos_c = torch.arange(1, T, device=h.device) * (self.C + 1) + (c - 1)
                logits = self.sparse_head[c](h[:, pos_c, :])        # (B, T-1, V)
                target_pos = torch.arange(1, T, device=h.device) * self.C + c
                target = input_ids[:, target_pos]                              # (B, T-1)

            # cross entropy + padding mask (force fp32 for numerical stability under AMP)
            loss_c = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=self.pad_id,
                reduction="sum"
            )
            valid_tokens = (target != self.pad_id).sum()
            loss_sparse += loss_c / valid_tokens.clamp(min=1)

            with torch.no_grad():
                valid_mask = target != self.pad_id                  # (B, T-1)
                if all_valid_mask is None:
                    all_valid_mask = valid_mask
                pred_top1 = logits.argmax(-1)
                top1_correct = (pred_top1 == target) & valid_mask
                top5_correct = (logits.topk(5, -1).indices == target.unsqueeze(-1)).any(-1) & valid_mask

                total_correct += top1_correct.sum()
                total_top5 += top5_correct.sum()
                total_tokens += valid_mask.sum()

                # Track all positions: all 3 codebooks must be correct
                all_item_correct &= (pred_top1 == target) | ~valid_mask  # Ignore padded positions

        loss_sparse = loss_sparse / self.C

        # Item-level metrics
        item_correct_masked = all_item_correct & all_valid_mask  # (B, T-1)
        recall = item_correct_masked.float().sum() / all_valid_mask.sum().clamp(min=1)
        correct_count = item_correct_masked.sum()
        total_count = all_valid_mask.sum()
        
        
        # ---------- ③ Dense InfoNCE ----------
        if not self.use_dense:
            zero = torch.zeros((), device=h.device)
            with torch.no_grad():
                usage = torch.stack([F.one_hot(input_ids[:, c::self.C], self.pad_id + 1).sum((0, 1)).float() for c in range(self.C)])
                prob = usage / usage.sum(1, keepdim=True)
                codebook_entropy = -(prob * (prob.add(1e-12).log())).sum(1).mean()
            return CobraOutput(
                loss=loss_sparse,
                loss_sparse=loss_sparse,
                loss_dense=zero,
                acc_correct=total_correct,
                acc_total=total_tokens,
                recall_correct=correct_count,
                recall_total=total_count,
                vec_cos_sim=zero,
                codebook_entropy=codebook_entropy,
            )

        # Predict v_{t+1} from e_{t+1}^{C-1} position (after seeing all sparse IDs of new item)
        # Aligned with paper: predict v_1, v_2, ..., v_{T-1} (T-1 predictions, same as sparse loss)
        vec_pos = torch.arange(1, T, device=h.device) * (self.C + 1) + (self.C - 1)
        vec_pred = h[:, vec_pos, :self.d_model]                     # (B, T-1, D)
        vec_gt = vecs[:, 1:, :].detach()                            # (B, T-1, D)
        

        T_dense = T - 1  # Number of dense predictions (aligned with sparse loss)
        Q = B * T_dense
        # Valid mask: take dense positions starting from item 1 (skip item 0)
        valid = seq_mask[:, (self.C + 1)::(self.C + 1)].reshape(-1)  # Skip first item's dense pos
        vec_pred = vec_pred.reshape(Q, -1)[valid]
        vec_gt = vec_gt.reshape(Q, -1)[valid]
        vec_pred = F.normalize(vec_pred, p=2, dim=-1, eps=1e-12)
        vec_gt = F.normalize(vec_gt, p=2, dim=-1, eps=1e-12)


        # in-batch InfoNCE (force fp32 to avoid bf16 overflow with low temperature)
        vec_pred_f = vec_pred.float()
        vec_gt_f = vec_gt.float()
        seq_ids_raw = torch.arange(B, device=h.device).unsqueeze(1)  # (B,1)
        seq_ids_raw = seq_ids_raw.expand(-1, T_dense).reshape(-1)    # (Q,)
        seq_ids = seq_ids_raw[valid]                                 # (Q_valid,)
        sim = (vec_pred_f @ vec_gt_f.T) / self.temperature
        target_item_ids = None
        if item_ids is not None:
            target_item_ids = item_ids[:, 1:].reshape(Q)[valid]
            sim = self._mask_duplicate_item_negatives(sim, target_item_ids)
        if self.dense_negative_scope == "cross_sequence":
            # Historical GenRec behavior: do not treat other transitions from
            # the same user sequence as negatives.
            same_seq = seq_ids.unsqueeze(0) == seq_ids.unsqueeze(1)
            same_seq.fill_diagonal_(False)
            sim = sim.masked_fill(same_seq, -1e9)
        if self.use_dense_memory_queue:
            filled = int(self.queue_filled)
            if filled > 0:
                # Clone the active slice before the matrix multiply.  The queue
                # is updated later in this forward pass, before backward runs;
                # a detached view would still share the buffer's version
                # counter and autograd would reject the in-place enqueue.
                queued_feats = self.feat_queue[:filled].detach().clone().float()
                queue_sim = (vec_pred_f @ queued_feats.T) / self.temperature
                # Repeated observations of the exact same catalog item are
                # positives, not negatives.  Keep SID collisions as hard
                # negatives because dense disambiguation is COBRA's purpose.
                duplicate_item = target_item_ids.unsqueeze(1).eq(
                    self.item_queue[:filled].unsqueeze(0)
                )
                queue_sim = queue_sim.masked_fill(duplicate_item, -1e9)
                sim = torch.cat([sim, queue_sim], dim=1)
        labels = torch.arange(sim.size(0), device=sim.device)
        loss_dense = F.cross_entropy(sim, labels, reduction="mean")

        if self.use_dense_memory_queue and self.training:
            with torch.no_grad():
                self._dequeue_and_enqueue(vec_gt, target_item_ids)

        """
        # cross-batch InfoNCE
        feat_neg = self.feat_queue.clone().detach()
        logits_pos = (vec_pred * vec_gt).sum(-1, keepdim=True)  # (Q,1)
        logits_neg = vec_pred @ feat_neg.t()             # (Q,K)
        logits = torch.cat([logits_pos, logits_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss_dense = F.cross_entropy(logits, labels, reduction="mean")

        with torch.no_grad():
            self._dequeue_and_enqueue(vec_gt)
        """

        # ---------- ④ Metrics ----------
        vec_cos_sim = F.cosine_similarity(vec_pred, vec_gt).mean()

        # ---------- ⑤ Codebook entropy ----------
        with torch.no_grad():
            usage = torch.stack([F.one_hot(input_ids[:, c::self.C], self.pad_id + 1).sum((0, 1)).float() for c in range(self.C)])
            prob = usage / usage.sum(1, keepdim=True)
            codebook_entropy = -(prob * (prob.add(1e-12).log())).sum(1).mean()

        return CobraOutput(
            loss=loss_sparse + loss_dense,
            loss_sparse=loss_sparse,
            loss_dense=loss_dense,
            acc_correct=total_correct,
            acc_total=total_tokens,
            recall_correct=correct_count,
            recall_total=total_count,
            vec_cos_sim=vec_cos_sim,
            codebook_entropy=codebook_entropy,
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        n_candidates: int = 10,
        temperature: float = 1.0,
        valid_sem_ids: Optional[torch.Tensor] = None,
    ) -> CobraGenerationOutput:
        """
        Generate sparse IDs and dense vectors with beam search.
        Aligned with training: c=0 uses dense position, c>0 uses previous codebook position.

        Args:
            input_ids: (B, T*C) - semantic IDs of history
            encoder_input_ids: (B, T, L) - tokenized text for encoder
            n_candidates: number of top candidates to return
            temperature: temperature for softmax
            valid_sem_ids: optional ``(N, C)`` catalog SID table.  When
                provided, beam expansion is constrained to prefixes that can
                complete to a real catalog item.

        Returns:
            CobraGenerationOutput with sem_ids, dense_vecs, scores
        """
        B = input_ids.size(0)
        if n_candidates <= 0:
            raise ValueError("n_candidates must be positive")
        K = n_candidates
        device = input_ids.device
        vocab_size = self.sparse_head[0].out_features
        if input_ids.size(1) % self.C != 0:
            raise ValueError("input_ids must contain complete history items")

        history_item_mask = input_ids[:, ::self.C].ne(self.pad_id)
        history_lengths = history_item_mask.sum(dim=1)
        if (history_lengths == 0).any():
            raise ValueError("COBRA generation requires at least one history item")
        prefix_masks = None
        if valid_sem_ids is not None:
            prefix_masks = self._get_catalog_prefix_masks(
                valid_sem_ids.to(device=device, dtype=torch.long)
            )

        # Encode only real history items.  This mirrors forward() and avoids
        # spending most of evaluation on padded text rows.
        if not self.use_dense:
            B_enc = encoder_input_ids.shape[0]
            T_enc = encoder_input_ids.shape[1] if encoder_input_ids.dim() == 3 else input_ids.shape[1] // self.C
            vecs = torch.zeros(B_enc, T_enc, self.d_model, device=device)
        else:
            vecs = self._encode_item_tokens(
                encoder_input_ids, item_mask=history_item_mask
            )
        T_items = vecs.size(1)  # Number of complete items with dense vectors

        beam_seqs = []  # Will collect generated tokens
        beam_scores = torch.zeros(B, 1, device=device)  # (B, 1) initially

        # Autoregressive generation for each codebook
        for c in range(self.C):
            if c == 0:
                # For c=0: predict from v_t (last dense position)
                # input_ids has T_items * C tokens (all complete items)
                n_complete = T_items

                seq_mask = (input_ids != self.pad_id)
                seq_mask = self.interleave_seq_mask(seq_mask, self.C, n_complete_items=n_complete)
                emb = self.cobra_emb(input_ids, vecs, seq_mask, n_complete_items=n_complete)
                h = self.decoder(emb, tgt_key_padding_mask=~seq_mask)  # (B, L, D)

                # Last observed dense token, accounting for each sample's true
                # history length rather than the batch maximum.
                last_dense_pos = (
                    (history_lengths - 1) * (self.C + 1) + self.C
                )

                h_c = h[torch.arange(B, device=device), last_dense_pos]  # (B, D)
                logits = self.sparse_head[0](h_c) / temperature  # (B, vocab_size)
                if prefix_masks is not None:
                    logits = logits.masked_fill(
                        ~prefix_masks[0][0].unsqueeze(0),
                        float("-inf"),
                    )
                log_probs = F.log_softmax(logits, dim=-1)  # (B, vocab_size)

                n_valid_first = (
                    int(prefix_masks[0][0].sum().item())
                    if prefix_masks is not None else vocab_size
                )
                first_k = min(K, n_valid_first)
                topk_scores, topk_ids = log_probs.topk(first_k, dim=-1)
                beam_seqs = [topk_ids]  # [(B, K)]
                beam_scores = topk_scores  # (B, K)

            else:
                # For c>0: predict from e_{t+1}^{c-1} (previous codebook position)
                # Insert generated tokens immediately after each sample's real
                # history (never after its right-padding gap) and re-run decoder.
                current_K = beam_scores.size(1)
                generated_so_far = torch.stack(beam_seqs, dim=-1)  # (B, K, c)
                h_c_beams = self._decode_generated_prefix(
                    input_ids=input_ids,
                    vecs=vecs,
                    generated_prefix=generated_so_far,
                    history_lengths=history_lengths,
                )
                h_c = h_c_beams.reshape(B * current_K, -1)
                logits = self.sparse_head[c](h_c) / temperature  # (B*K, vocab_size)
                if prefix_masks is not None:
                    prefix_index = torch.zeros(
                        B, current_K, device=device, dtype=torch.long
                    )
                    for prev_c in range(c):
                        prefix_index = (
                            prefix_index * vocab_size
                            + generated_so_far[:, :, prev_c]
                        )
                    allowed = prefix_masks[c].index_select(
                        0, prefix_index.reshape(-1)
                    )
                    logits = logits.masked_fill(
                        ~allowed, float("-inf")
                    )
                log_probs = F.log_softmax(logits, dim=-1)  # (B*K, vocab_size)
                log_probs = log_probs.view(B, current_K, vocab_size)  # (B, K, vocab_size)

                # Combine with previous scores
                combined_scores = beam_scores.unsqueeze(-1) + log_probs  # (B, K, vocab_size)
                combined_scores = combined_scores.view(B, -1)  # (B, K * vocab_size)

                if prefix_masks is not None:
                    valid_expansions = allowed.view(B, current_K, vocab_size)
                    next_k = min(
                        K,
                        int(valid_expansions.sum((1, 2)).min().item()),
                    )
                else:
                    next_k = min(K, current_K * vocab_size)
                topk_scores, topk_indices = combined_scores.topk(next_k, dim=-1)

                # Decode indices
                beam_indices = topk_indices // vocab_size  # (B, K)
                token_indices = topk_indices % vocab_size  # (B, K)

                # Gather previous sequences for selected beams
                new_beam_seqs = []
                for prev_tokens in beam_seqs:
                    gathered = prev_tokens.gather(1, beam_indices)  # (B, K)
                    new_beam_seqs.append(gathered)
                new_beam_seqs.append(token_indices)

                beam_seqs = new_beam_seqs
                beam_scores = topk_scores

        # Stack beam sequences: list of C tensors (B, K) -> (B, K, C)
        sem_ids = torch.stack(beam_seqs, dim=-1)  # (B, K, C)

        # Dense refinement must run *after* the complete generated SID is fed
        # back into the decoder.  The old code reused the hidden state that
        # predicted the final code, so the dense vector never saw that code.
        h_last = self._decode_generated_prefix(
            input_ids=input_ids,
            vecs=vecs,
            generated_prefix=sem_ids,
            history_lengths=history_lengths,
        )
        dense_vecs = F.normalize(h_last, p=2, dim=-1)  # (B, K, D)

        return CobraGenerationOutput(
            sem_ids=sem_ids,        # (B, K, C)
            dense_vecs=dense_vecs,  # (B, K, D)
            scores=beam_scores,     # (B, K)
        )
    
    def generate_itemvec(self, encoder_input_ids: torch.Tensor):
        """
        generate itememb
        Args:
            encoder_input_ids: (B, T, L) - tokenized text for encoder
        Returns:
            itememb
        """
        vecs = self.encoder(encoder_input_ids)
        vecs = F.normalize(vecs, p=2, dim=-1, eps=1e-12)
        return vecs

    def fuse_generated(
        self,
        gen_output: CobraGenerationOutput,
        item_dense_vecs: torch.Tensor,
        item_sem_ids: torch.Tensor,
        sem_id_to_items: dict,
        n_candidates: int = 10,
        tau: float = 1.0,
        psi: float = 16.0,
    ) -> BeamFusionOutput:
        """Fuse an already-generated sparse/dense beam into item rankings.

        Separating generation from fusion makes validation-only ``tau``/``psi``
        search cheap and exact: every grid point sees the same generated beam,
        while the expensive decoder runs only once.
        """
        device = gen_output.sem_ids.device
        gen_dense_vecs = F.normalize(gen_output.dense_vecs, p=2, dim=-1)
        item_dense_vecs = F.normalize(item_dense_vecs, p=2, dim=-1)
        beam_scores_softmax = torch.softmax(tau * gen_output.scores, dim=-1)

        all_item_ids = []
        all_scores = []
        sem_ids_cpu = gen_output.sem_ids.tolist()
        n_generated_beams = gen_output.sem_ids.size(1)
        neg_inf = float("-inf")

        for b in range(gen_output.sem_ids.size(0)):
            cand_items: list = []
            cand_beam: list = []
            for m in range(n_generated_beams):
                items = sem_id_to_items.get(tuple(sem_ids_cpu[b][m]))
                if items:
                    cand_items.extend(items)
                    cand_beam.extend([m] * len(items))

            if not cand_items:
                all_item_ids.append(torch.full(
                    (n_candidates,), -1, dtype=torch.long, device=device
                ))
                all_scores.append(torch.full(
                    (n_candidates,), neg_inf, device=device
                ))
                continue

            ci = torch.as_tensor(cand_items, device=device, dtype=torch.long)
            cb = torch.as_tensor(cand_beam, device=device, dtype=torch.long)
            sims = (item_dense_vecs[ci] * gen_dense_vecs[b][cb]).sum(-1)

            logits = psi * sims
            seg_max = torch.full(
                (n_generated_beams,), neg_inf, device=device
            ).scatter_reduce(0, cb, logits, reduce="amax", include_self=True)
            exp_logits = torch.exp(logits - seg_max[cb])
            seg_sum = torch.zeros(
                n_generated_beams, device=device
            ).scatter_add(0, cb, exp_logits)
            sim_softmax = exp_logits / seg_sum[cb].clamp_min(1e-12)

            fused = beam_scores_softmax[b][cb] * sim_softmax
            uniq, inv = torch.unique(ci, return_inverse=True)
            uniq_scores = torch.zeros(
                uniq.size(0), device=device
            ).scatter_reduce(0, inv, fused, reduce="amax", include_self=True)
            k = min(n_candidates, uniq.size(0))
            topk_scores, topk_idx = uniq_scores.topk(k)
            topk_items = uniq[topk_idx]
            if k < n_candidates:
                pad = n_candidates - k
                topk_items = torch.cat([
                    topk_items,
                    torch.full((pad,), -1, dtype=torch.long, device=device),
                ])
                topk_scores = torch.cat([
                    topk_scores,
                    torch.full((pad,), neg_inf, device=device),
                ])
            all_item_ids.append(topk_items)
            all_scores.append(topk_scores)

        final_item_ids = torch.stack(all_item_ids, dim=0)
        final_scores = torch.stack(all_scores, dim=0)
        final_sem_ids = torch.full(
            (*final_item_ids.shape, self.C),
            -1,
            dtype=item_sem_ids.dtype,
            device=device,
        )
        valid_items = final_item_ids.ge(0)
        final_sem_ids[valid_items] = item_sem_ids[final_item_ids[valid_items]]
        return BeamFusionOutput(
            item_ids=final_item_ids,
            sem_ids=final_sem_ids,
            scores=final_scores,
        )

    def beam_fusion(
        self,
        input_ids: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        item_dense_vecs: torch.Tensor,
        item_sem_ids: torch.Tensor,
        sem_id_to_items: dict,
        n_candidates: int = 10,
        n_beam: int = 20,
        tau: float = 1.0,
        psi: float = 16.0,
        constrain_to_catalog: bool = False,
    ) -> BeamFusionOutput:
        """
        BeamFusion: Coarse-to-fine generation following COBRA paper.

        Paper formula: Φ(v̂, ID, a) = Softmax(τ·φ_ID) × Softmax(ψ·cos(v̂, a))

        Process:
        1. Beam search generates M sparse IDs with beam scores
        2. For each ID, find candidate items C(ID) that match this semantic ID
        3. Generate dense vector for each ID, compute similarity with candidates
        4. Fuse beam score and similarity score using multiplication

        Args:
            input_ids: (B, T*C) - semantic IDs of history
            encoder_input_ids: (B, T, L) - tokenized text for encoder
            item_dense_vecs: (N, D) - pre-computed dense vectors for all items
            item_sem_ids: (N, C) - semantic IDs for all items
            sem_id_to_items: dict mapping sem_id tuple -> list of item indices
            n_candidates: number of final candidates to return (K)
            n_beam: number of beam search candidates (M)
            tau: coefficient for beam score in BeamFusion
            psi: coefficient for similarity score in BeamFusion
            constrain_to_catalog: prevent beam budget from being spent on SID
                tuples that do not correspond to any catalog item

        Returns:
            BeamFusionOutput with item_ids, sem_ids, scores
        """
        B = input_ids.size(0)
        device = input_ids.device

        # Step 1: Generate M sparse IDs via beam search
        gen_output = self.generate(
            input_ids=input_ids,
            encoder_input_ids=encoder_input_ids,
            n_candidates=n_beam,
            temperature=1.0,
            valid_sem_ids=item_sem_ids if constrain_to_catalog else None,
        )
        return self.fuse_generated(
            gen_output=gen_output,
            item_dense_vecs=item_dense_vecs,
            item_sem_ids=item_sem_ids,
            sem_id_to_items=sem_id_to_items,
            n_candidates=n_candidates,
            tau=tau,
            psi=psi,
        )
    
if __name__ == "__main__":
    cobra = Cobra()
    #cobra.load_state_dict(torch.load("./out/cobra/decoder/cobra_480000.pt")).cuda()
    cobra.cuda()
    cobra.eval()
    input_ids = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 512 * 3, 512 * 3, 512 * 3],
    ]).cuda()
    encoder_input_ids = torch.tensor([
        [
            [2, 3, 4, 0, 0, 0],
            [5, 1, 2, 4, 0, 0],
            [5, 1, 2, 4, 2, 1],
        ],
        [
            [5, 1, 2, 3, 7, 8],
            [5, 6, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ],
    ]).cuda()
    encoder_attention_mask = torch.tensor([
        [[1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1]],
        [[1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]],
    ]).cuda()
    encoder_token_type_ids = torch.tensor([
        [[0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]],
    ]).cuda()
    out = cobra.generate(
        input_ids=input_ids,
        encoder_input_ids=encoder_input_ids,
        encoder_attention_mask=encoder_attention_mask,
        encoder_token_type_ids=encoder_token_type_ids,
    )
