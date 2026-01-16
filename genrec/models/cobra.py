"""
COBRA: Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations
https://arxiv.org/pdf/2503.02453
"""

import torch
from typing import NamedTuple
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
    ) -> None:
        super().__init__()
        self.C = n_codebooks
        self.d_model = d_model
        self.pad_id = id_vocab_size * self.C

        # Initialize encoder based on type
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
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, new_feats: torch.Tensor) -> None:
        """
        Enqueue new features, dequeue the oldest ones (circular queue).

        Args:
            new_feats: (n, D)  already L2-normalized features
        """
        n = new_feats.size(0)
        K = self.queue_size
        ptr = int(self.queue_ptr)          

        # case 1: batch size >= queue size
        if n >= K:
            self.feat_queue.copy_(new_feats[-K:])
            self.queue_ptr[0] = 0
            return

        # case 2: no wrap-around, write directly [ptr : ptr+n)
        end_ptr = ptr + n
        if end_ptr <= K:
            self.feat_queue[ptr:end_ptr] = new_feats
        else:
            # case 3: need wrap-around, write in two segments
            first_len = K - ptr                 # how many can be written at the end of the queue
            self.feat_queue[ptr:] = new_feats[:first_len]
            self.feat_queue[:end_ptr - K] = new_feats[first_len:]

        # update pointer (guarantee 0 ≤ queue_ptr < K)
        self.queue_ptr[0] = end_ptr % K


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
        mask=None
    ) -> CobraOutput:
        """
        Forward Pass

        Args:
            input_ids: (B, T*C) - semantic IDs
            encoder_input_ids: (B, T, L) - tokenized text for encoder
        Returns:
            CobraOutput
        """

        vecs = self.encoder(encoder_input_ids)

        B, T = input_ids.shape

        seq_mask = (input_ids != self.pad_id)
        
        seq_mask = self.interleave_seq_mask(seq_mask, self.C)
        # print("seq_mask:", seq_mask)
        # ---------- ① Decoder ----------
        emb = self.cobra_emb(input_ids, vecs, seq_mask)
        
        h = self.decoder(emb, tgt_key_padding_mask=~seq_mask)       # (B,L,D)
        L = h.size(1)

        T = T // self.C
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

            # cross entropy + padding mask
            loss_c = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
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


        # in-batch InfoNCE
        seq_ids_raw = torch.arange(B, device=h.device).unsqueeze(1)  # (B,1)
        seq_ids_raw = seq_ids_raw.expand(-1, T_dense).reshape(-1)    # (Q,)
        seq_ids = seq_ids_raw[valid]                                 # (Q_valid,)
        same_seq = seq_ids.unsqueeze(0) == seq_ids.unsqueeze(1)
        same_seq.fill_diagonal_(False) 
        sim = (vec_pred @ vec_gt.T) / self.temperature
        sim = sim.masked_fill(same_seq, -1e4)
        labels = torch.arange(sim.size(0), device=sim.device)
        loss_dense = F.cross_entropy(sim, labels, reduction="mean")
        # calculate negative number
        dense_loss_neg_number = torch.tensor(((~same_seq).sum().item() - sim.size(0)) / Q, device=sim.device)

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
            usage = torch.stack([F.one_hot(input_ids[:, c::3], self.pad_id + 1).sum((0, 1)).float() for c in range(self.C)])
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
    ) -> CobraGenerationOutput:
        """
        Generate sparse IDs and dense vectors with beam search.
        Aligned with training: c=0 uses dense position, c>0 uses previous codebook position.

        Args:
            input_ids: (B, T*C) - semantic IDs of history
            encoder_input_ids: (B, T, L) - tokenized text for encoder
            n_candidates: number of top candidates to return
            temperature: temperature for softmax

        Returns:
            CobraGenerationOutput with sem_ids, dense_vecs, scores
        """
        B = input_ids.size(0)
        K = n_candidates
        device = input_ids.device
        vocab_size = self.sparse_head[0].out_features

        # Encode text to dense vectors
        vecs = self.encoder(encoder_input_ids)  # (B, T, D)
        T_items = vecs.size(1)  # Number of complete items with dense vectors

        beam_seqs = []  # Will collect generated tokens
        beam_scores = torch.zeros(B, 1, device=device)  # (B, 1) initially
        h_last = None  # Store hidden state from last codebook for dense vec computation

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

                # Last position is the dense position of the last complete item
                seq_lens = seq_mask.sum(dim=1)  # (B,)
                last_dense_pos = seq_lens - 1

                h_c = h[torch.arange(B, device=device), last_dense_pos]  # (B, D)
                logits = self.sparse_head[0](h_c) / temperature  # (B, vocab_size)
                # If only 1 codebook, this is the last position for dense vec
                if self.C == 1:
                    h_last = h_c.unsqueeze(1).expand(-1, K, -1)  # (B, K, D)
                log_probs = F.log_softmax(logits, dim=-1)  # (B, vocab_size)

                topk_scores, topk_ids = log_probs.topk(K, dim=-1)  # (B, K)
                beam_seqs = [topk_ids]  # [(B, K)]
                beam_scores = topk_scores  # (B, K)

            else:
                # For c>0: predict from e_{t+1}^{c-1} (previous codebook position)
                # Append generated tokens and re-run decoder
                current_K = beam_scores.size(1)

                # Expand input_ids for each beam: (B, T*C) -> (B, K, T*C)
                expanded_input_ids = input_ids.unsqueeze(1).expand(-1, current_K, -1)  # (B, K, T*C)

                # Append previously generated tokens (codebooks 0..c-1)
                generated_so_far = torch.stack(beam_seqs, dim=-1)  # (B, K, c)
                new_input_ids = torch.cat([expanded_input_ids, generated_so_far], dim=-1)  # (B, K, T*C + c)

                # Flatten batch and beam dimensions
                flat_input_ids = new_input_ids.view(B * current_K, -1)  # (B*K, T*C + c)

                # Expand vecs for beams
                flat_vecs = vecs.unsqueeze(1).expand(-1, current_K, -1, -1)  # (B, K, T, D)
                flat_vecs = flat_vecs.reshape(B * current_K, T_items, -1)  # (B*K, T, D)

                # n_complete_items is still T_items (the partial tokens are the new item being generated)
                n_complete = T_items

                seq_mask = (flat_input_ids != self.pad_id)
                seq_mask = self.interleave_seq_mask(seq_mask, self.C, n_complete_items=n_complete)
                emb = self.cobra_emb(flat_input_ids, flat_vecs, seq_mask, n_complete_items=n_complete)
                h = self.decoder(emb, tgt_key_padding_mask=~seq_mask)  # (B*K, L, D)

                # Get position of the last token (c-1 th codebook of new item)
                # Structure: [complete items with dense] + [partial new item without dense]
                # Last position is the last generated sparse token
                seq_lens = seq_mask.sum(dim=1)  # (B*K,)
                last_sparse_pos = seq_lens - 1  # (B*K,)

                h_c = h[torch.arange(B * current_K, device=device), last_sparse_pos]  # (B*K, D)
                logits = self.sparse_head[c](h_c) / temperature  # (B*K, vocab_size)
                log_probs = F.log_softmax(logits, dim=-1)  # (B*K, vocab_size)
                log_probs = log_probs.view(B, current_K, vocab_size)  # (B, K, vocab_size)

                # Combine with previous scores
                combined_scores = beam_scores.unsqueeze(-1) + log_probs  # (B, K, vocab_size)
                combined_scores = combined_scores.view(B, -1)  # (B, K * vocab_size)

                topk_scores, topk_indices = combined_scores.topk(K, dim=-1)  # (B, K)

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

                # Store hidden state from last codebook for dense vec computation
                # Must be after beam selection to match final beam order
                if c == self.C - 1:
                    h_c_reshaped = h_c.view(B, current_K, -1)  # (B, K, D)
                    h_last = h_c_reshaped.gather(1, beam_indices.unsqueeze(-1).expand(-1, -1, h_c_reshaped.size(-1)))  # (B, K, D)

        # Stack beam sequences: list of C tensors (B, K) -> (B, K, C)
        sem_ids = torch.stack(beam_seqs, dim=-1)  # (B, K, C)

        # Dense vectors from h_last (last codebook position, aligned with training)
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
