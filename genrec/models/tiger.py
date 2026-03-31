"""
TIGER: Recommender Systems with Generative Retrieval.

This module implements the TIGER model for generative retrieval in sequential
recommendation. TIGER uses semantic IDs from RQ-VAE to represent items and
generates next-item predictions through autoregressive decoding.

Key Components:
    - Tiger: Main model with encoder-decoder architecture
    - TrieNode: Trie structure for constrained decoding
    - TigerOutput: Named tuple for model outputs
    - TigerGenerateOutput: Named tuple for generation outputs

Features:
    - Semantic ID embeddings with multi-codebook support
    - Trie-based constrained decoding for valid item generation
    - Beam search with prefix constraints

Reference:
    TIGER: https://arxiv.org/abs/2305.05065
"""

import torch
from torch import nn
import os
import torch.nn.functional as F
from torch.nn import (
    TransformerEncoder, TransformerEncoderLayer,
    TransformerDecoder, TransformerDecoderLayer
)
from safetensors.torch import load_file
from typing import NamedTuple, Optional
from collections import defaultdict
from einops import rearrange

from typing import Dict, Tuple
from genrec.modules.normalize import RMSNorm, RootMeanSquareLayerNorm
from genrec.modules.embedding import SemIdEmbedding, UserIdEmbedding
from genrec.modules.transformer import TransformerEncoderDecoder


class TrieNode(defaultdict):
    """
    Simple trie node, value is still TrieNode, support node[token] cascade creation
    """
    def __init__(self):
        super().__init__(TrieNode)
        self.is_end = False

def build_trie(valid_item_ids: torch.Tensor) -> TrieNode:
    """
    build trie
    Args:
        valid_item_ids: (B, T) or (B, T, C)
    Returns:
        TrieNode
    """
    root = TrieNode()
    if valid_item_ids.dim() == 3:
        flat = valid_item_ids.view(-1, valid_item_ids.size(-1))
    elif valid_item_ids.dim() == 2:
        flat = valid_item_ids
    else:
        flat = valid_item_ids.unsqueeze(0)
    for seq in flat.tolist():
        node = root
        for tok in seq:
            node = node[tok]
        node.is_end = True
    return root

DEAD_NODE = TrieNode()

def build_tensor_trie(
    valid_item_ids: torch.Tensor,
    codebook_size: int,
) -> tuple:
    """
    Convert valid_item_ids into GPU-friendly tensor trie.

    Returns:
        children_mask: (num_nodes, codebook_size) bool — valid child tokens per node
        transition: (num_nodes, codebook_size) int32 — next node ID per (node, token)
    """
    # Build Python trie first (on CPU)
    root = TrieNode()
    if valid_item_ids.dim() == 3:
        flat = valid_item_ids.view(-1, valid_item_ids.size(-1))
    elif valid_item_ids.dim() == 2:
        flat = valid_item_ids
    else:
        flat = valid_item_ids.unsqueeze(0)

    for seq in flat.tolist():
        node = root
        for tok in seq:
            node = node[tok]
        node.is_end = True

    # BFS to assign integer IDs to nodes
    node_list = [root]  # node_list[0] = root
    node_id_map = {id(root): 0}
    queue = [root]

    while queue:
        next_queue = []
        for node in queue:
            for tok, child in sorted(node.items()):
                if id(child) not in node_id_map:
                    node_id_map[id(child)] = len(node_list)
                    node_list.append(child)
                    next_queue.append(child)
        queue = next_queue

    num_nodes = len(node_list)

    # Build tensors
    children_mask = torch.zeros(num_nodes, codebook_size, dtype=torch.bool)
    transition = torch.zeros(num_nodes, codebook_size, dtype=torch.int32)
    # Dead node ID = num_nodes (out of range, won't be indexed for valid tokens)
    dead_id = num_nodes

    for node_idx, node in enumerate(node_list):
        for tok, child in node.items():
            if 0 <= tok < codebook_size:
                children_mask[node_idx, tok] = True
                transition[node_idx, tok] = node_id_map[id(child)]

    # Add dead node row (no valid children)
    children_mask = torch.cat([
        children_mask,
        torch.zeros(1, codebook_size, dtype=torch.bool)
    ], dim=0)
    transition = torch.cat([
        transition,
        torch.full((1, codebook_size), dead_id, dtype=torch.int32)
    ], dim=0)

    return children_mask, transition, num_nodes

class TigerOutput(NamedTuple):
    """
    Tiger output
    """
    logits: torch.Tensor
    loss: torch.Tensor

class TigerGenerationOutput(NamedTuple):
    """
    Tiger generation output
    """
    sem_ids: torch.Tensor
    log_probas: torch.Tensor


class Tiger(nn.Module):
    """
    TIGER: Recommender Systems with Generative Retrieval
    """
    def __init__(
        self,
        embedding_dim: int,
        attn_dim: int,
        dropout: float,
        num_heads: int,
        n_layers: int,
        num_item_embeddings: int,
        num_user_embeddings: int,
        sem_id_dim: int,
        max_pos: int = 2048,
        d_kv: int = 0,
        dim_feedforward: int = 1024,
        share_position_bias: bool = False,
        scale_attn: bool = True,
        add_final_norm: bool = False,
        decoder_bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.trie_root = None
        self.embedding_dim = embedding_dim
        self.attn_dim = attn_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.num_item_embeddings = num_item_embeddings
        self.num_user_embeddings = num_user_embeddings
        self.sem_id_dim = sem_id_dim
        self.max_pos = max_pos
        self.use_proj = (attn_dim != embedding_dim)

        self.bos_embedding = nn.Parameter(torch.randn(embedding_dim))
        self.norm = RMSNorm(embedding_dim)
        self.norm_context = RMSNorm(embedding_dim)
        self.drop = nn.Dropout(p=dropout)
        self.sem_id_embedding = SemIdEmbedding(
            num_embeddings=num_item_embeddings,
            sem_ids_dim=sem_id_dim,
            embeddings_dim=embedding_dim
        )
        self.user_id_embedding = UserIdEmbedding(
            num_embeddings=num_user_embeddings,
            embeddings_dim=embedding_dim
        )

        if self.use_proj:
            self.in_proj = nn.Linear(embedding_dim, attn_dim, bias=False)
            self.in_proj_context = nn.Linear(embedding_dim, attn_dim, bias=False)

        self.transformer = TransformerEncoderDecoder(
            d_model=attn_dim,
            nhead=num_heads,
            num_encoder_layers=n_layers // 2,
            num_decoder_layers=n_layers // 2,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_cls=RootMeanSquareLayerNorm,
            d_kv=d_kv,
            share_position_bias=share_position_bias,
            scale_attn=scale_attn,
            add_final_norm=add_final_norm,
            decoder_bidirectional=decoder_bidirectional,
        )
        # vocab_size = num_item_embeddings * sem_id_dim + 1 (matching embedding layer)
        self.vocab_size = num_item_embeddings * sem_id_dim + 1
        self.output_head = nn.Linear(attn_dim, self.vocab_size, bias=False)

        # Causal mask cache: size -> mask tensor
        self._causal_mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}

        # Tensor trie cache (built lazily)
        self._trie_children_mask: Optional[torch.Tensor] = None
        self._trie_transition: Optional[torch.Tensor] = None
        self._trie_num_nodes: int = 0

    
    def _get_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Get cached causal mask for given size."""
        key = (size, device)
        if key not in self._causal_mask_cache:
            self._causal_mask_cache[key] = nn.Transformer.generate_square_subsequent_mask(
                size, device=device
            )
        return self._causal_mask_cache[key]

    def forward(
        self,
        user_input_ids: torch.Tensor,
        item_input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_token_type_ids: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> TigerOutput:
        """
        Forward pass.
        """
        if seq_mask is None:
            seq_mask = torch.ones_like(item_input_ids, dtype=torch.long, device=item_input_ids.device)
        
        seq_lengths = seq_mask.sum(dim=1)
        user_emb = self.user_id_embedding(user_input_ids)
        item_emb = self.sem_id_embedding(item_input_ids, token_type_ids)
        B, N, D = item_emb.shape

        encoder_input = torch.cat([user_emb, item_emb], dim=1)

        if target_input_ids is not None:
            target_emb = self.sem_id_embedding(target_input_ids, target_token_type_ids)
            decoder_input = torch.cat([self.bos_embedding.repeat(B, 1, 1), target_emb], dim=1)
        else:
            decoder_input = self.bos_embedding.repeat(B, 1, 1)

        encoder_mask = torch.cat([
            torch.ones((seq_mask.size(0), 1), dtype=seq_mask.dtype, device=seq_mask.device),  # user token
            seq_mask
        ], dim=1)
        f_mask = torch.zeros_like(encoder_mask, dtype=torch.float32)
        f_mask[~encoder_mask.bool()] = 1
        f_mask = f_mask.bool()

        encoder_input = self.drop(self.norm_context(encoder_input))
        decoder_input = self.drop(self.norm(decoder_input))
        if self.use_proj:
            encoder_input = self.in_proj_context(encoder_input)
            decoder_input = self.in_proj(decoder_input)

        # causal mask for decoder (cached)
        causal_mask = self._get_causal_mask(decoder_input.shape[1], decoder_input.device)
        decoder_out = self.transformer(
            src=encoder_input,
            tgt=decoder_input,
            tgt_mask=causal_mask,
            src_key_padding_mask=f_mask,
            memory_key_padding_mask=f_mask,
        )
        
        logits = self.output_head(decoder_out)
        loss_logits = logits[:, :-1, :]

        """
        step_logits = []
        for t in range(decoder_out.shape[1]):                  # t = 0 .. T-1
            l = self.output_heads[t](decoder_out[:, t, :])     # (B, V)
            step_logits.append(l.unsqueeze(1))                 # (B, 1, V)

        logits = torch.cat(step_logits, dim=1)[:, :-1, :]
        """
        
        """
        decoder_out = self.out_proj(decoder_out)
        step_logits = []
        for t in range(min(self.sem_id_dim, decoder_out.shape[1])):
            dec_vec = decoder_out[:, t, :]
            start = t * self.num_item_embeddings
            end = (t + 1) * self.num_item_embeddings
            weight_slice = self.sem_id_embedding.emb.weight[start:end]
            logits_t = F.linear(dec_vec, weight_slice)
            step_logits.append(logits_t.unsqueeze(1))
        logits = torch.cat(step_logits, dim=1)
        """

        if target_input_ids is not None and target_input_ids.shape[1] == self.sem_id_dim:
            # Convert to full vocab indices: token_type * num_embeddings + input_id
            target_vocab_ids = target_token_type_ids * self.num_item_embeddings + target_input_ids
            loss = F.cross_entropy(
                loss_logits.reshape(-1, loss_logits.size(-1)),
                target_vocab_ids.reshape(-1),
                reduction="none"
            ).reshape(B, -1)
            loss = loss.sum(dim=1).mean()
        else:
            loss = None
        return TigerOutput(
            logits=logits,
            loss=loss
        )
    
    def load_pretrained(self, path: str):
        """
        Load pretrained model
        """
        state_dict = load_file(os.path.join(path, "model.safetensors"))
        self.load_state_dict(state_dict, strict=True)

    def next_valid_tokens(self, node: TrieNode):
        """
        Return valid tokens
        """
        return list(node.keys())

    def _encode_context(
        self,
        user_input_ids: torch.Tensor,
        item_input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Encode context (user + item history) once for reuse during generation."""
        user_emb = self.user_id_embedding(user_input_ids)
        item_emb = self.sem_id_embedding(item_input_ids, token_type_ids)
        B, N, D = item_emb.shape
        encoder_input = torch.cat([user_emb, item_emb], dim=1)

        encoder_mask = torch.cat([
            torch.ones((seq_mask.size(0), 1), dtype=seq_mask.dtype, device=seq_mask.device),
            seq_mask
        ], dim=1)
        f_mask = encoder_mask == 0

        encoder_input = self.drop(self.norm_context(encoder_input))
        if self.use_proj:
            encoder_input = self.in_proj_context(encoder_input)
        memory = self.transformer.encoder(encoder_input, key_padding_mask=f_mask)
        return memory, f_mask

    def _decode_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt_ids: Optional[torch.Tensor],
        tgt_type: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Single decode step with cached encoder output."""
        B = memory.size(0)
        if tgt_ids is None:
            decoder_input = self.bos_embedding.repeat(B, 1, 1)
        else:
            target_emb = self.sem_id_embedding(tgt_ids, tgt_type)
            decoder_input = torch.cat([self.bos_embedding.repeat(B, 1, 1), target_emb], dim=1)

        decoder_input = self.drop(self.norm(decoder_input))
        if self.use_proj:
            decoder_input = self.in_proj(decoder_input)

        causal_mask = self._get_causal_mask(decoder_input.shape[1], decoder_input.device)
        decoder_out = self.transformer.decoder(
            decoder_input,
            memory=memory,
            attn_mask=causal_mask,
            memory_key_padding_mask=memory_mask,
        )
        logits = self.output_head(decoder_out)
        return logits[:, -1, :]

    def _build_tensor_trie(self, valid_item_ids: torch.Tensor, device: torch.device):
        """Build and cache tensor trie on GPU."""
        if self._trie_children_mask is not None:
            return
        children_mask, transition, num_nodes = build_tensor_trie(
            valid_item_ids, self.num_item_embeddings
        )
        self._trie_children_mask = children_mask.to(device)
        self._trie_transition = transition.to(device).long()
        self._trie_num_nodes = num_nodes
        self._trie_dead_node_id = num_nodes  # last row is dead node

    def generate(
        self,
        user_input_ids: torch.Tensor,
        item_input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        temperature: float = 0.2,
        n_top_k_candidates: int = 10,
        valid_item_ids: Optional[torch.Tensor] = None,
        use_trie: bool = True,
    ) -> "TigerGenerationOutput":
        """
        Generate semantic IDs with beam search, encoder caching, tensor trie,
        and GPU-based beam deduplication.
        """
        B, K = user_input_ids.size(0), n_top_k_candidates
        device = user_input_ids.device
        NIE = self.num_item_embeddings  # codebook_size (e.g. 256)

        # Encode context once
        memory, memory_mask = self._encode_context(
            user_input_ids, item_input_ids, token_type_ids, seq_mask
        )
        # Expand for beam search
        memory = memory.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, memory.size(1), -1)
        memory_mask = memory_mask.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)

        beam_seqs = torch.empty(B, K, 0, dtype=torch.long, device=device)
        beam_logps = torch.zeros(B, K, device=device)

        # Tensor trie: GPU-based constraint
        if use_trie:
            self._build_tensor_trie(valid_item_ids, device)
            # beam_node_ids: (B, K) — current trie node for each beam
            beam_node_ids = torch.zeros(B, K, dtype=torch.long, device=device)  # root=0

        # Sequence encoding for GPU dedup: base-NIE encoding
        # seq_encoded[b,k] = token[0]*NIE^(steps_remaining-1) + ... + token[step]
        beam_seq_encoded = torch.zeros(B, K, dtype=torch.long, device=device)

        R = 6
        KK = min(K * R, NIE)

        for step in range(self.sem_id_dim):
            tgt_ids = beam_seqs.view(B * K, -1)
            if tgt_ids.numel() == 0:
                tgt_ids_, tgt_type_ = None, None
            else:
                tgt_ids_ = tgt_ids
                tgt_type_ = torch.arange(tgt_ids.size(1), device=device).unsqueeze(0).expand(B * K, -1)

            logits = self._decode_step(memory, memory_mask, tgt_ids_, tgt_type_)

            vocab_offset = step * NIE

            if use_trie:
                # Tensor trie constraint — single GPU indexing, no Python loops
                flat_node_ids = beam_node_ids.view(B * K)  # (B*K,)
                valid_mask = self._trie_children_mask[flat_node_ids]  # (B*K, codebook_size)
                legal_mask = torch.zeros(B * K, logits.size(-1), dtype=torch.bool, device=device)
                legal_mask[:, vocab_offset:vocab_offset + NIE] = valid_mask
                logits = logits.masked_fill(~legal_mask, -1e32)
            else:
                mask = torch.full_like(logits, float('-inf'))
                mask[:, vocab_offset:vocab_offset + NIE] = 0
                logits = logits + mask

            log_probs = torch.log_softmax(logits / temperature, dim=-1)
            cand_logp, cand_token = torch.topk(log_probs, k=KK, dim=-1)
            cand_token = cand_token - vocab_offset  # now in [0, NIE)
            cand_logp = cand_logp.view(B, K, KK)
            cand_token = cand_token.view(B, K, KK)

            total_logp = (beam_logps.unsqueeze(-1) + cand_logp).view(B, -1)  # (B, K*KK)
            total_tok = cand_token.view(B, -1)  # (B, K*KK)
            total_src = torch.arange(K, device=device).view(1, K, 1).expand(B, K, KK).reshape(B, -1)

            # GPU-based beam dedup using encoded sequences
            # Encode candidate sequences: parent_enc * NIE + new_token
            parent_enc = beam_seq_encoded.unsqueeze(-1).expand(B, K, KK).reshape(B, -1)  # (B, K*KK)
            cand_enc = parent_enc * NIE + total_tok  # (B, K*KK)

            # Sort by score descending
            sorted_scores, sort_idx = total_logp.sort(descending=True)  # (B, K*KK)
            sorted_tok = torch.gather(total_tok, 1, sort_idx)
            sorted_src = torch.gather(total_src, 1, sort_idx)
            sorted_enc = torch.gather(cand_enc, 1, sort_idx)

            # Dedup: for each batch, pick top-K unique encoded sequences
            # Use vectorized approach: mark first occurrence of each unique encoding
            new_beam_seqs = torch.zeros(B, K, step + 1, dtype=torch.long, device=device)
            new_beam_logps = torch.full((B, K), -1e32, device=device)
            new_beam_node_ids = torch.zeros(B, K, dtype=torch.long, device=device) if use_trie else None
            new_beam_seq_encoded = torch.zeros(B, K, dtype=torch.long, device=device)

            for b in range(B):
                # Batch GPU→CPU transfer: 3 .tolist() calls instead of ~5400 .item() calls
                enc_list = sorted_enc[b].tolist()
                tok_list = sorted_tok[b].tolist()
                src_list = sorted_src[b].tolist()

                seen = set()
                picks = []
                for j in range(len(enc_list)):
                    if len(picks) >= K:
                        break
                    if enc_list[j] not in seen:
                        seen.add(enc_list[j])
                        picks.append(j)

                n_picks = len(picks)
                if n_picks == 0:
                    continue

                pick_idx = torch.tensor(picks, dtype=torch.long, device=device)
                parent_beams = sorted_src[b][pick_idx]

                if step > 0:
                    new_beam_seqs[b, :n_picks, :step] = beam_seqs[b][parent_beams]
                new_beam_seqs[b, :n_picks, step] = sorted_tok[b][pick_idx]
                new_beam_logps[b, :n_picks] = sorted_scores[b][pick_idx]
                new_beam_seq_encoded[b, :n_picks] = sorted_enc[b][pick_idx]
                if use_trie:
                    parent_nids = beam_node_ids[b][parent_beams]
                    child_toks = sorted_tok[b][pick_idx]
                    new_beam_node_ids[b, :n_picks] = self._trie_transition[parent_nids, child_toks]

            beam_seqs = new_beam_seqs
            beam_logps = new_beam_logps
            beam_seq_encoded = new_beam_seq_encoded
            if use_trie:
                beam_node_ids = new_beam_node_ids

        return TigerGenerationOutput(
            sem_ids=beam_seqs,
            log_probas=beam_logps,
        )

if __name__ == "__main__":
    torch.manual_seed(42)
    model = Tiger(
        embedding_dim=128,
        attn_dim=512,
        dropout=0.3,
        num_heads=8,
        n_layers=8,
        num_item_embeddings=256,
        num_user_embeddings=2000,
        sem_id_dim=3,
        max_pos=512 * 3
    )
    model.cuda()
    model.eval()
    user_input_ids = torch.tensor([[1], [2]]).cuda()
    
    item_input_ids = torch.tensor([
        [43, 38, 217, 62, 183, 153, 72, 119, 121, 230, 237, 113, 3, 40, 41, 43, 52, 180, 768, 768, 768],
        [75, 40, 33, 69, 69, 226, 3, 89, 210, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768]]).cuda()
    token_type_ids = torch.tensor([
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0],
        [ 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).cuda()
    target_input_ids = torch.tensor([[142, 39, 121],
        [194, 17, 237]]).cuda()
    target_token_type_ids = torch.tensor([[0, 1, 2], [0, 1, 2]]).cuda()
    seq_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).cuda()
    with torch.no_grad():
        out = model(
            user_input_ids=user_input_ids,
            item_input_ids=item_input_ids,
            token_type_ids=token_type_ids,
            target_input_ids=target_input_ids,
            target_token_type_ids=target_token_type_ids,
            seq_mask=seq_mask,
        )

    item_input_ids = torch.tensor([
        [43, 38, 217, 62, 183, 153, 72, 119, 121, 230, 237, 113, 3, 40, 41, 43, 52, 180],
        [75, 40, 33, 69, 69, 226, 3, 89, 210, 768, 768, 768, 768, 768, 768, 768, 768, 768]]).cuda()
    token_type_ids = torch.tensor(
        [[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        [ 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    ).cuda()
    target_input_ids = torch.tensor(
        [[142, 39, 121],
        [194, 17, 237]]
    ).cuda()
    target_token_type_ids = torch.tensor([[0, 1, 2], [0, 1, 2]]).cuda()
    seq_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).cuda()
    with torch.no_grad():
        out = model(
            user_input_ids=user_input_ids,
            item_input_ids=item_input_ids,
            token_type_ids=token_type_ids,
            target_input_ids=target_input_ids,
            target_token_type_ids=target_token_type_ids,
            seq_mask=seq_mask,
        )

    print("===================")

    valid_item_ids = torch.tensor([
        [0, 0, 0],
        [1, 1, 1]]    
    )   
    
    valid_item_ids = []
    for i in range(50):
        for j in range(56):
            for k in range(56):
                valid_item_ids.append([i, j, k])
    valid_item_ids = torch.tensor(valid_item_ids)
    #model.load_pretrained("./out/tiger/amazon_electronics/tiger_final.pt")
    with torch.inference_mode():
        generated = model.generate(
            user_input_ids=user_input_ids,
            item_input_ids=item_input_ids,
            token_type_ids=token_type_ids,
            temperature=1,
            n_top_k_candidates=10,
            valid_item_ids=valid_item_ids,
            seq_mask=seq_mask,
        )
    print(item_input_ids)
    print(generated)


    item_input_ids = torch.tensor([
        [43, 38, 217, 62, 183, 153, 72, 119, 121, 230, 237, 113, 3, 40, 41, 43, 52, 180],
        [75, 40, 33, 69, 69, 226, 3, 89, 210, 768, 768, 768, 768, 768, 768, 768, 768, 768]]).cuda()
    token_type_ids = torch.tensor(
        [[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    ).cuda()
    target_input_ids = torch.tensor(
        [[142, 39, 121],
        [194, 17, 237]]
    ).cuda()
    target_token_type_ids = torch.tensor([[0, 1, 2], [0, 1, 2]]).cuda()
    seq_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).cuda()
    with torch.inference_mode():
        generated = model.generate(
            user_input_ids=user_input_ids,
            item_input_ids=item_input_ids,
            token_type_ids=token_type_ids,
            temperature=1,
            n_top_k_candidates=10,
            valid_item_ids=valid_item_ids,
            seq_mask=seq_mask,
        )
    print(item_input_ids)
    print(generated)
