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
        self.pos_embedding = nn.Embedding(max_pos, embedding_dim)
        self.decoder_pos_embedding = nn.Embedding(sem_id_dim, embedding_dim)

        self.in_proj = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.in_proj_context = nn.Linear(embedding_dim, attn_dim, bias=False)

        self.transformer = TransformerEncoderDecoder(
            d_model=attn_dim,
            nhead=num_heads,
            num_encoder_layers=n_layers // 2,
            num_decoder_layers=n_layers // 2,
            dim_feedforward=1024,
            dropout=dropout,
            norm_cls=RootMeanSquareLayerNorm,
        )
        self.out_proj = nn.Linear(attn_dim, embedding_dim, bias=False)
        # vocab_size = num_item_embeddings * sem_id_dim + 1 (matching embedding layer)
        self.vocab_size = num_item_embeddings * sem_id_dim + 1
        self.output_head = nn.Linear(attn_dim, self.vocab_size, bias=False)

    
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

        pos = torch.arange(N, device=item_emb.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos.long())
        #encoder_input = torch.cat([user_emb, item_emb + pos_emb], dim=1)
        encoder_input = torch.cat([user_emb, item_emb], dim=1)

        if target_input_ids is not None:
            target_emb = self.sem_id_embedding(target_input_ids, target_token_type_ids)
            decoder_pos_emb = self.decoder_pos_embedding(target_token_type_ids)
            #decoder_input = torch.cat([self.bos_embedding.repeat(B, 1, 1), target_emb + decoder_pos_emb], dim=1)
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

        encoder_input = self.in_proj_context(self.drop(self.norm_context(encoder_input)))
        decoder_input = self.in_proj(self.drop(self.norm(decoder_input)))

        # causal mask for decoder 
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            decoder_input.shape[1],
            device=decoder_input.device
        )
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
        encoder_input = torch.cat([user_emb, item_emb], dim=1)

        encoder_mask = torch.cat([
            torch.ones((seq_mask.size(0), 1), dtype=seq_mask.dtype, device=seq_mask.device),
            seq_mask
        ], dim=1)
        f_mask = encoder_mask == 0

        encoder_input = self.in_proj_context(self.drop(self.norm_context(encoder_input)))
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

        decoder_input = self.in_proj(self.drop(self.norm(decoder_input)))

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            decoder_input.shape[1], device=decoder_input.device
        )
        decoder_out = self.transformer.decoder(
            decoder_input,
            memory=memory,
            attn_mask=causal_mask,
            memory_key_padding_mask=memory_mask,
        )
        logits = self.output_head(decoder_out)
        return logits[:, -1, :]

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
        Generate semantic IDs with beam search and encoder caching.

        Args:
            use_trie: If True, use trie constraint for valid item IDs. If False, skip trie (faster).
        """
        # Beam search with encoder caching
        B, K = user_input_ids.size(0), n_top_k_candidates
        device = user_input_ids.device

        # Encode context once
        memory, memory_mask = self._encode_context(
            user_input_ids, item_input_ids, token_type_ids, seq_mask
        )
        # Expand for beam search
        memory = memory.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, memory.size(1), -1)
        memory_mask = memory_mask.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)

        beam_seqs = torch.empty(B, K, 0, dtype=torch.long, device=device)
        beam_logps = torch.zeros(B, K, device=device)

        # Trie (only build if use_trie=True)
        if use_trie:
            if self.trie_root is None:
                self.trie_root = build_trie(valid_item_ids.to("cpu"))
            beam_nodes = [[self.trie_root for _ in range(K)] for _ in range(B)]

        R = 6
        KK = min(K * R, self.num_item_embeddings)

        for step in range(self.sem_id_dim):
            tgt_ids = beam_seqs.view(B * K, -1)
            if tgt_ids.numel() == 0:
                tgt_ids_, tgt_type_ = None, None
            else:
                tgt_ids_ = tgt_ids
                tgt_type_ = torch.arange(tgt_ids.size(1), device=device).unsqueeze(0).expand(B * K, -1)

            logits = self._decode_step(memory, memory_mask, tgt_ids_, tgt_type_)

            # Convert trie's raw token IDs to vocab indices based on current step
            vocab_offset = step * self.num_item_embeddings

            if use_trie:
                # Apply trie constraint
                legal_mask = torch.full_like(logits, False, dtype=torch.bool)
                for b in range(B):
                    for k in range(K):
                        idx = b * K + k
                        valid_set = self.next_valid_tokens(beam_nodes[b][k])
                        if valid_set:
                            valid_vocab_ids = [vocab_offset + tok for tok in valid_set]
                            legal_mask[idx, valid_vocab_ids] = True
                logits = logits.masked_fill(~legal_mask, -1e32)
            else:
                # No trie constraint - only mask to current step's vocab range
                mask = torch.full_like(logits, float('-inf'))
                mask[:, vocab_offset:vocab_offset + self.num_item_embeddings] = 0
                logits = logits + mask

            log_probs = torch.log_softmax(logits / temperature, dim=-1)

            probs = torch.softmax(logits / temperature, dim=-1)
            cand_token = torch.multinomial(probs, num_samples=KK)
            cand_logp = torch.gather(log_probs, 1, cand_token)
            cand_token = cand_token - vocab_offset
            cand_logp = cand_logp.view(B, K, KK)
            cand_token = cand_token.view(B, K, KK)

            total_logp = (beam_logps.unsqueeze(-1) + cand_logp).view(B, -1)
            total_tok = cand_token.view(B, -1)
            total_src = torch.arange(K, device=device).view(1, K, 1).expand(B, K, KK).reshape(B, -1)

            new_seqs = []
            new_scores = []
            if use_trie:
                new_nodes = []

            for b in range(B):
                scores_b, order_b = total_logp[b].sort(descending=True)
                tokens_b = total_tok[b][order_b]
                parent_b = total_src[b][order_b]

                picked_idx = []
                seen = set()

                for j in range(scores_b.size(0)):
                    if len(picked_idx) == K:
                        break
                    p = parent_b[j].item()
                    tid = tokens_b[j].item()

                    seq = torch.cat([
                        beam_seqs[b, p], torch.tensor([tid], device=device)
                    ])
                    key = tuple(seq.tolist())
                    if key in seen:
                        continue
                    seen.add(key)
                    picked_idx.append(j)

                    new_seqs.append(seq)
                    new_scores.append(scores_b[j])
                    if use_trie:
                        parent_node = beam_nodes[b][p]
                        new_nodes.append(parent_node.get(tid, DEAD_NODE))

                while len(picked_idx) < K:
                    new_seqs.append(torch.zeros_like(seq))
                    new_scores.append(torch.tensor(-1e32, device=device))
                    if use_trie:
                        new_nodes.append(self.trie_root)
                    picked_idx.append(-1)

            lens = torch.tensor([s.size(0) for s in new_seqs], device=device)
            max_L = lens.max().item()
            padded = torch.stack([
                torch.nn.functional.pad(s, (0, max_L - s.size(0)))
                for s in new_seqs
            ])

            beam_seqs = padded.view(B, K, max_L)
            beam_logps = torch.stack(new_scores).view(B, K)
            if use_trie:
                beam_nodes = [new_nodes[i * K:(i + 1) * K] for i in range(B)]

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
