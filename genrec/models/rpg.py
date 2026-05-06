"""
RPG: Recommendation with Parallel Generation (Meta, KDD 2025)

GPT-2 backbone with n_digit parallel ResBlock prediction heads.
Each head predicts one codebook token via contrastive logits against
the corresponding partition of the shared embedding table.

Graph-constrained decoding is used at test time to refine predictions
via item-item similarity propagation.

Reference: https://arxiv.org/abs/2501.12489
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model


class ResBlock(nn.Module):
    """Residual block with zero-initialized linear + SiLU activation."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class RPGModel(nn.Module):
    """
    RPG model: GPT-2 + n_digit parallel ResBlock heads.

    Args:
        n_items: Total number of items (including padding at index 0).
        n_codebook: Number of codebooks (= n_digit).
        codebook_size: Size of each codebook.
        item_id2tokens: Tensor [n_items, n_codebook] mapping item IDs to token IDs.
        n_embd: GPT-2 hidden size.
        n_layer: Number of GPT-2 layers.
        n_head: Number of attention heads.
        n_inner: FFN inner dimension.
        max_seq_len: Maximum input sequence length.
        temperature: Temperature for contrastive logits.
        embd_pdrop: Embedding dropout.
        attn_pdrop: Attention dropout.
        resid_pdrop: Residual dropout.
        num_beams: Number of beams for graph propagation.
        n_edges: Number of edges per node in the adjacency list.
        propagation_steps: Number of graph propagation steps.
        chunk_size: Chunk size for building item-item similarity matrix.
    """

    def __init__(
        self,
        n_items: int,
        n_codebook: int,
        codebook_size: int,
        item_id2tokens: torch.Tensor,
        n_embd: int = 448,
        n_layer: int = 2,
        n_head: int = 4,
        n_inner: int = 1024,
        max_seq_len: int = 50,
        temperature: float = 0.07,
        embd_pdrop: float = 0.5,
        attn_pdrop: float = 0.5,
        resid_pdrop: float = 0.0,
        num_beams: int = 50,
        n_edges: int = 50,
        propagation_steps: int = 3,
        chunk_size: int = 1024,
    ):
        super().__init__()
        self.n_items = n_items
        self.n_codebook = n_codebook
        self.codebook_size = codebook_size
        self.n_embd = n_embd
        self.temperature = temperature
        self.num_beams = num_beams
        self.n_edges = n_edges
        self.propagation_steps = propagation_steps
        self.chunk_size = chunk_size

        # Item ID -> token IDs mapping (register as buffer so it moves with .to())
        self.register_buffer("item_id2tokens", item_id2tokens)

        # Vocab: 0=padding, 1..n_codebook*codebook_size=tokens, last=eos
        vocab_size = n_codebook * codebook_size + 2  # +1 padding, +1 eos

        gpt2config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_seq_len,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function="gelu_new",
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=1e-12,
            initializer_range=0.02,
        )
        self.gpt2 = GPT2Model(gpt2config)

        # Parallel prediction heads (one ResBlock per codebook)
        self.pred_heads = nn.ModuleList([ResBlock(n_embd) for _ in range(n_codebook)])

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        # Graph decoding state
        self.use_graph_decoding = False
        self._graph_initialized = False
        self.adjacency = None

    @property
    def n_parameters_str(self) -> str:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb = sum(p.numel() for p in self.gpt2.wte.parameters() if p.requires_grad)
        return (
            f"Embedding params: {emb:,}\n"
            f"Non-embedding params: {total - emb:,}\n"
            f"Total trainable params: {total:,}"
        )

    def forward(self, input_ids, attention_mask, labels=None, seq_lens=None):
        """
        Forward pass.

        Args:
            input_ids: [B, L] item IDs (0-based, 0=padding).
            attention_mask: [B, L] binary mask.
            labels: [B, L] target item IDs (-100=ignore).
            seq_lens: [B] actual sequence lengths (for generate).

        Returns:
            dict with 'loss' (if labels given) and 'final_states'.
        """
        # Look up token embeddings for each item, average across codebooks
        input_tokens = self.item_id2tokens[input_ids]  # [B, L, n_codebook]
        input_embs = self.gpt2.wte(input_tokens).mean(dim=-2)  # [B, L, n_embd]

        outputs = self.gpt2(inputs_embeds=input_embs, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, L, n_embd]

        # Parallel prediction heads: [B, L, n_codebook, n_embd]
        final_states = torch.stack(
            [self.pred_heads[i](hidden) for i in range(self.n_codebook)], dim=2
        )

        result = {"final_states": final_states}

        if labels is not None:
            label_mask = labels.view(-1) != -100
            selected = final_states.view(-1, self.n_codebook, self.n_embd)[label_mask]
            selected = F.normalize(selected, dim=-1)  # [N, n_codebook, n_embd]

            # Token embeddings: skip padding (idx 0) and eos (last)
            token_emb = self.gpt2.wte.weight[1:-1]  # [n_codebook*codebook_size, n_embd]
            token_emb = F.normalize(token_emb, dim=-1)
            token_embs = torch.chunk(token_emb, self.n_codebook, dim=0)

            token_labels = self.item_id2tokens[labels.view(-1)[label_mask]]  # [N, n_codebook]

            losses = []
            for i in range(self.n_codebook):
                logits_i = torch.matmul(selected[:, i, :], token_embs[i].T) / self.temperature
                target_i = token_labels[:, i] - i * self.codebook_size - 1
                losses.append(self.loss_fct(logits_i, target_i))

            result["loss"] = torch.mean(torch.stack(losses))

        return result

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, seq_lens, n_return=10):
        """
        Generate top-K item predictions.

        Args:
            input_ids: [B, L] item IDs.
            attention_mask: [B, L] mask.
            seq_lens: [B] actual sequence lengths.
            n_return: Number of items to return per sample.

        Returns:
            preds: [B, n_return, 1] predicted item IDs.
        """
        out = self.forward(input_ids, attention_mask, seq_lens=seq_lens)
        final_states = out["final_states"]  # [B, L, n_codebook, n_embd]

        # Gather states at the last valid position
        idx = (seq_lens - 1).view(-1, 1, 1, 1).expand(-1, 1, self.n_codebook, self.n_embd)
        states = final_states.gather(dim=1, index=idx)  # [B, 1, n_codebook, n_embd]
        states = F.normalize(states, dim=-1)

        token_emb = self.gpt2.wte.weight[1:-1]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, self.n_codebook, dim=0)

        logits = [
            torch.matmul(states[:, 0, i, :], token_embs[i].T) / self.temperature
            for i in range(self.n_codebook)
        ]
        logits = [F.log_softmax(l, dim=-1) for l in logits]
        token_logits = torch.cat(logits, dim=-1)  # [B, n_codebook * codebook_size]

        if self.use_graph_decoding:
            if not self._graph_initialized:
                self._init_graph()
            return self._graph_propagation(token_logits, n_return)

        # Direct scoring: for each item, gather its token logits and average
        # item_id2tokens[1:] = all items except padding
        all_tokens = self.item_id2tokens[1:, :]  # [n_items-1, n_codebook]
        item_logits = torch.gather(
            token_logits.unsqueeze(1).expand(-1, self.n_items - 1, -1),
            dim=-1,
            index=(all_tokens - 1).unsqueeze(0).expand(token_logits.shape[0], -1, -1),
        ).mean(dim=-1)  # [B, n_items-1]

        preds = item_logits.topk(n_return, dim=-1).indices + 1  # +1 for 1-based item IDs
        return preds.unsqueeze(-1)

    def _build_ii_sim_mat(self):
        """Build item-item similarity matrix from token embeddings."""
        n_items = self.n_items
        n_codebook = self.n_codebook
        codebook_size = self.codebook_size

        token_embs = self.gpt2.wte.weight[1:-1].view(n_codebook, codebook_size, -1)
        token_embs = F.normalize(token_embs, dim=-1)
        token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))
        token_sims_01 = 0.5 * (token_sims + 1.0)  # [n_codebook, codebook_size, codebook_size]

        device = self.gpt2.device
        item_sim = torch.zeros((n_items, n_items), device=device, dtype=torch.float32)

        for i_start in range(1, n_items, self.chunk_size):
            i_end = min(i_start + self.chunk_size, n_items)
            tokens_i = self.item_id2tokens[i_start:i_end]

            for j_start in range(1, n_items, self.chunk_size):
                j_end = min(j_start + self.chunk_size, n_items)
                tokens_j = self.item_id2tokens[j_start:j_end]

                block = torch.zeros(
                    (i_end - i_start, j_end - j_start), device=device, dtype=torch.float32
                )
                for k in range(n_codebook):
                    row_inds = tokens_i[:, k] - k * codebook_size - 1
                    col_inds = tokens_j[:, k] - k * codebook_size - 1
                    temp = token_sims_01[k].index_select(0, row_inds).index_select(1, col_inds)
                    block += temp

                item_sim[i_start:i_end, j_start:j_end] = block / n_codebook

        return item_sim

    def _init_graph(self):
        """Initialize the decoding graph (adjacency list)."""
        print("Building item-item similarity matrix for graph decoding...")
        sim_mat = self._build_ii_sim_mat()
        self.adjacency = torch.topk(sim_mat, k=self.n_edges, dim=-1).indices
        self._graph_initialized = True
        print("Graph initialized.")

    def _graph_propagation(self, token_logits, n_return):
        """
        Graph-constrained decoding via neighbor propagation.

        Args:
            token_logits: [B, n_codebook * codebook_size] log-softmax scores.
            n_return: Number of items to return.

        Returns:
            preds: [B, n_return, 1] predicted item IDs.
            visited_counts: [B, 1] number of visited nodes.
        """
        B = token_logits.shape[0]
        device = token_logits.device

        visited = {b: set() for b in range(B)}

        # Random initial nodes
        topk_nodes = torch.randint(1, self.n_items, (B, self.num_beams), device=device)
        for b in range(B):
            for node in topk_nodes[b].cpu().tolist():
                visited[b].add(node)

        for _ in range(self.propagation_steps):
            neighbors = self.adjacency[topk_nodes].view(B, -1)
            next_nodes = []
            for b in range(B):
                unique_neighbors = torch.unique(neighbors[b])
                for node in unique_neighbors.cpu().tolist():
                    visited[b].add(node)

                scores = torch.gather(
                    token_logits[b].unsqueeze(0).expand(unique_neighbors.shape[0], -1),
                    dim=-1,
                    index=(self.item_id2tokens[unique_neighbors] - 1),
                ).mean(dim=-1)

                top_idx = torch.topk(scores, min(self.num_beams, len(scores))).indices
                next_nodes.append(unique_neighbors[top_idx])

            # Pad to num_beams if needed
            max_len = max(n.shape[0] for n in next_nodes)
            padded = torch.ones((B, max_len), dtype=torch.long, device=device)
            for b, n in enumerate(next_nodes):
                padded[b, : n.shape[0]] = n
            topk_nodes = padded[:, : self.num_beams]

        visited_counts = torch.tensor(
            [[len(visited[b])] for b in range(B)], dtype=torch.float32
        )
        preds = topk_nodes[:, :n_return].unsqueeze(-1)
        return preds, visited_counts
