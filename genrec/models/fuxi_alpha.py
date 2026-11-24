"""Padded PyTorch FuXi-alpha for GenRec sequence training.

Mathematical implementation of FuXi-alpha (arXiv:2502.03036): separate
position/time/content attention channels and a multistage gated FFN.
The public USTC-StarTeam/FuXi-alpha revision 69b46a9ddff8f6fa40c057a02b9f86e7bf38983e
is an external numerical reference, not a runtime dependency. No jagged
extension, upstream trainer or upstream source distribution is required.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _rms(x, eps):
    # Use the same native operator when available; older Torch has this fallback.
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.shape[-1],), eps=eps)
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


class FuXiRelativeBias(nn.Module):
    def __init__(self, length, num_buckets=128):
        super().__init__()
        self.length = length
        self.num_buckets = num_buckets
        self._ts_w = nn.Parameter(torch.empty(num_buckets + 1).normal_(0, 0.02))
        self._pos_w = nn.Parameter(torch.empty(2 * length - 1).normal_(0, 0.02))

    def forward(self, timestamps):
        n = self.length
        index = torch.arange(n, device=timestamps.device)
        position = self._pos_w[n - 1 + index[None, :] - index[:, None]][None]
        next_times = torch.cat([timestamps[:, 1:], timestamps[:, -1:]], dim=1)
        delta = next_times[:, :, None] - timestamps[:, None, :]
        bucket = (torch.log(delta.abs().clamp(min=1)) / 0.301).long().clamp(0, self.num_buckets)
        return position, self._ts_w[bucket]


class FuXiMultistageFFN(nn.Module):
    def __init__(self, channels, width, hidden, dropout, single_stage=False, eps=1e-6):
        super().__init__()
        self.lin0 = nn.Linear(channels, width)
        self.single_stage, self.dropout, self.eps = single_stage, dropout, eps
        if not single_stage:
            self.lin1 = nn.Linear(width, hidden, bias=False)
            self.lin2 = nn.Linear(hidden, width, bias=False)
            self.lin3 = nn.Linear(width, hidden, bias=False)
        for name in ("lin0", "lin1", "lin2", "lin3"):
            if hasattr(self, name):
                nn.init.xavier_uniform_(getattr(self, name).weight)

    def forward(self, channels, residual):
        x = residual + self.lin0(F.dropout(channels, self.dropout, self.training))
        if not self.single_stage:
            z = F.dropout(_rms(x, self.eps), self.dropout, self.training)
            x = x + self.lin2(F.silu(self.lin1(z)) * self.lin3(z))
        return x


class FuXiAlphaBlock(nn.Module):
    def __init__(self, width, max_length, num_heads=1, dqk=None, dv=None,
                 dropout=0.2, ffn_multiply=1, single_stage=False, eps=1e-6):
        super().__init__()
        self.width, self.length, self.heads = width, max_length, num_heads
        self.dqk, self.dv = dqk or width, dv or width
        self.eps = eps
        self._rel_attn_bias = FuXiRelativeBias(max_length)
        self._uvqk = nn.Parameter(torch.empty(width, num_heads * (4 * self.dv + 2 * self.dqk)).normal_(0, 0.02))
        self._mffn = FuXiMultistageFFN(
            3 * num_heads * self.dv, width, int(width * ffn_multiply), dropout, single_stage, eps,
        )

    def forward(self, x, timestamps, valid_mask):
        b, n, _ = x.shape
        if n != self.length or timestamps.shape != (b, n):
            raise ValueError("FuXi requires its fixed padded length and matching timestamps")
        z = F.silu(_rms(x, self.eps) @ self._uvqk)
        h, vdim, qdim = self.heads, self.dv, self.dqk
        u, value, query, key = torch.split(z, [3 * h * vdim, h * vdim, h * qdim, h * qdim], dim=-1)
        value = value.reshape(b, n, h, vdim) * valid_mask[:, :, None, None]
        query = query.reshape(b, n, h, qdim)
        key = key.reshape(b, n, h, qdim)
        causal = torch.ones(n, n, device=x.device, dtype=x.dtype).tril()
        content = F.silu(torch.einsum("bnhd,bmhd->bhnm", query, key)) / n
        content = content * causal[None, None]
        position, time = self._rel_attn_bias(timestamps)
        channels = [
            torch.einsum("bnm,bmhd->bnhd", position.expand(b, -1, -1) * causal, value),
            torch.einsum("bnm,bmhd->bnhd", time * causal, value),
            torch.einsum("bhnm,bmhd->bnhd", content, value),
        ]
        # Concatenate inside each head, matching the public channel layout.
        attended = torch.cat(channels, dim=-1).reshape(b, n, 3 * h * vdim)
        output = self._mffn(u * _rms(attended, self.eps), x)
        return output * valid_mask[:, :, None]


class FuXiAlpha(nn.Module):
    def __init__(self, num_items, max_seq_len=200, max_output_len=11, embed_dim=50,
                 num_blocks=2, num_heads=1, dqk=None, dv=None, dropout=0.2,
                 ffn_multiply=1, ffn_single_stage=False, temperature=0.05,
                 l2_norm_eps=1e-6, candidate_ids=None, interaction_bf16=True, _block_factory=None):
        super().__init__()
        if min(num_items, max_seq_len, embed_dim, num_blocks, num_heads) < 1 or temperature <= 0:
            raise ValueError("model dimensions and temperature must be positive")
        self.num_items, self.max_seq_len = num_items, max_seq_len
        self.padded_length = max_seq_len + max_output_len
        self.embed_dim, self.dropout = embed_dim, dropout
        self.temperature, self.l2_norm_eps = temperature, l2_norm_eps
        self.interaction_bf16 = interaction_bf16
        catalogue = torch.arange(1, num_items + 1) if candidate_ids is None else torch.as_tensor(candidate_ids, dtype=torch.long)
        if catalogue.ndim != 1 or not len(catalogue) or catalogue.min() < 1 or catalogue.max() > num_items:
            raise ValueError("candidate_ids must contain valid non-padding catalogue IDs")
        self.register_buffer("candidate_ids", catalogue)
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(self.padded_length, embed_dim)
        block_factory = _block_factory or FuXiAlphaBlock
        self.layers = nn.ModuleList([
            block_factory(embed_dim, self.padded_length, num_heads, dqk, dv,
                           dropout, ffn_multiply, ffn_single_stage)
            for _ in range(num_blocks)
        ])
        nn.init.trunc_normal_(self.item_embedding.weight, std=0.02, a=-0.04, b=0.04)
        self.item_embedding.weight.data[0].zero_()
        nn.init.xavier_normal_(self.position_embedding.weight)

    def encode(self, input_ids, timestamps):
        if input_ids.shape[1] != self.padded_length or timestamps.shape != input_ids.shape:
            raise ValueError("use the fixed right-padded FuXi sequence collator")
        valid = input_ids.ne(0)
        if torch.any(valid[:, 1:] & ~valid[:, :-1]):
            raise ValueError("FuXi input sequences must be right padded")
        pos = self.position_embedding(torch.arange(self.padded_length, device=input_ids.device))
        x = self.item_embedding(input_ids) * math.sqrt(self.embed_dim) + pos
        x = F.dropout(x, self.dropout, self.training) * valid[:, :, None]
        for layer in self.layers:
            x = layer(x, timestamps, valid)
        return F.normalize(x, dim=-1, eps=self.l2_norm_eps)

    def forward(self, input_ids, timestamps, targets=None, num_negatives=128,
                negative_ids=None, last_only=False):
        hidden = self.encode(input_ids, timestamps)
        if targets is None:
            if last_only:
                lengths = input_ids.ne(0).sum(-1)
                if torch.any(lengths == 0):
                    raise ValueError("ranking requires a nonempty history")
                hidden = hidden[torch.arange(len(hidden), device=hidden.device), lengths - 1]
            candidates = F.normalize(self.item_embedding.weight, dim=-1, eps=self.l2_norm_eps)
            return (hidden @ candidates.T) / self.temperature, None
        valid = targets.ne(0)
        if not valid.any():
            raise ValueError("sampled-softmax needs at least one non-padding target")
        positives = targets[valid]
        queries = hidden[valid]
        if negative_ids is None:
            offsets = torch.randint(len(self.candidate_ids), (len(positives), num_negatives), device=targets.device)
            negative_ids = self.candidate_ids[offsets]
        if negative_ids.ndim != 2 or negative_ids.shape[0] != len(positives):
            raise ValueError("negative_ids must have one row per supervised position")
        pos = F.normalize(self.item_embedding(positives), dim=-1, eps=self.l2_norm_eps)
        neg = F.normalize(self.item_embedding(negative_ids), dim=-1, eps=self.l2_norm_eps)
        # The public GeneralizedInteractionModule forces BF16 on CUDA even
        # when the encoder/optimizer run in FP32. Keep this local to the loss;
        # its brute-force evaluation path uses the ordinary ranking dtype.
        with torch.autocast(device_type=queries.device.type, dtype=torch.bfloat16,
                            enabled=self.interaction_bf16 and queries.is_cuda):
            positive_score = torch.bmm(pos.unsqueeze(1), queries.unsqueeze(2)).squeeze(2) / self.temperature
            negative_score = torch.bmm(neg, queries.unsqueeze(2)).squeeze(2) / self.temperature
        negative_score = negative_score.masked_fill(negative_ids.eq(positives[:, None]), -5e4)
        scores = torch.cat([positive_score, negative_score], dim=1)
        # Upstream log-softmax runs in the interaction dtype; its per-position
        # losses are then weighted/reduced in FP32.
        return None, (-F.log_softmax(scores, dim=1)[:, 0]).float().mean()
