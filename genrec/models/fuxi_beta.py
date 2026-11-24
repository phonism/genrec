"""Independent padded FuXi-beta; the pinned public implementation is an oracle.

Reference: USTC-StarTeam/FuXi-beta@4fc73bc6b3a4d66d6a37e4178d567ca83edb7a73.
Implements the published default power temporal function and attention-free
position/time channels. Uses GenRec's shared embeddings, sampled loss and FFN.
"""
import torch
from torch import nn
from torch.nn import functional as F

from genrec.models.fuxi_alpha import FuXiAlpha, FuXiMultistageFFN, _rms


class BetaRelativeBias(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.length = length
        self._pow_a = nn.Parameter(torch.empty(1).uniform_(-0.2, 0.2))
        self._pow_b = nn.Parameter(torch.empty(1).uniform_(0.4, 0.8))
        self._pos_w = nn.Parameter(torch.empty(2 * length - 1).normal_(0, 0.02))

    def forward(self, timestamps):
        p = torch.arange(self.length, device=timestamps.device)
        pos = self._pos_w[self.length - 1 + p[None] - p[:, None]][None]
        next_time = torch.cat([timestamps[:, 1:], timestamps[:, -1:]], 1)
        delta = next_time[:, :, None] - timestamps[:, None, :]
        temporal = self._pow_a * (delta.clamp(min=0) + 1).pow(-self._pow_b)
        return pos, temporal


class FuXiBetaBlock(nn.Module):
    def __init__(self, width, max_length, num_heads=1, dqk=None, dv=None,
                 dropout=0.2, ffn_multiply=1, single_stage=False, eps=1e-6):
        super().__init__()
        if single_stage:
            raise ValueError("the public FuXi-beta default uses the multistage FFN")
        self.heads, self.dv, self.eps = num_heads, dv or width, eps
        self._uvqk = nn.Parameter(torch.empty(width, 3 * self.heads * self.dv).normal_(0, 0.02))
        self._rel_attn_bias = BetaRelativeBias(max_length)
        self._mffn = FuXiMultistageFFN(2 * self.heads * self.dv, width,
                                      int(width * ffn_multiply), dropout, eps=eps)

    def forward(self, x, timestamps, valid_mask):
        b, n, _ = x.shape
        h, dv = self.heads, self.dv
        projected = F.silu(_rms(x, self.eps) @ self._uvqk)
        u, v = projected.split([2 * h * dv, h * dv], -1)
        value = v.reshape(b, n, h, dv) * valid_mask[:, :, None, None]
        position, temporal = self._rel_attn_bias(timestamps)
        causal = torch.ones(n, n, device=x.device, dtype=x.dtype).tril()
        channels = [torch.einsum("bnm,bmhd->bnhd", weights * causal, value)
                    for weights in [position.expand(b, -1, -1), temporal]]
        mixed = torch.cat(channels, -1).reshape(b, n, 2 * h * dv)
        return self._mffn(u * _rms(mixed, self.eps), x) * valid_mask[:, :, None]


class FuXiBeta(FuXiAlpha):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, _block_factory=FuXiBetaBlock, **kwargs)
