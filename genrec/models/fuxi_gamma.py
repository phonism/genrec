"""Padded reconstruction of the released dense FuXi-gamma base path.

Numerical reference: Yeedzhi/FuXi-gamma@b434e5daab49cc973740af7b4d219f7133b9d84b.
The release's PyTorch path uses dense causal position/time mixing: this module
does not claim to reproduce its paper's custom diagonal-sparse kernels.
No upstream trainer/model is imported at runtime.
"""
import torch
from torch import nn
from torch.nn import functional as F

from genrec.models.fuxi_alpha import FuXiAlpha


class GammaRMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return self.gamma * (x / torch.sqrt(x.square().mean(-1, keepdim=True) + self.eps))


class GammaRelativeBias(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.length = length
        self._alpha = nn.Parameter(torch.empty(1).uniform_(-0.6, 0.6))
        self._beta = nn.Parameter(torch.empty(1).uniform_(0.4, 0.6))
        self._gamma = nn.Parameter(torch.tensor(0.8), requires_grad=False)
        self._pos_w = nn.Parameter(torch.empty(2 * length - 1).normal_(0, 0.02))

    def forward(self, timestamps):
        p = torch.arange(self.length, device=timestamps.device)
        position = self._pos_w[self.length - 1 + p[None] - p[:, None]][None]
        next_time = torch.cat([timestamps[:, 1:], timestamps[:, -1:]], 1)
        delta = (next_time[:, :, None] - timestamps[:, None, :]).float().abs()
        return self._alpha * self._gamma.pow(delta.pow(self._beta)), position


class GammaSwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.w1, self.w2 = nn.Linear(width, hidden, bias=False), nn.Linear(width, hidden, bias=False)
        self.w3 = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class FuXiGammaBlock(nn.Module):
    def __init__(self, width, max_length, num_heads=1, dqk=None, dv=None,
                 dropout=0.2, ffn_multiply=1, single_stage=False, eps=1e-6):
        super().__init__()
        if num_heads * (dv or width) != width:
            raise ValueError("released FuXi-gamma requires num_heads * dv == embed_dim")
        if single_stage:
            raise ValueError("released FuXi-gamma uses the separate SwiGLU FFN")
        self.dropout = dropout
        self._uv = nn.Parameter(torch.empty(width, 3 * width).normal_(0, 0.02))
        self._rel_attn = GammaRelativeBias(max_length)
        self._o = nn.Linear(2 * width, width)
        nn.init.xavier_uniform_(self._o.weight)
        self._swiglu_ffn = GammaSwiGLU(width, int(width * ffn_multiply))
        self._norm_input, self._norm_attn_output, self._norm_ffn = (
            GammaRMSNorm(width, eps), GammaRMSNorm(2 * width, eps), GammaRMSNorm(width, eps))

    def forward(self, x, timestamps, valid_mask):
        n, width = x.shape[1:]
        u, v = F.silu(self._norm_input(x) @ self._uv).split([2 * width, width], -1)
        v = v * valid_mask[:, :, None]
        time, position = self._rel_attn(timestamps)
        causal = torch.ones(n, n, device=x.device, dtype=x.dtype).tril()
        mixed = torch.cat([torch.einsum("bnm,bmd->bnd", weights * causal, v)
                           for weights in [time, position.expand(len(x), -1, -1)]], -1)
        y = x + self._o(F.dropout(u * self._norm_attn_output(mixed), self.dropout, self.training))
        y = y + F.dropout(self._swiglu_ffn(self._norm_ffn(y)), self.dropout, self.training)
        return y * valid_mask[:, :, None]


class FuXiGamma(FuXiAlpha):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, _block_factory=FuXiGammaBlock, **kwargs)
