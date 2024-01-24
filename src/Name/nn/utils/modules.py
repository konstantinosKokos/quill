import torch
from torch import Tensor
from torch.nn import Module, Parameter, Linear, Dropout
from torch.nn.functional import silu

from .attention import taylor_atn_fn


class SwiGLU(Module):
    def __init__(self, input_dim: int, interm_dim: int, output_dim: int):
        super(SwiGLU, self).__init__()
        self.w_in = Linear(input_dim, interm_dim, bias=False)
        self.v = Linear(input_dim, interm_dim, bias=False)
        self.w_out = Linear(interm_dim, output_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        interm = self.w_in(x)
        interm = silu(interm) * self.v(x)
        return self.w_out(interm)


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class TMHA(Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.transformations = Linear(dim, head_dim * num_heads * 2 + dim, bias=False)
        self.wo = Linear(in_features=dim, out_features=dim, bias=False)
        self.num_heads = num_heads
        self.qk_dim = num_heads * head_dim
        self.dim = dim

    def forward(
            self,
            x: Tensor,
            mask: Tensor,
            rotator: Tensor) -> Tensor:
        x = self.transformations(x)
        qs = x[..., :self.qk_dim].view(x.size(0), x.size(1), self.num_heads, -1)
        ks = x[..., self.qk_dim:(2*self.qk_dim)].view(x.size(0), x.size(1), self.num_heads, -1)
        vs = x[..., 2*self.qk_dim:].view(x.size(0), x.size(1), self.num_heads, -1)
        qs[mask] = torch.einsum('...ij,...hj->...hi', rotator[mask], qs[mask])
        ks[mask] = torch.einsum('...ij,...hi->...hj', rotator[mask], ks[mask])
        out = taylor_atn_fn(qs, ks, vs, mask)
        return self.wo(out)


class ResidualFFN(Module):
    def __init__(self, dim: int, intermediate: int, dropout_rate: float):
        super(ResidualFFN, self).__init__()
        self.pre_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, intermediate, dim)
        self.dropout = Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        ffn = self.pre_norm(x)
        ffn = self.ffn(ffn)
        ffn = self.dropout(ffn)
        return ffn + x


class EncoderLayer(Module):
    def __init__(self, num_heads: int, dim: int, dropout_rate: float, head_dim: int):
        super(EncoderLayer, self).__init__()
        self.mha_norm = RMSNorm(dim)
        self.mha = TMHA(dim, num_heads, head_dim)
        self.res_ffn = ResidualFFN(dim, 4 * dim, dropout_rate)
        self.dropout = Dropout(dropout_rate)

    def forward(self, encoder_input: Tensor, attention_mask: Tensor, rotator: Tensor) -> Tensor:
        mha_x = self.mha_norm(encoder_input)
        mha_x = self.mha.forward(mha_x, attention_mask, rotator)
        mha_x = self.dropout(mha_x)
        mha_x = encoder_input + mha_x
        return self.res_ffn(mha_x)
