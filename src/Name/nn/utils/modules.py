import torch
from torch import Tensor
from torch.nn import Module, Parameter, Linear, Dropout
from torch.nn.functional import silu, embedding

from .attention import atn_fn


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


class LMHA(Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.transformations = Linear(dim, 2 * num_heads * head_dim + dim, bias=False)
        self.wo = Linear(in_features=dim, out_features=dim, bias=False)
        self.num_heads = num_heads
        self.qk_dim = num_heads * head_dim
        self.dim = dim

    def forward(
            self,
            x: Tensor,
            pe: Tensor,
            mask: Tensor
    ) -> Tensor:
        x = self.transformations(x)
        qs = x[..., :self.qk_dim].view(x.size(0), x.size(1), self.num_heads, -1)
        ks = x[..., self.qk_dim:(2*self.qk_dim)].view(x.size(0), x.size(1), self.num_heads, -1)
        vs = x[..., 2*self.qk_dim:].view(x.size(0), x.size(1), self.num_heads, -1)
        qs, ks = Rotary.apply_rotary_position_embeddings(pe, qs, ks)
        out = atn_fn(qs, ks, vs, mask)
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
        self.mha = LMHA(dim, num_heads, head_dim)
        self.res_ffn = ResidualFFN(dim, 4 * dim, dropout_rate)
        self.dropout = Dropout(dropout_rate)

    def forward(self, encoder_input: Tensor, pe: Tensor, attention_mask: Tensor) -> Tensor:
        mha_x = self.mha_norm(encoder_input)
        mha_x = self.mha.forward(mha_x, pe, attention_mask)
        mha_x = self.dropout(mha_x)
        mha_x = encoder_input + mha_x
        return self.res_ffn(mha_x)


class Rotary(Module):
    weight: Tensor

    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        super().__init__()
        self.register_buffer('weight', self._init_weight(num_positions, embedding_dim))

    @staticmethod
    def _init_weight(n_pos: int, dim: int) -> Tensor:
        out = torch.zeros(n_pos, dim, dtype=torch.float)
        position_enc = torch.tensor(
            [[pos / (10000 ** (2 * (j // 2) / dim)) for j in range(dim)] for pos in range(n_pos)]
        )
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.sin(position_enc[:, 0::2])
        out[:, sentinel:] = torch.cos(position_enc[:, 1::2])
        return out

    @torch.no_grad()
    def forward(self, max_seq_len: int) -> Tensor:
        positions = torch.arange(
            start=0, end=max_seq_len, dtype=torch.long, device=self.weight.device
        )
        return embedding(positions, self.weight)

    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer):
        num_qs = query_layer.shape[1]
        num_ks = key_layer.shape[1]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
            query_layer
        )
        query_layer = query_layer * cos_pos[None, :num_qs, :, None] + rotate_half_query_layer * sin_pos[None, :num_qs,
                                                                                                :, None]
        rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
        key_layer = key_layer * cos_pos[None, :num_ks, :, None] + rotate_half_key_layer * sin_pos[None, :num_ks, :,
                                                                                          None]
        return query_layer, key_layer
