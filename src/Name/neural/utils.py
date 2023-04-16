import torch
from opt_einsum import contract
from torch import Tensor
from torch.nn import Module, Parameter, Linear, Dropout
from torch.nn.utils.rnn import pad_sequence as _pad_sequence
from typing import TypeVar

_T = TypeVar('_T')


def pad_sequence(xs: list[Tensor], padding_value: float) -> Tensor:
    return _pad_sequence(xs, batch_first=True, padding_value=padding_value)


def pad_to_length(xs: list[list[_T]], pad_to: int, value: _T) -> list[list[_T]]:
    if any(len(x) > pad_to for x in xs):
        raise ValueError
    return [x + ([value] * (pad_to - len(x))) for x in xs]


def swish(x: Tensor, b: int = 1) -> Tensor:
    return x * torch.sigmoid(b * x)


class Swish(Module):
    def __init__(self, b: int = 1):
        super(Swish, self).__init__()
        self.b = b

    def forward(self, x: Tensor):
        return swish(x, self.b)


class SwiGLU(Module):
    def __init__(self, input_dim: int, interm_dim: int, output_dim: int):
        super(SwiGLU, self).__init__()
        self.w_in = Linear(input_dim, interm_dim)
        self.v = Linear(input_dim, interm_dim)
        self.w_out = Linear(interm_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        interm = self.w_in(x)
        interm = swish(interm) * self.v(x)
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


def atn_fn(queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor) -> Tensor:
    dk, num_heads = keys.shape[-2:]
    dividend = torch.sqrt(torch.tensor(dk, device=queries.device, dtype=torch.float))

    weights = contract('bidh,bodh->bioh', queries, keys) / dividend
    weights = weights.masked_fill_(mask.unsqueeze(-1), value=-1e10)
    weights = weights.softmax(dim=-2)
    return torch.einsum('bioh,bodh->bidh', weights, values).flatten(-2)


class MHA(Module):
    def __init__(self, num_heads: int, dim: int, dropout_rate: float = 0.1):
        super(MHA, self).__init__()
        self.num_heads = num_heads
        if dim % num_heads != 0:
            raise ValueError('dim must be divisible by num_heads')
        self.q_transformation = Linear(in_features=dim, out_features=dim, bias=False)
        self.k_transformation = Linear(in_features=dim, out_features=dim, bias=False)
        self.v_transformation = Linear(in_features=dim, out_features=dim, bias=False)
        self.wo = Linear(in_features=dim, out_features=dim, bias=False)
        self.dropout = Dropout(dropout_rate)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        qs = self.q_transformation(queries).view(queries.shape[0], queries.shape[1], -1, self.num_heads)
        ks = self.k_transformation(keys).view(keys.shape[0], keys.shape[1], -1, self.num_heads)
        vs = self.v_transformation(values).view(values.shape[0], values.shape[1], -1, self.num_heads)
        mha = atn_fn(qs, ks, vs, mask)
        mha = self.dropout(mha)
        return self.wo(mha)
