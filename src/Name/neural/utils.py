import torch
from opt_einsum import contract
from torch import Tensor
from torch.nn import Module, Parameter, Linear, Dropout
from torch.nn.utils.rnn import pad_sequence as _pad_sequence
from typing import TypeVar, Callable
from warnings import warn
from math import cos, radians
from random import sample

_T = TypeVar('_T')


def permute(xs: list[_T]) -> list[_T]:
    return sample(xs, len(xs))


def select(xs: list[_T], ids: list[int]) -> list[_T]:
    return [xs[i] for i in ids]


def sublists(xs: list[_T], of_size: int) -> list[list[_T]]:
    return [xs[i:i+of_size] for i in range(0, len(xs), of_size)]


def make_schedule(warmup_steps: int,
                  total_steps: int,
                  warmdown_steps: int,
                  max_lr: float,
                  min_lr: float) -> Callable[[int], float]:
    linear_schedule = make_linear_schedule(warmup_steps, max_lr)
    cosine_schedule = make_cosine_schedule(warmdown_steps, max_lr, min_lr)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return linear_schedule(step)
        elif step < total_steps - warmdown_steps:
            return max_lr
        elif step > total_steps:
            warn(f"Step is greater than total steps")
            return min_lr
        return cosine_schedule(step - (total_steps - warmdown_steps))
    return schedule


def make_linear_schedule(warmup_steps: int, max_lr: float) -> Callable[[int], float]:
    def linear_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps * max_lr
        return max_lr
    return linear_schedule


def make_cosine_schedule(decay_steps: int, max_lr: float, min_lr: float) -> Callable[[int], float]:
    def cosine_schedule(step: int) -> float:
        if step <= decay_steps:
            return min_lr + (max_lr - min_lr) * (cos(radians(step / decay_steps * 180)) + 1) / 2
        return min_lr
    return cosine_schedule


def pad_sequence(xs: list[Tensor], padding_value: float) -> Tensor:
    return _pad_sequence(xs, batch_first=True, padding_value=padding_value)


def swish(x: Tensor, b: int = 1) -> Tensor:
    gate = torch.sigmoid(b * x)
    return x * gate


def binary_stats(predictions: list[bool], truths: list[bool]) -> tuple[int, int, int, int]:
    tp = sum([x == y for x, y in zip(predictions, truths) if y])
    fn = sum([x != y for x, y in zip(predictions, truths) if y])
    tn = sum([x == y for x, y in zip(predictions, truths) if not y])
    fp = sum([x != y for x, y in zip(predictions, truths) if not y])
    return tp, fn, tn, fp


def macro_binary_stats(tp: int, fn: int, tn: int, fp: int) -> tuple[float, float, float, float]:
    prec = tp / (tp + fp + 1e-08)
    rec = tp / (tp + fn + 1e-08)
    f1 = 2 * prec * rec / (prec + rec + 1e-08)
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    return accuracy, f1, prec, rec


class Swish(Module):
    def __init__(self, b: int = 1):
        super(Swish, self).__init__()
        self.b = b

    def forward(self, x: Tensor):
        return swish(x, self.b)


class SwiGLU(Module):
    def __init__(self, input_dim: int, interm_dim: int, output_dim: int):
        super(SwiGLU, self).__init__()
        self.w_in = Linear(input_dim, interm_dim, bias=False)
        self.v = Linear(input_dim, interm_dim, bias=False)
        self.w_out = Linear(interm_dim, output_dim, bias=False)

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
    weights = weights.masked_fill_((~mask[:, None, :, None]), value=-1e10)
    weights = weights.softmax(dim=-2)
    return torch.einsum('bioh,bodh->bidh', weights, values).flatten(-2)


def lin_atn_fn(queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor) -> Tensor:
    d_atn, num_heads = queries.shape[-2:]
    queries = queries / d_atn

    mask = mask[:, :, None, None]
    keys = keys.masked_fill(~mask, 0)
    values = values.masked_fill(~mask, 0)

    queries = queries.softmax(dim=-2)
    keys = keys.softmax(dim=-3)

    context = contract('bnkh,bnvh->bkvh', keys, values)
    return contract('bnkh,bkvh->bnvh', queries, context).flatten(-2)


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


class LinearMHA(Module):
    def __init__(self, num_heads: int, dim: int, d_atn: int, dropout_rate: float = 0.1):
        super(LinearMHA, self).__init__()
        self.num_heads = num_heads
        if (dim % num_heads != 0) or (d_atn % num_heads != 0):
            raise ValueError('dim must be divisible by num_heads')
        self.q_transformation = Linear(in_features=dim, out_features=d_atn * num_heads, bias=False)
        self.k_transformation = Linear(in_features=dim, out_features=d_atn * num_heads, bias=False)
        self.v_transformation = Linear(in_features=dim, out_features=dim, bias=False)
        self.wo = Linear(in_features=dim, out_features=dim, bias=False)
        self.dropout = Dropout(dropout_rate)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        qs = self.q_transformation(queries).view(queries.shape[0], queries.shape[1], -1, self.num_heads)
        ks = self.k_transformation(keys).view(keys.shape[0], keys.shape[1], -1, self.num_heads)
        vs = self.v_transformation(values).view(values.shape[0], values.shape[1], -1, self.num_heads)
        mha = lin_atn_fn(qs, ks, vs, mask)
        mha = self.dropout(mha)
        return self.wo(mha)


class ResidualFFN(Module):
    def __init__(self, dim: int, intermediate: int, dropout_rate: float):
        super(ResidualFFN, self).__init__()
        self.pre_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, intermediate, dim)
        self.dropout = Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        norm = self.pre_norm(x)
        ffn = self.ffn(norm)
        return ffn + norm
