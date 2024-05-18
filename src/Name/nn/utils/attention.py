import torch
from torch import Tensor
from torch.nn.functional import elu


def phi(x: Tensor) -> Tensor:
    return elu(x + 1)


def atn_fn(queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor) -> Tensor:
    batch_size, seq_len, num_heads, dk = keys.size()
    queries = queries * (dk ** -0.5)
    queries, keys = map(phi, (queries, keys))

    keys = keys.masked_fill(~mask[:, :, None, None], value=0.)
    values = values.masked_fill(~mask[:, :, None, None], value=0.)
    kv = torch.einsum('bnhd,bnhe->bhde', keys, values)
    qk_inv = 1. / torch.einsum('bnhd,bmhd->bnh', queries, keys).clamp(min=1e-12)
    return torch.einsum('bnhd,bhde,bnh->bnhe', queries, kv, qk_inv).flatten(-2)
