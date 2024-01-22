import torch
from torch import Tensor


def taylor_2(x: Tensor) -> Tensor:
    x0 = torch.ones(x.size()[:-1], device=x.device).unsqueeze(-1)
    x2 = (torch.einsum('...i,...j->...ij', x, x)).flatten(-2) * (0.5 ** 0.5)
    return torch.cat((x0, x, x2), dim=-1)


def taylor_atn_fn(queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor) -> Tensor:
    batch_size, seq_len, num_heads, dk = keys.size()
    dividend = torch.sqrt(torch.tensor(dk, device=queries.device, dtype=torch.float))
    queries = queries / dividend
    queries, keys = map(taylor_2, (queries, keys))

    keys = keys.masked_fill(~mask[:, :, None, None], value=0.)
    values = values.masked_fill(~mask[:, :, None, None], value=0.)
    kv = torch.einsum('bnhd,bnhe->bhde', keys, values)
    qk_inv = 1. / torch.einsum('bnhd,bmhd->bnh', queries, keys).clamp(min=1e-12)
    return torch.einsum('bnhd,bhde,bnh->bnhe', queries, kv, qk_inv).flatten(-2)
