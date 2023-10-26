import torch
from torch import Tensor
from opt_einsum import contract as _contract


def contract(indices: str, *tensors: Tensor) -> Tensor:
    return _contract(indices, *tensors)   # type: ignore


def atn_fn(queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor) -> Tensor:
    batch_size, seq_len, dk, num_heads = keys.shape
    dividend = torch.sqrt(torch.tensor(dk, device=queries.device, dtype=torch.float))

    weights = contract('bidh,bodh->bioh', queries, keys) / dividend

    if mask.shape == (batch_size, seq_len):
        mask = mask[:, None, :]

    weights = weights.masked_fill_((~mask[:, :, :, None]), value=-1e10)
    weights = weights.softmax(dim=-2)
    return contract('bioh,bodh->bidh', weights, values).flatten(-2)


def lin_atn_fn(queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor) -> Tensor:
    _, _, d_atn, _ = queries.shape
    dividend = torch.sqrt(torch.tensor(d_atn, device=queries.device, dtype=torch.float))
    queries = queries / dividend

    mask = mask[:, :, None, None]
    keys = keys.masked_fill(~mask, 0)
    values = values.masked_fill(~mask, 0)

    queries = queries.softmax(dim=-2)
    keys = keys.softmax(dim=-3)

    context = contract('bnkh,bnvh->bkvh', keys, values)
    return contract('bnkh,bkvh->bnvh', queries, context).flatten(-2)
