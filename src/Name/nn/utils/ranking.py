import pdb

import torch
from torch import Tensor
from torch.nn.functional import dropout
from torch_geometric.nn.pool import global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch

from enum import Enum


def global_min_pool(x: Tensor, batch_ids: Tensor, size: int | None = None) -> Tensor:
    return -global_max_pool(-x, batch_ids, size=size)


class Strategy(Enum):
    MIN = global_min_pool
    MAX = global_max_pool
    AVE = global_mean_pool
    ALL = lambda x, batch_ids, size: x

    def __call__(self, x: Tensor, batch_ids: Tensor, size: int | None = None) -> Tensor:
        return self.value(x, batch_ids, size)


def margin_ranking(
        preds: Tensor,
        targets: Tensor,
        edge_index: Tensor,
        margin: float = 0.1,
        pos_strategy: Strategy = Strategy.MAX,
        neg_strategy: Strategy = Strategy.MAX,
        positive_sampling: float = 0.95,
        negative_sampling: float = 0.95) -> Tensor:
    if len(preds) == 0:
        return preds

    assert neg_strategy != Strategy.ALL, 'Whoa there cowboy!'

    positive_mask = targets & (torch.rand_like(targets, dtype=torch.float) < positive_sampling)
    negative_mask = targets.logical_not() & (torch.rand_like(targets, dtype=torch.float) < negative_sampling)

    positive_preds = preds[positive_mask]
    negative_preds = preds[negative_mask]
    positive_mask = edge_index[1, positive_mask]
    negative_mask = edge_index[1, negative_mask]

    positives, negatives = None, None
    if len(positive_mask) > 0:
        positives = pos_strategy(positive_preds, positive_mask, size=max(edge_index[1]) + 1)
    if len(negative_mask) > 0:
        negatives = neg_strategy(negative_preds, negative_mask, size=max(edge_index[1]) + 1)
        if pos_strategy == Strategy.ALL:
            negatives = negatives[edge_index[1, positive_mask]]

    if positives is None:
        positives = torch.zeros_like(negatives)
    if negatives is None:
        negatives = torch.zeros_like(positives)

    return torch.clamp(negatives - positives + margin, min=0)


def rank_candidates(x: Tensor, batch_ids: Tensor) -> tuple[Tensor, Tensor]:
    dense_x, x_mask = to_dense_batch(x=x, batch=batch_ids, fill_value=-1e08)
    _, perm = dense_x.sort(dim=-1, descending=True)
    # todo: you cant recover original values without the extra indexing ops
    return perm, x_mask.sum(dim=-1)


def average_precision(ranked_suggestions: list[int], relevant_items: set[int]) -> float:
    ap = 0
    for item in relevant_items:
        to_reach = set(ranked_suggestions[:ranked_suggestions.index(item) + 1])
        ap += len(to_reach & relevant_items)/(len(to_reach) + 1e-08)
    return ap/(len(relevant_items) + 1e-08)


def rprecision(ranked_suggestions: list[int], relevant_items: set[int]) -> float:
    return len(set(ranked_suggestions[:len(set(relevant_items))]) & relevant_items) / (len(relevant_items) + 1e-08)


def evaluate_rankings(predictions: Tensor, batch_ids: Tensor, truths: Tensor) -> list[tuple[float, float]]:
    if len(batch_ids) == 0:
        return []

    ranked_suggestions = rank_candidates(predictions, batch_ids)[0].cpu().tolist()
    dense_truths, _ = to_dense_batch(truths, batch_ids, fill_value=0)
    relevant_items = [{i for i, value in enumerate(row) if value} for row in dense_truths.cpu().bool().tolist()]
    return [(average_precision(rs, ri), rprecision(rs, ri))
            for rs, ri in zip(ranked_suggestions, relevant_items)]


