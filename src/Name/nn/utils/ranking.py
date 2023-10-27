import pdb

import torch
from torch import Tensor
from torch_geometric.nn.pool import global_max_pool
from torch_geometric.utils import to_dense_batch


def margin_ranking(
        preds: Tensor,
        targets: Tensor,
        edge_index: Tensor,
        margin: float = 0.1,
        positive_sampling: float = 0.95,
        negative_sampling: float = 0.95) -> Tensor:
    if len(preds) == 0:
        return preds

    positive_mask = targets & (torch.rand_like(targets, dtype=torch.float) < positive_sampling)
    negative_mask = targets.logical_not() & (torch.rand_like(targets, dtype=torch.float) < negative_sampling)

    positive_preds = preds[positive_mask]
    negative_preds = preds[negative_mask]
    positive_mask = edge_index[1, positive_mask]
    negative_mask = edge_index[1, negative_mask]

    negatives = global_max_pool(negative_preds, negative_mask, size=max(edge_index[1]) + 1)

    negative_pairs = negatives[edge_index[1, positive_mask]]
    difference_mask = (negative_pairs - positive_preds) > 0
    positive_mask = positive_mask[difference_mask]
    positives = global_max_pool(positive_preds[difference_mask], positive_mask, size=max(edge_index[1]) + 1)
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


