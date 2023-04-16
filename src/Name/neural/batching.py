import pdb

import torch
from torch import Tensor, device
from ..data.tokenization import TokenizedSample
from .utils import pad_to_length


def make_collator(cast_to: device = device('cpu'),
                  goal_id: int = -1,
                  padding: tuple[int, int, int, int] = (-1, -1, -1, -1)):
    def collator(samples: list[TokenizedSample]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        num_scopes = len(samples)
        most_trees = max(len(scope) + len(holes) for scope, holes in samples)
        trees = [scope + [goal_type for goal_type, _ in holes] for scope, holes in samples]
        most_tokens = max(max(len(tree) for tree in batch) for batch in trees)

        trees_per_sample, goals_per_sample = zip(*[(len(scope), len(holes)) for scope, holes in samples])
        scope_ranges = [range((start := i * most_trees), start + offset) for i, offset in enumerate(trees_per_sample)]
        goal_ranges = [range(scope_range.start, scope_range.start + num_goals)
                       for scope_range, num_goals in zip(scope_ranges, goals_per_sample)]
        edge_index = [(scope_index, goal_index)
                      for scope_range, goal_range in zip(scope_ranges, goal_ranges)
                      for scope_index in scope_range for goal_index in goal_range]
        gold_labels = [i in refs for scope, holes in samples for i in range(len(scope)) for _, refs in holes]

        padded_trees = [pad_to_length(batch, most_tokens, padding) for batch in trees]
        padded_batches = pad_to_length(padded_trees, most_trees, [padding] * most_tokens)

        dense_batch = torch.tensor(padded_batches)
        token_padding_mask = (dense_batch != torch.tensor(padding)).all(dim=-1)
        tree_padding_mask = token_padding_mask.any(dim=-1).unsqueeze(-2).expand(-1, most_trees, -1)
        goal_mask = (dense_batch[:, :, :, -1] != goal_id).all(-1)
        diag_mask = torch.eye(most_trees, dtype=torch.bool).unsqueeze(0).expand(num_scopes, -1, -1)
        token_attention_mask = token_padding_mask.flatten(0, 1).unsqueeze(-2).expand(-1, most_tokens, -1)
        tree_attention_mask = tree_padding_mask & (goal_mask.unsqueeze(-1) | diag_mask)

        return (dense_batch.permute(-1, 0, 1, 2).to(cast_to),
                token_attention_mask.to(cast_to),
                tree_attention_mask.to(cast_to),
                torch.tensor(edge_index).t().to(cast_to),
                torch.tensor(gold_labels).to(cast_to))
    return collator
