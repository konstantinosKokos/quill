import pdb

import torch
from torch import Tensor, device
from ..data.tokenization import TokenizedSample, TokenizedFile, TokenizedTree, DeBruijn
from .utils import pad_to_length
from random import sample
from math import ceil
from typing import Callable, TypeVar


_T = TypeVar('_T')


def permute(xs: list[_T]) -> list[_T]:
    return sample(xs, len(xs))


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


class Sampler:
    def __init__(self, data: list[TokenizedFile], max_types: int, max_tokens: int, max_db: int):
        def filter_tree(tree: TokenizedTree) -> bool:
            return len(tree) <= max_tokens and all([index < max_db for nt, index, _, _ in tree if nt == 4])

        def filter_hole(hole: tuple[TokenizedTree, list[int]]) -> bool:
            tree, goal = hole
            return filter_tree(tree) and any(index != -1 for index in goal)

        def filter_file(file: TokenizedFile) -> bool:
            scope, holes = file
            return len(scope) <= max_types and all(map(filter_tree, scope)) and len(holes) > 0

        filtered = [(scope, [hole for hole in holes if filter_hole(hole)]) for scope, holes in data]
        self.filtered = [file for file in filtered if filter_file(file)]
        print(f'Kept {len(self.filtered)} of {len(data)} files,'
              f' and {sum(len(hs) for _, hs in self.filtered)} of {sum(len(hs) for _, hs in data)} holes.')
        self.indices = [(i, list(range(len(self.filtered[i][1])))) for i in range(len(self.filtered))]

    def iter(self, batch_size: int):
        perm_indices = [(scope_idx, permute(hole_ids)) for scope_idx, hole_ids in self.indices]
        epoch_indices = [(scope_idx, hole_ids[batch_size * j:min(batch_size * (j + 1), len(hole_ids))])
                         for scope_idx, hole_ids in perm_indices
                         for j in range(0, ceil(len(hole_ids) / batch_size))]
        epoch_indices = permute(epoch_indices)
        for scope_idx, hole_ids in epoch_indices:
            batch = [(self.filtered[scope_idx][0], [self.filtered[scope_idx][1][h] for h in hole_ids])]
            yield batch
