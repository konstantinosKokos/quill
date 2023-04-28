import pdb

import torch
from torch import Tensor, device
from ..data.tokenization import TokenizedSample, TokenizedFile, TokenizedTree
from .utils import sublists, permute, select
from math import ceil
from typing import Iterator
from itertools import groupby
from torch.nn.functional import pad as _pad
from torch.nn.utils.rnn import pad_sequence as _pad_sequence


def make_collator(cast_to: device = device('cpu'),
                  goal_id: int = -1,
                  padding: int = -1):
    def _longt(xs) -> Tensor:
        return torch.tensor(xs, device=cast_to, dtype=torch.long)

    def _boolt(xs) -> Tensor:
        return torch.tensor(xs, device=cast_to, dtype=torch.bool)

    def pad_tree(tree: TokenizedTree, to: int) -> Tensor:
        return _pad(_longt(tree), pad=(0, 0, 0, to - len(tree)), mode='constant', value=padding)

    def pad_seq(file: list[Tensor]) -> Tensor:
        return _pad_sequence(file, batch_first=True, padding_value=padding)

    def collator(samples: list[TokenizedSample]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        num_scopes = len(samples)
        scope_sizes, goal_sizes = zip(*[(len(scope), len(holes)) for scope, holes in samples])
        most_trees = max(x+y for x, y in zip(scope_sizes, goal_sizes))
        trees = [[*scope, *[goal_type for goal_type, _ in holes]] for scope, holes in samples]
        most_tokens = max(max(len(tree) for tree in file) for file in trees)

        scope_ranges = [range((start := i * most_trees), start + offset) for i, offset in enumerate(scope_sizes)]
        goal_ranges = [range((start := scope_range.stop), start + num_goals)
                       for scope_range, num_goals in zip(scope_ranges, goal_sizes)]
        source_index = [torch.arange(scope_range.start, scope_range.stop, device=cast_to).repeat(num_goals)
                        for scope_range, num_goals in zip(scope_ranges, goal_sizes)]
        target_index = [torch.arange(goal_range.start, goal_range.stop, device=cast_to).repeat_interleave(scope_size)
                        for goal_range, scope_size in zip(goal_ranges, scope_sizes)]
        edge_index = torch.stack((torch.cat(source_index), torch.cat(target_index)))
        gold_labels = _boolt([i in refs for scope, holes in samples for _, refs in holes for i in range(len(scope))])
        dense_batch = pad_seq([pad_seq([pad_tree(tree, most_tokens) for tree in file]) for file in trees])

        token_padding_mask = (dense_batch != padding).any(dim=-1)
        token_attention_mask = token_padding_mask.flatten(0, 1).unsqueeze(-2).expand(-1, most_tokens, -1)

        tree_padding_mask = token_padding_mask.any(dim=-1)
        is_goal = (dense_batch[:, :, :, -1] == goal_id).all(dim=-1) & tree_padding_mask
        scope_attention_mask = (~is_goal & tree_padding_mask).unsqueeze(-2).expand(-1, most_trees, -1)
        diag_mask = torch.eye(most_trees, dtype=torch.bool, device=cast_to).unsqueeze(0).expand(num_scopes, -1, -1)
        tree_attention_mask = scope_attention_mask | (diag_mask & tree_padding_mask.unsqueeze(-1))
        return (dense_batch.permute(-1, 0, 1, 2),
                token_attention_mask,
                tree_attention_mask,
                edge_index,
                gold_labels)
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
        self.hole_counts = [len(holes) for _, holes in self.filtered]
        self.max_types = max_types

    def iter(self, batch_size_s: int, batch_size_h: int) -> Iterator[list[TokenizedSample]]:
        permuted_holes = [permute(list(range(nhs))) for nhs in self.hole_counts]
        epoch_indices = [(s_idx, selection)
                         for s_idx, phs in enumerate(permuted_holes)
                         for selection in sublists(phs, batch_size_h)]
        epoch_indices = permute(epoch_indices)
        for b in range(ceil(len(epoch_indices) / batch_size_s)):
            batch_indices = epoch_indices[batch_size_s * b:min(batch_size_s * (b + 1), len(epoch_indices))]
            batch_indices = [(scope_id, [hole for _, holes in items for hole in holes])
                             for scope_id, items
                             in groupby(sorted(batch_indices, key=lambda x: x[0]), key=lambda x: x[0])]
            yield [(self.filtered[scope_idx][0], select(self.filtered[scope_idx][1], hole_ids))
                   for scope_idx, hole_ids in batch_indices]
1