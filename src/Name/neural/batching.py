import pdb

import torch
from torch import Tensor, device
from ..data.tokenization import TokenizedSample, TokenizedFile, TokenizedTree
from .utils import sublists, permute, select, pad_sequence
from math import ceil
from typing import Iterator, Callable
from itertools import groupby
from torch.nn.functional import pad as _pad

EightTensors = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


def make_collator(cast_to: device = device('cpu'),
                  pad_value: int = -1,) -> Callable[[list[TokenizedSample], float], EightTensors]:
    def _longt(xs) -> Tensor:
        return torch.tensor(xs, device=cast_to, dtype=torch.long)

    def _boolt(xs) -> Tensor:
        return torch.tensor(xs, device=cast_to, dtype=torch.bool)

    def pad_tree(tree: TokenizedTree, to: int) -> Tensor:
        return _pad(_longt(tree), pad=(0, 0, 0, to - len(tree)), mode='constant', value=pad_value)

    def pad_seq(file: list[Tensor]) -> Tensor:
        return pad_sequence(file, padding_value=pad_value)

    def collator(samples: list[TokenizedSample], lm_chance: float) -> EightTensors:
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

        token_padding_mask = (dense_batch != pad_value).any(dim=-1)
        token_attention_mask = token_padding_mask.flatten(0, 1)

        tree_padding_mask = token_padding_mask.any(dim=-1)

        ref_mask = (dense_batch[:, :, :, 0] == 3) & (dense_batch[:, :, :, 1] != -1)
        lm_mask = (torch.rand_like(token_padding_mask, dtype=torch.float) < lm_chance) & ref_mask
        masked_refs = dense_batch[lm_mask]
        mask_values = dense_batch[lm_mask][:, 1]
        masked_refs[:, 0] = 5
        masked_refs[:, 1] = 0
        dense_batch.masked_scatter_(lm_mask.unsqueeze(-1), masked_refs)
        batch_pointers = torch.arange(0, num_scopes, device=cast_to).view(-1, 1, 1) * torch.ones_like(token_padding_mask)
        batch_pointers = batch_pointers[lm_mask]

        # is_goal = (dense_batch[:, :, :, -1] == goal_id).all(dim=-1) & tree_padding_mask
        # scope_attention_mask = (~is_goal & tree_padding_mask).unsqueeze(-2).expand(-1, most_trees, -1)
        # diag_mask = torch.eye(most_trees, dtype=torch.bool, device=cast_to).unsqueeze(0).expand(num_scopes, -1, -1)
        # tree_attention_mask = scope_attention_mask | (diag_mask & tree_padding_mask.unsqueeze(-1))
        return (dense_batch.permute(-1, 0, 1, 2),
                token_attention_mask,
                tree_padding_mask,
                # tree_attention_mask,
                edge_index,
                gold_labels,
                lm_mask,
                mask_values,
                batch_pointers)
    return collator


class Sampler:
    def __init__(self, data: list[TokenizedFile], max_scope_size: int, max_type_size: int, max_db_index: int):
        def filter_tree(tree: TokenizedTree) -> bool:
            return len(tree) <= max_type_size and all([index < max_db_index for nt, index, _, _ in tree if nt == 4])

        def filter_hole(hole: tuple[TokenizedTree, list[int]]) -> bool:
            tree, goal = hole
            return filter_tree(tree) and any(index != -1 for index in goal)

        def filter_file(file: TokenizedFile) -> bool:
            scope, holes = file
            return len(scope) <= max_scope_size and all(map(filter_tree, scope)) and len(holes) > 0

        filtered = [(scope, [hole for hole in holes if filter_hole(hole)]) for scope, holes in data]
        self.filtered = [file for file in filtered if filter_file(file)]
        print(f'Kept {len(self.filtered)} of {len(data)} files,'
              f' and {sum(len(hs) for _, hs in self.filtered)} of {sum(len(hs) for _, hs in data)} holes.')
        self.hole_counts = [len(holes) for _, holes in self.filtered]
        self.max_types = max_scope_size

    def itersize(self, batch_size_s: int, batch_size_h: int) -> int:
        return ceil(sum([ceil(len(holes)/batch_size_h) for _, holes in self.filtered])/batch_size_s)

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
