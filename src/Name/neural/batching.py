import torch
from torch import Tensor, device
from ..data.tokenization import TokenizedSample, TokenizedFile, TokenizedTree
from math import ceil
from typing import Iterator, Callable, NamedTuple, TypeVar
from itertools import groupby
from torch.nn.functional import pad as _pad
from torch.nn.utils.rnn import pad_sequence
from random import sample


_T = TypeVar('_T')


def permute(xs: list[_T]) -> list[_T]:
    return sample(xs, len(xs))


def select(xs: list[_T], ids: list[int]) -> list[_T]:
    return [xs[i] for i in ids]


def sublists(xs: list[_T], of_size: int) -> list[list[_T]]:
    return [xs[i:i+of_size] for i in range(0, len(xs), of_size)]


class Batch(NamedTuple):
    dense_scopes:       Tensor
    dense_goals:        Tensor
    scope_token_mask:   Tensor
    scope_tree_mask:    Tensor
    goal_token_mask:    Tensor
    edge_index:         Tensor
    scope_ref_mask:     Tensor
    scope_ref_ids:      Tensor
    goal_ref_mask:      Tensor
    goal_ref_ids:       Tensor
    batch_pts:          Tensor | None
    gold_lemmas:        Tensor | None
    lm_mask:            Tensor | None
    masked_values:      Tensor | None


def make_collator(cast_to: device = device('cpu'),
                  pad_value: int = -1) -> Callable[[list[TokenizedSample], float], Batch]:
    def _longt(xs) -> Tensor:
        return torch.tensor(xs, device=cast_to, dtype=torch.long)

    def pad_tree(tree: TokenizedTree, to: int) -> Tensor:
        return _pad(_longt(tree), pad=(0, 0, 0, to - len(tree)), mode='constant', value=pad_value)

    def pad_array(file: list[Tensor]) -> Tensor:
        return pad_sequence(file, padding_value=pad_value, batch_first=True)

    def collator(samples: list[TokenizedSample], lm_chance: float) -> Batch:
        num_scopes = len(samples)
        scopes, holes = zip(*[(scope, holes) for scope, holes in samples])
        scope_lens = [len(scope) for scope in scopes]
        hole_counts = [len(hole) for hole in holes]
        most_entries = max(scope_lens)
        most_holes = max(hole_counts)
        max_entry_len = max(max(len(tree) for tree in scope) for scope in scopes)
        max_goal_len = max(max(len(goal_type) for goal_type, _ in holes) for holes in holes)

        # dense input
        dense_scopes = pad_array([pad_array([pad_tree(tree, max_entry_len) for tree in scope]) for scope in scopes])
        dense_goals = pad_array([pad_array([pad_tree(tree, max_goal_len) for tree, _ in holes]) for holes in holes])

        # lemma input/output
        source_index = [torch.arange((start := i * most_entries), start + scope_len).repeat(hole_count) for
                        i, (scope_len, hole_count) in enumerate(zip(scope_lens, hole_counts))]
        target_index = [torch.arange((start := i * most_holes), start + hole_count).repeat_interleave(scope_len)
                        for i, (scope_len, hole_count) in enumerate(zip(scope_lens, hole_counts))]
        edge_index = torch.stack((torch.cat(source_index), torch.cat(target_index)))
        gold_lemmas = torch.tensor([i in refs
                                    for scope, holes in samples
                                    for _, refs in holes
                                    for i in range(len(scope))], dtype=torch.bool, device=cast_to)

        # padding masks
        scope_token_mask = (dense_scopes != pad_value).any(dim=-1)
        scope_tree_mask = scope_token_mask.any(dim=-1)
        goal_token_mask = (dense_goals != pad_value).any(dim=-1)

        # reference handling
        batch_pts = (torch.arange(0, num_scopes, device=cast_to)).view(-1, 1, 1)
        batch_offsets = batch_pts * most_entries
        scope_offsets = dense_scopes[:, :, :, 1] + batch_offsets
        scope_ref_mask = (dense_scopes[:, :, :, 0] == 3) & (dense_scopes[:, :, :, 1] != -1)
        goal_ref_mask = (dense_goals[:, :, :, 0] == 3) & (dense_goals[:, :, :, 1] != -1)
        goal_offsets = dense_goals[:, :, :, 1] + batch_offsets
        goal_ref_values = goal_offsets[goal_ref_mask]

        # LM masking
        lm_mask = (torch.rand_like(scope_ref_mask, dtype=torch.float) < lm_chance) & scope_ref_mask
        masked_refs = dense_scopes[lm_mask]
        masked_values = dense_scopes[lm_mask][:, 1]
        masked_refs[:, 0] = 5
        masked_refs[:, 1] = 0
        dense_scopes.masked_scatter_(lm_mask.unsqueeze(-1), masked_refs)
        batch_pts = batch_pts * torch.ones_like(scope_token_mask)
        batch_pts = batch_pts[lm_mask]
        scope_ref_mask = scope_ref_mask & (~lm_mask)

        scope_ref_values = scope_offsets[scope_ref_mask]

        return Batch(dense_scopes=dense_scopes.permute(3, 0, 1, 2),
                     dense_goals=dense_goals.permute(3, 0, 1, 2),
                     scope_token_mask=scope_token_mask.flatten(0, 1),
                     scope_tree_mask=scope_tree_mask,
                     goal_token_mask=goal_token_mask.flatten(0, 1),
                     gold_lemmas=gold_lemmas,
                     lm_mask=lm_mask,
                     masked_values=masked_values,
                     batch_pts=batch_pts,
                     edge_index=edge_index,
                     scope_ref_mask=scope_ref_mask,
                     scope_ref_ids=scope_ref_values,
                     goal_ref_mask=goal_ref_mask,
                     goal_ref_ids=goal_ref_values)
    return collator


class Sampler:
    def __init__(self, data: list[TokenizedFile],
                 max_scope_entries: int,
                 max_scope_size: int,
                 max_goal_size: int,
                 max_db_index: int):
        def filter_tree(tree: TokenizedTree, up_to: int) -> bool:
            return len(tree) <= up_to and all([index < max_db_index for nt, index, _, _ in tree if nt == 4])

        def filter_hole(hole: tuple[TokenizedTree, list[int]]) -> bool:
            tree, goal = hole
            return filter_tree(tree, max_goal_size) and any(index != -1 for index in goal)

        def filter_file(file: TokenizedFile) -> bool:
            scope, holes = file
            return all((len(scope) <= max_scope_entries,
                        all(filter_tree(tree, max_scope_size) for tree in scope),
                        len(holes)))

        filtered = [(scope, [hole for hole in holes if filter_hole(hole)]) for scope, holes in data]
        self.filtered = [file for file in filtered if filter_file(file)]
        print(f'Kept {len(self.filtered)} of {len(data)} files,'
              f' and {sum(len(hs) for _, hs in self.filtered)} of {sum(len(hs) for _, hs in data)} holes.')
        self.hole_counts = [len(holes) for _, holes in self.filtered]
        self.max_types = max_scope_entries

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
