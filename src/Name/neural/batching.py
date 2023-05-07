import torch
from torch import Tensor, device
from ..data.tokenization import TokenizedSample, TokenizedFile, TokenizedTree
from .utils import sublists, permute, select, pad_sequence
from math import ceil
from typing import Iterator, Callable, NamedTuple
from itertools import groupby
from torch.nn.functional import pad as _pad
from random import random
from itertools import takewhile


class Batch(NamedTuple):
    dense_scopes:       Tensor
    dense_goals:        Tensor
    scope_token_mask:   Tensor
    scope_tree_mask:    Tensor
    goal_token_mask:    Tensor
    gold_lemmas:        Tensor | None
    ref_mask:           Tensor
    masked_values:      Tensor
    batch_pts:          Tensor


def filter_unreferenced(file: TokenizedFile, negative_sampling: float) -> TokenizedFile:
    scope, goals = file

    def refers_to(tree: TokenizedTree, excluding: set[int]) -> set[int]:
        direct = {tv for tt, tv, _, _ in tree if tt == 3 and tv not in excluding}
        excluding |= direct
        return {indirect
                for reference in direct
                for indirect in refers_to(scope[reference], excluding)} | direct

    def rename(tree: TokenizedTree, using: dict[int, int]) -> TokenizedTree:
        return [(tt, using[tv] if tt == 3 else tv, np, using[tp]) for tt, tv, np, tp in tree]

    all_references = set.union(*[refers_to(tree, set()) for tree in [*scope, *[goal_type for goal_type, _ in goals]]])
    all_references |= {ref for _, names_used in goals for ref in names_used}
    removed = [idx for idx in range(len(scope)) if idx not in all_references or random() > negative_sampling]
    renames = {kept: kept - sum(map(lambda _: 1, takewhile(lambda r: r < kept, removed))) for kept in range(len(scope))}
    renames[-1] = -1
    return ([rename(tree, renames) for idx, tree in enumerate(scope) if idx not in removed],
            [(rename(goal_type, renames), [renames[ref] for ref in names_used]) for goal_type, names_used in goals])


def make_collator(cast_to: device = device('cpu'),
                  pad_value: int = -1,
                  goal_id: int = -1) -> Callable[[list[TokenizedSample], float, float], Batch]:
    def _longt(xs) -> Tensor:
        return torch.tensor(xs, device=cast_to, dtype=torch.long)

    def _boolt(xs) -> Tensor:
        return torch.tensor(xs, device=cast_to, dtype=torch.bool)

    def pad_tree(tree: TokenizedTree, to: int) -> Tensor:
        return _pad(_longt(tree), pad=(0, 0, 0, to - len(tree)), mode='constant', value=pad_value)

    def pad_goal(goal: list[int], to: int) -> Tensor:
        return _pad(_longt(goal), pad=(0, to - len(goal)), mode='constant', value=goal[-1])

    def pad_seq(file: list[Tensor]) -> Tensor:
        return pad_sequence(file, padding_value=pad_value)

    def collator(samples: list[TokenizedSample], lm_chance: float, negative_sampling: float) -> Batch:
        num_scopes = len(samples)
        scopes, goals = zip(*[(scope, holes) for scope, holes in samples])
        most_entries = max(len(scope) for scope in scopes)
        most_holes = max(len(holes) for holes in goals)
        most_lemmas = max(max(len(goal) for _, goal in holes) for holes in goals)
        max_entry_len = max(max(len(tree) for tree in scope) for scope in scopes)
        max_goal_len = max(max(len(goal_type) for goal_type, _ in holes) for holes in goals)

        # dense input
        dense_scopes = pad_seq([pad_seq([pad_tree(tree, max_entry_len) for tree in scope]) for scope in scopes])
        dense_goals = pad_seq([pad_seq([pad_tree(tree, max_goal_len) for tree, _ in holes]) for holes in goals])

        # lemma output
        true_indices = torch.stack([torch.stack([pad_goal(goal, most_lemmas) for _, goal in holes]) for holes in goals])
        gold_lemmas = torch.zeros(num_scopes, most_holes, most_entries, dtype=torch.bool, device=cast_to)
        gold_lemmas.scatter_(2, true_indices.long(), True)

        # masks
        scope_token_mask = (dense_scopes != pad_value).any(dim=-1)
        scope_tree_mask = scope_token_mask.any(dim=-1)
        goal_token_mask = (dense_goals != pad_value).any(dim=-1)

        # LM masking
        ref_mask = (dense_scopes[:, :, :, 0] == 3) & (dense_scopes[:, :, :, 1] != -1)
        lm_mask = (torch.rand_like(scope_token_mask, dtype=torch.float) < lm_chance) & ref_mask
        masked_refs = dense_scopes[lm_mask]
        masked_values = dense_scopes[lm_mask][:, 1]
        masked_refs[:, 0] = 5
        masked_refs[:, 1] = 0
        dense_scopes.masked_scatter_(lm_mask.unsqueeze(-1), masked_refs)
        batch_pts = torch.arange(0, num_scopes, device=cast_to).view(-1, 1, 1)
        batch_pts = batch_pts * torch.ones_like(scope_token_mask)
        batch_pts = batch_pts[lm_mask]

        return Batch(dense_scopes=dense_scopes,
                     dense_goals=dense_goals,
                     scope_token_mask=scope_token_mask,
                     scope_tree_mask=scope_tree_mask,
                     goal_token_mask=goal_token_mask,
                     gold_lemmas=gold_lemmas,
                     ref_mask=ref_mask,
                     masked_values=masked_values,
                     batch_pts=batch_pts)
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
