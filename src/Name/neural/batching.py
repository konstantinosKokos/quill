from typing import NamedTuple, Iterator, TypeVar, Type

import torch
from torch import Tensor, device
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence

from math import ceil
from random import sample
from itertools import groupby

from ..data.tokenization import TokenizedFile, TokenizedAST


class BatchedASTs(NamedTuple):
    tokens:             Tensor
    token_mask:         Tensor
    reference_mask:     Tensor
    reference_ids:      Tensor


class Batch(NamedTuple):
    scope_types:        BatchedASTs
    scope_definitions:  BatchedASTs
    hole_types:         BatchedASTs
    batch_pts:          Tensor
    edge_index:         Tensor
    lemmas:             Tensor | None


class Collator:
    def __init__(self, pad_value: int, mode: Type[list] | Type[set] | Type[object], cast_to: device):
        self.cast_to = cast_to
        self.mode = mode
        self.pad_value = pad_value

    def lemma_pp(self, premises: list[int]) -> list[int]:
        if self.mode == list:
            return premises
        elif self.mode == set:
            return list(set(premises))
        elif self.mode == object:
            return premises[:1]
        raise ValueError

    def tensor(self, xs) -> Tensor:
        return torch.tensor(xs, device=self.cast_to, dtype=torch.long)

    def pad_ast(self, tree: TokenizedAST, to: int) -> Tensor:
        return pad(self.tensor(tree), pad=(0, 0, 0, to - len(tree)), mode='constant', value=self.pad_value)

    def pad_asts(self, trees: list[TokenizedAST], to: int) -> Tensor:
        return pad_sequence([self.pad_ast(tree, to) for tree in trees], padding_value=self.pad_value, batch_first=True)

    def pad_arrays(self, xss: list[list[TokenizedAST]]) -> Tensor:
        max_len = max(len(x) for xs in xss for x in xs)
        return pad_sequence([self.pad_asts(xs, max_len) for xs in xss], padding_value=self.pad_value, batch_first=True)

    def make_token_mask(self, xs: Tensor) -> Tensor:
        return (xs != self.pad_value).any(dim=-1)

    def __call__(self, files: list[TokenizedFile]) -> Batch:
        num_files = len(files)

        # unpacked tokenized files
        _, scopes, holes = zip(*files)
        scope_types: list[list[TokenizedAST]] = [[st for st, _ in scope] for scope in scopes]
        scope_definitions: list[list[TokenizedAST]] = [[sd for _, sd in scope] for scope in scopes]
        hole_types: list[list[TokenizedAST]] = [[h[0] for h in hs] for hs in holes]

        # lengths and sizes
        scope_lens = [len(st) for st in scope_types]
        hole_counts = [len(hs) for hs in hole_types]
        longest_scope = max(scope_lens)
        most_holes = max(hole_counts)

        # scope/hole pair indexing
        source_index = [torch.arange((start := i * longest_scope), start + scope_len).repeat(hole_count)
                        for i, (scope_len, hole_count) in enumerate(zip(scope_lens, hole_counts))]
        target_index = [torch.arange((start := i * most_holes), start + hole_count).repeat_interleave(scope_len)
                        for i, (scope_len, hole_count) in enumerate(zip(scope_lens, hole_counts))]
        edge_index = torch.stack((torch.cat(source_index), torch.cat(target_index))).to(self.cast_to)

        # dense padded batches
        scope_types: Tensor = self.pad_arrays(scope_types)
        scope_definitions: Tensor = self.pad_arrays(scope_definitions)
        hole_types: Tensor = self.pad_arrays(hole_types)

        # reference offsets
        batch_pts = torch.arange(0, num_files, device=self.cast_to).view(-1, 1, 1)
        batch_pts = batch_pts * longest_scope
        scope_type_offsets = scope_types[:, :, :, 1] + batch_pts
        scope_def_offsets = scope_definitions[:, :, :, 1] + batch_pts
        hole_type_offsets = hole_types[:, :, :, 1] + batch_pts
        scope_type_ref_mask = (scope_types[:, :, :, 0] == 3) & (scope_types[:, :, :, 1] != -1)
        scope_def_ref_mask = (scope_definitions[:, :, :, 0] == 3) & (scope_definitions[:, :, :, 1] != -1)
        hole_type_ref_mask = (hole_types[:, :, :, 0] == 3) & (hole_types[:, :, :, 1] != -1)
        scope_type_ref_values = scope_type_offsets[scope_type_ref_mask]
        scope_def_ref_values = scope_def_offsets[scope_def_ref_mask]
        hole_type_ref_values = hole_type_offsets[hole_type_ref_mask]

        # target encoding
        lemmas: Tensor
        if self.mode == list:
            raise NotImplementedError
        lemmas = torch.tensor(
            [i in self.lemma_pp(ps) for _, scope, holes in files for _, _, ps in holes for i in range(len(scope))],
            dtype=torch.bool, device=self.cast_to)

        return Batch(scope_types=
                     BatchedASTs(tokens=scope_types.permute(3, 0, 1, 2),
                                 token_mask=self.make_token_mask(scope_types).flatten(0, 1),
                                 reference_mask=scope_type_ref_mask,
                                 reference_ids=scope_type_ref_values),
                     scope_definitions=
                     BatchedASTs(tokens=scope_definitions.permute(3, 0, 1, 2),
                                 token_mask=self.make_token_mask(scope_definitions).flatten(0, 1),
                                 reference_mask=scope_def_ref_mask,
                                 reference_ids=scope_def_ref_values),
                     hole_types=
                     BatchedASTs(tokens=hole_types.permute(3, 0, 1, 2),
                                 token_mask=self.make_token_mask(hole_types).flatten(0, 1),
                                 reference_mask=hole_type_ref_mask,
                                 reference_ids=hole_type_ref_values),
                     batch_pts=batch_pts,
                     edge_index=edge_index,
                     lemmas=lemmas)


def filter_data(files: list[TokenizedFile],
                max_scope_size: int,
                max_ast_len: int,
                max_db_index: int) -> Iterator[TokenizedFile]:
    def db_check(ast: TokenizedAST) -> bool:
        return not any([token_type in {4, 5} and token_value >= max_db_index for token_type, token_value, _, _ in ast])

    for name, scope, holes in files:
        holes = [(ht, hd, hl) for ht, hd, hl in holes if len(ht) <= max_ast_len and len(hl) and db_check(ht)]
        scope_check = len(scope) <= max_scope_size and all(
            [len(t_ast) <= max_ast_len and db_check(t_ast) and
             len(d_ast) <= max_ast_len and db_check(d_ast)
             for t_ast, d_ast in scope])
        if len(holes) and scope_check:
            yield name, scope, holes


_T = TypeVar('_T')


def permute(xs: list[_T]) -> list[_T]:
    return sample(xs, len(xs))


def select(xs: list[_T], ids: list[int]) -> list[_T]:
    return [xs[i] for i in ids]


def sublists(xs: list[_T], of_size: int) -> list[list[_T]]:
    return [xs[i:i+of_size] for i in range(0, len(xs), of_size)]


class Sampler:
    def __init__(self, data: list[TokenizedFile]):
        self.data = data

    def itersize(self, batch_size_s: int, batch_size_h: int) -> int:
        return ceil(sum([ceil(len(holes)/batch_size_h) for _, _, holes in self.data])/batch_size_s)

    @property
    def hole_counts(self) -> list[int]:
        return [len(hs) for _, _, hs in self.data]

    def iter(self, batch_size_s: int, batch_size_h: int) -> Iterator[list[TokenizedFile]]:
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
            yield [(self.data[scope_idx][0], self.data[scope_idx][1], select(self.data[scope_idx][2], hole_ids))
                   for scope_idx, hole_ids in batch_indices]
