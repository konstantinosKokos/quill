import pdb

from ..data.tokenization import TokenizedAST, TokenizedFile

from typing import NamedTuple, Iterator, TypeVar

import torch
from torch import Tensor
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence

from math import ceil
from random import sample

from itertools import groupby


def scope_causal_mask(
        num_entries: int,
        hole_to_scope: Tensor,
        allow_self_loops: bool) -> Tensor:
    hole_to_scope = hole_to_scope.unsqueeze(-1)
    if not allow_self_loops:
        hole_to_scope -= 1
    return hole_to_scope.ge(torch.arange(num_entries))


class BatchedASTs(NamedTuple):
    tokens:         Tensor
    padding_mask:   Tensor
    reference_mask: Tensor

    @property
    def num_trees(self) -> int: return self.tokens.size(0)


class Batch(NamedTuple):
    dense_scopes:       BatchedASTs
    dense_holes:        BatchedASTs
    edge_index:         Tensor
    scope_to_batch:     Tensor
    scope_positions:    Tensor
    hole_positions:     Tensor
    holes_to_batch:     Tensor
    scope_sort:         Tensor
    premises:           Tensor


class Collator:
    def __init__(self, device: str, pad_value: int, allow_self_loops: bool):
        self.device = device
        self.pad_value = pad_value
        self.allow_self_loops = allow_self_loops

    def tensor(self, xs) -> Tensor:
        return torch.tensor(xs, device=self.device, dtype=torch.long)

    def pad_ast(self, tree: TokenizedAST, to: int) -> Tensor:
        return pad(self.tensor(tree), pad=(0, 0, 0, to - len(tree)), mode='constant', value=self.pad_value)

    def pad_asts(self, trees: list[TokenizedAST], to: int) -> Tensor:
        return pad_sequence([self.pad_ast(tree, to) for tree in trees], batch_first=True, padding_value=self.pad_value)

    def pad_ast_seqs(self, xss: list[list[TokenizedAST]]) -> Tensor:
        max_len = max(len(x) for xs in xss for x in xs)
        return self.pad_asts([x for xs in xss for x in xs], to=max_len)

    def token_mask(self, xs: Tensor) -> Tensor:
        return (xs != self.pad_value).any(dim=-1)

    @staticmethod
    def reference_mask(xs: Tensor) -> Tensor:
        return (xs[:, :, 0] == 3) & (xs[:, :, 1] != -1)

    @property
    def offset(self):
        return -int(not self.allow_self_loops)

    def __call__(self, files: list[TokenizedFile]) -> Batch:
        # lengths and sizes
        scope_lens = [len(file.scope_asts) for file in files]
        num_holes = [len(file.hole_asts) for file in files]

        # dense scopes and holes
        scope_asts = self.pad_ast_seqs([file.scope_asts for file in files])
        hole_asts = self.pad_ast_seqs([file.hole_asts for file in files])
        scope_to_batch = self.tensor(
            [batch_id for batch_id, scope_len in enumerate(scope_lens) for _ in range(scope_len)])
        holes_to_batch = self.tensor(
            [batch_id for batch_id, nh in enumerate(num_holes) for _ in range(nh)])
        scope_positions = self.tensor([i for file in files for i in range(len(file.scope_asts))])
        hole_positions = self.tensor([i for file in files for i in file.hole_to_scope])

        # edge index and ground truth
        src_index, tgt_index, premise_selection = [], [], []
        for batch_id, file in enumerate(files):
            src_offset = sum(scope_lens[:batch_id])
            hole_offset = sum(num_holes[:batch_id])
            for hole_idx, defined_at in enumerate(file.hole_to_scope):
                src_index += list(range(src_offset, src_offset + defined_at))
                tgt_index += [hole_offset + hole_idx] * defined_at
                premise_selection += [entry in file.premises[hole_idx] for entry in range(defined_at)]
        edge_index = torch.stack((self.tensor(src_index), self.tensor(tgt_index)))
        premises = self.tensor(premise_selection)

        # references and offsets
        scope_ref_offsets = self.tensor([sum(scope_lens[:scope_to_batch[entry]])
                                         for entry in range(len(scope_asts))])
        hole_ref_offsets = self.tensor([sum(scope_lens[:holes_to_batch[hole]])
                                        for hole in range(len(hole_asts))])
        scope_ref_mask = self.reference_mask(scope_asts)
        hole_ref_mask = self.reference_mask(hole_asts)
        scope_ref_offsets = torch.where(scope_ref_mask, scope_ref_offsets.unsqueeze(-1), 0)
        scope_asts[:, :, 1] += scope_ref_offsets
        hole_ref_offsets = torch.where(hole_ref_mask, hole_ref_offsets.unsqueeze(-1), 0)
        hole_asts[:, :, 1] += hole_ref_offsets

        # topological sort
        topo_sort = self.tensor([rank for file in files for rank in file.entry_sort])

        return Batch(
            dense_scopes=BatchedASTs(
                tokens=scope_asts,
                padding_mask=self.token_mask(scope_asts),
                reference_mask=scope_ref_mask
            ),
            dense_holes=BatchedASTs(
                tokens=hole_asts,
                padding_mask=self.token_mask(hole_asts),
                reference_mask=hole_ref_mask
            ),
            edge_index=edge_index,
            scope_positions=scope_positions,
            hole_positions=hole_positions,
            scope_to_batch=scope_to_batch,
            holes_to_batch=holes_to_batch,
            scope_sort=topo_sort,
            premises=premises,
        )


def filter_data(files: list[TokenizedFile],
                max_scope_size: int,
                max_ast_len: int) -> Iterator[TokenizedFile]:

    for file in files:
        if (len(file.hole_asts)
                and len(file.scope_asts) <= max_scope_size
                and max(len(ast) for ast in file.hole_asts) <= max_ast_len
                and max(len(ast) for ast in file.scope_asts) <= max_ast_len):
            yield file


_T = TypeVar('_T')


def make_permutation(of_size: int) -> list[int]:
    return sample(list(range(of_size)), of_size)


def select(xs: list[_T], permutation: list[int]) -> list[_T]:
    return [xs[i] for i in permutation]


def sublists(xs: list[_T], of_size: int) -> list[list[_T]]:
    return [xs[i:i+of_size] for i in range(0, len(xs), of_size)]


class Sampler:
    def __init__(self, files: list[TokenizedFile]):
        self.files = files

    def itersize(self, batch_size_s: int, batch_size_h: int) -> int:
        return ceil(sum([ceil(len(file.hole_asts)/batch_size_h)
                         for file in self.files]) / batch_size_s)

    @property
    def hole_counts(self) -> list[int]:
        return [len(file.hole_asts) for file in self.files]

    def iter(self, batch_size_s: int, batch_size_h: int) -> Iterator[list[TokenizedFile]]:
        permutation_indices = [make_permutation(nhs) for nhs in self.hole_counts]
        epoch_indices = [(s_idx, selection)
                         for s_idx, permutation in enumerate(permutation_indices)
                         for selection in sublists(permutation, batch_size_h)]
        epoch_permutation = make_permutation(len(epoch_indices))
        permuted_epoch = select(epoch_indices, epoch_permutation)

        for batch_indices in sublists(permuted_epoch, batch_size_s):
            condensed = [(scope_id, [hid for _, holes in items for hid in holes])
                         for scope_id, items
                         in groupby(sorted(batch_indices, key=lambda x: x[0]), key=lambda x: x[0])]
            yield [TokenizedFile(
                file=self.files[file_id].file,
                backrefs=self.files[file_id].backrefs,
                entry_sort=self.files[file_id].entry_sort,
                scope_asts=self.files[file_id].scope_asts,
                hole_asts=select(self.files[file_id].hole_asts, hole_ids),
                hole_to_scope=select(self.files[file_id].hole_to_scope, hole_ids),
                premises=select(self.files[file_id].premises, hole_ids)) for file_id, hole_ids in condensed]
