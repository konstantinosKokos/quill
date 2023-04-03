from __future__ import annotations

import torch
from .utils import pad_sequence
from torch.nn import Module, Parameter
from torch.nn.functional import linear
from torch.nn.init import normal_ as normal
from torch.nn.utils.parametrizations import orthogonal


class BinaryTreeEncoder(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.primitives: Parameter = Parameter(normal(torch.empty(2, dim, dim)))
        # todo: change the identity depending on how the positional encodings are used
        self.identity: Parameter = Parameter(torch.eye(dim, dim).unsqueeze(0), requires_grad=False)
        self.dim: int = dim
        self.precomputed: tuple[torch.Tensor, torch.Tensor] | None = None

    def embed_positions(self, positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        # todo: this can be made much more efficient by reusing subsequence maps
        word_seq = [torch.tensor(self.node_pos_to_path(pos), device=self.primitives.device, dtype=torch.long)
                    for pos in positions]
        word_ten = pad_sequence(word_seq, padding_value=2)
        maps = self.identity.repeat(len(positions), 1, 1)
        for depth in range(word_ten.shape[1]):
            maps[word_ten[:, depth] == 0] = linear(maps[word_ten[:, depth] == 0], self.primitives[0])
            maps[word_ten[:, depth] == 1] = linear(maps[word_ten[:, depth] == 1], self.primitives[1])
        return torch.tensor(list(positions), device=self.primitives.device, dtype=torch.long), maps

    def precompute(self, positions: int | list[int]) -> None:
        if isinstance(positions, int):
            positions = sorted(set(range(1, positions + 1)))  # right inclusive dense
        self.precomputed = self.embed_positions(positions)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        indices = torch.ravel(positions)
        embeddings = torch.index_select(input=self.precomputed[1], dim=0, index=indices)
        return embeddings.view(*positions.shape, self.dim, self.dim)

    @staticmethod
    def node_pos_to_path(idx: int) -> list[int]:
        return [] if idx == 1 else [idx % 2] + BinaryTreeEncoder.node_pos_to_path(idx // 2)

    @staticmethod
    def orthogonal(dim: int) -> BinaryTreeEncoder:
        return orthogonal(BinaryTreeEncoder(dim), name='primitives', orthogonal_map='matrix_exp')  # type: ignore


class SequentialPositionEncoder(Module):
    def __init__(self, dim: int, freq: int = 1000):
        super(SequentialPositionEncoder, self).__init__()
        self.dim = dim
        self.freq = freq
        pe = torch.zeros(freq, dim)
        position = torch.arange(0, freq, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) *
                             - (torch.log(torch.tensor(freq, dtype=torch.float)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self._positional_embeddings = Parameter(pe, requires_grad=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        indices = torch.ravel(positions)
        embeddings = torch.index_select(input=self._positional_embeddings, dim=0, index=indices)
        return embeddings.view(*positions.shape, self.dim)
