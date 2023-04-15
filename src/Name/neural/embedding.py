from __future__ import annotations

import pdb

import torch
from torch import Tensor
from .utils import pad_sequence, Swish
from torch.nn import Module, Parameter, Embedding, Conv1d, Sequential
from torch.nn.functional import linear
from torch.nn.init import normal_ as normal
from torch.nn.utils.parametrizations import orthogonal


class BinaryPathEncoder(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.primitives: Parameter = Parameter(normal(torch.empty(2, dim, dim)))
        # todo: change the identity depending on how the positional encodings are used
        self.identity: Parameter = Parameter(torch.ones(dim).unsqueeze(0), requires_grad=False)
        self.dim: int = dim
        self.precomputed: tuple[Tensor, Tensor] | None = None

    def embed_positions(self, positions: list[int]) -> tuple[Tensor, Tensor]:
        # todo: this can be made much more efficient by reusing subsequence maps
        word_seq = [torch.tensor(self.node_pos_to_path(pos), device=self.primitives.device, dtype=torch.long)
                    if pos > 0 else torch.tensor([])
                    for pos in positions]
        word_ten = pad_sequence(word_seq, padding_value=2)
        maps = self.identity.repeat(len(positions), 1)
        for depth in range(word_ten.shape[1]):
            maps[word_ten[:, depth] == 0] = linear(maps[word_ten[:, depth] == 0], self.primitives[0])
            maps[word_ten[:, depth] == 1] = linear(maps[word_ten[:, depth] == 1], self.primitives[1])
        return torch.tensor(list(positions), device=self.primitives.device, dtype=torch.long), maps

    def precompute(self, positions: int | list[int]) -> None:
        if isinstance(positions, int):
            positions = sorted(set(range(1, positions + 1)))  # right inclusive dense
        self.precomputed = self.embed_positions(positions)

    def forward(self, positions: Tensor) -> Tensor:
        indices = torch.ravel(positions)
        embeddings = torch.index_select(input=self.precomputed[1], dim=0, index=indices)
        return embeddings.view(*positions.shape, self.dim)

    @staticmethod
    def node_pos_to_path(idx: int) -> list[int]:
        return [] if idx == 1 else [idx % 2] + BinaryPathEncoder.node_pos_to_path(idx // 2)

    @staticmethod
    def orthogonal(dim: int) -> BinaryPathEncoder:
        return orthogonal(BinaryPathEncoder(dim), name='primitives', orthogonal_map='matrix_exp')  # type: ignore


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

    def forward(self, positions: Tensor) -> Tensor:
        indices = torch.ravel(positions)
        embeddings = torch.index_select(input=self._positional_embeddings, dim=0, index=indices)
        return embeddings.view(*positions.shape, self.dim)


class TokenEncoder(Module):
    def __init__(self,
                 num_ops: int,
                 num_leaves: int,
                 dim: int,
                 max_scope_size: int = 1000):
        super(TokenEncoder, self).__init__()
        self.type_encoder = Embedding(num_embeddings=4, embedding_dim=dim)
        self.op_encoder = Embedding(num_embeddings=num_ops, embedding_dim=dim)
        self.leaf_encoder = Embedding(num_embeddings=num_leaves, embedding_dim=dim)
        self.path_encoder = BinaryPathEncoder.orthogonal(dim)
        self.reference_encoder = Embedding(max_scope_size + 2, dim, padding_idx=0)
        # todo: non-compositional, maybe replace with URNN
        self.db_encoder = SequentialPositionEncoder(dim, freq=50)
        self.channel_conv = Sequential(Conv1d(in_channels=4, out_channels=8, kernel_size=1),
                                       Swish(),
                                       Conv1d(in_channels=8, out_channels=1, kernel_size=1))

    def forward(self, dense_batch: Tensor) -> Tensor:
        token_types, token_values, node_positions, tree_positions = dense_batch
        type_embeddings = self.type_encoder(token_types)
        unique_paths, inverse = node_positions.unique(return_inverse=True)
        self.path_encoder.precompute(unique_paths.cpu().tolist())
        path_embeddings = self.path_encoder(inverse)
        ground_embeddings = self.reference_encoder(tree_positions.clamp(0))

        token_value_embeddings = torch.zeros_like(ground_embeddings)

        op_mask = token_types == 0
        leaf_mask = token_types == 1
        ref_mask = token_types == 2
        db_mask = token_types == 3
        goal_mask = tree_positions == -1

        token_value_embeddings[op_mask] = self.op_encoder(token_values[op_mask])
        token_value_embeddings[leaf_mask] = self.leaf_encoder(token_values[leaf_mask])
        token_value_embeddings[ref_mask] = self.reference_encoder(token_values[ref_mask])
        token_value_embeddings[db_mask] = self.db_encoder(token_values[db_mask])
        ground_embeddings[goal_mask] = self.reference_encoder.weight[-1]

        pre_fusion = torch.stack((type_embeddings, token_value_embeddings, path_embeddings, ground_embeddings), -2)
        (num_scopes, num_types, num_tokens, _, _) = pre_fusion.shape
        pre_fusion = pre_fusion.flatten(0, 2)
        return self.channel_conv(pre_fusion).squeeze(-2).view(num_scopes, num_types, num_tokens, -1)
