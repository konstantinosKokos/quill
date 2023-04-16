from __future__ import annotations

import torch
from torch import Tensor
from .utils import pad_sequence
from torch.nn import Module, Parameter, Embedding, Conv1d, Sequential, ReLU
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

    def embed_positions(self, positions: list[int]) -> Tensor:
        # todo: this can be made much more efficient by reusing subsequence maps
        word_seq = [torch.tensor(self.node_pos_to_path(pos), device=self.primitives.device, dtype=torch.long)
                    if pos > 0 else torch.tensor([])
                    for pos in positions]
        word_ten = pad_sequence(word_seq, padding_value=2)
        maps = self.identity.repeat(len(positions), 1)
        for depth in range(word_ten.shape[1]):
            maps[word_ten[:, depth] == 0] = linear(maps[word_ten[:, depth] == 0], self.primitives[0])
            maps[word_ten[:, depth] == 1] = linear(maps[word_ten[:, depth] == 1], self.primitives[1])
        return maps

    def forward(self, unique: Tensor, mapping: Tensor) -> Tensor:
        embeddings = self.embed_positions(unique.cpu().tolist())
        indices = torch.ravel(mapping)
        return torch.index_select(input=embeddings, dim=0, index=indices).view(*mapping.shape, self.dim)

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
                 max_scope_size: int = 1000,
                 max_db_size: int = 20):
        super(TokenEncoder, self).__init__()
        self.num_leaves = num_leaves
        self.num_ops = num_ops
        self.max_scope_size = max_scope_size
        self.max_db_size = max_db_size
        self.fixed_embeddings = Embedding(num_embeddings=num_ops+num_leaves, embedding_dim=dim)
        self.path_encoder = BinaryPathEncoder.orthogonal(dim)
        self.reference_encoder = Embedding(max_scope_size + 1, dim, padding_idx=0)
        self.db_encoder = SequentialPositionEncoder(dim, freq=max_db_size)
        self.channel_conv = Sequential(Conv1d(in_channels=3, out_channels=6, kernel_size=1),
                                       ReLU(),
                                       Conv1d(in_channels=6, out_channels=1, kernel_size=1))

    def forward(self, dense_batch: Tensor) -> Tensor:
        token_types, token_values, node_positions, tree_positions = dense_batch

        pad_mask = token_types == -1
        sos_mask = token_types == 0
        op_mask = token_types == 1
        leaf_mask = token_types == 2
        ref_mask = token_types == 3
        db_mask = token_types == 4
        goal_mask = tree_positions == -1 & (~pad_mask)

        path_embeddings = self.path_encoder.forward(*node_positions.unique(return_inverse=True))
        tree_embeddings = self.reference_encoder(tree_positions.clamp(min=0, max=self.max_scope_size))
        tree_embeddings[goal_mask] = self.reference_encoder.weight[-1]

        content_embeddings = torch.zeros_like(path_embeddings)
        content_embeddings[sos_mask] = self.fixed_embeddings(token_values[sos_mask])
        content_embeddings[op_mask] = self.fixed_embeddings(token_values[op_mask] + 1)
        content_embeddings[leaf_mask] = self.fixed_embeddings(token_values[leaf_mask] + self.num_ops + 1)
        content_embeddings[ref_mask] = self.reference_encoder(token_values[ref_mask])
        content_embeddings[db_mask] = self.db_encoder(token_values[db_mask])

        out = torch.stack((content_embeddings, path_embeddings, tree_embeddings), -2)
        (num_scopes, num_types, num_tokens, _, _) = out.shape
        out = out.flatten(0, 2)
        return self.channel_conv(out).squeeze(1).view(num_scopes, num_types, num_tokens, -1)
