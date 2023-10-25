from __future__ import annotations

import pdb

import torch
from torch import Tensor
from torch.nn import Module, Parameter, Embedding
from torch.nn.functional import linear
from torch.nn.init import normal_ as normal
from torch.nn.utils.parametrizations import orthogonal as orth_param
from torch.nn.utils.rnn import pad_sequence


class BinaryPathEncoder(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.primitives: Parameter = Parameter(normal(torch.empty(2, dim, dim)))
        self.identity: Parameter = Parameter(torch.ones(dim).unsqueeze(0))
        self._pos_to_path: dict[int, list[bool]] = {}
        self.dim: int = dim

    def embed_positions(self, positions: list[int]) -> Tensor:
        word_seq = [torch.tensor(self.pos_to_path(pos), device=self.primitives.device, dtype=torch.long)
                    if pos > 0 else torch.empty(0, device=self.primitives.device, dtype=torch.long)
                    for pos in positions]
        word_ten = pad_sequence(word_seq, padding_value=2, batch_first=True)
        maps = self.identity.repeat(len(positions), 1)
        for depth in range(word_ten.shape[1]):
            maps[word_ten[:, depth] == 0] = linear(maps[word_ten[:, depth] == 0], self.primitives[0])
            maps[word_ten[:, depth] == 1] = linear(maps[word_ten[:, depth] == 1], self.primitives[1])
        return maps

    def forward(self, unique: Tensor) -> Tensor:
        return self.embed_positions(unique.cpu().tolist())

    def apply_mapping(self, embeddings: Tensor, mapping: Tensor) -> Tensor:
        indices = torch.ravel(mapping)
        return torch.index_select(input=embeddings, dim=0, index=indices).view(*mapping.shape, self.dim)

    def pos_to_path(self, idx: int) -> list[int]:
        if idx in self._pos_to_path:
            return self._pos_to_path[idx]
        self._pos_to_path[idx] = [] if idx == 1 else [idx % 2] + self.pos_to_path(idx // 2)
        return self._pos_to_path[idx]

    @staticmethod
    def orthogonal(dim: int) -> BinaryPathEncoder:
        return orth_param(BinaryPathEncoder(dim), name='primitives', orthogonal_map='cayley')  # type: ignore


class SequentialPositionEncoder(Module):
    def __init__(self, dim: int, freq: int):
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


class TokenEmbedding(Module):
    def __init__(self,
                 dim: int):
        super(TokenEmbedding, self).__init__()
        self.dim = dim
        self.path_encoder = BinaryPathEncoder.orthogonal(dim // 2)
        self.embeddings = Embedding(num_embeddings=11, embedding_dim=dim // 2)
        """
        Embedding map:
            0 [SOS]
            _ BinOp
                1 [PiSimple]
                2 [PiDependent]
                3 [Lambda]
                4 [Application]
            _ NullOp
                5 [Sort]
                6 [Level]
                7 [Literal]
                8 [Abs]
            _ UnaryOp
                8 [deBruijn]
            9 [oos]
            10 [mask]
        """

    def forward(self, dense_batch: Tensor) -> Tensor:
        token_types, token_values, node_positions = dense_batch

        sos_mask = token_types == 0
        bop_mask = token_types == 1
        nop_mask = token_types == 2
        # ref_mask = (token_types == 3) & (token_values != -1)
        oos_mask = (token_types == 3) & (token_values == -1)
        db_mask = (token_types == 4)

        unique_paths, inverse = node_positions.unique(return_inverse=True)
        db_paths = torch.bucketize(token_values[db_mask], unique_paths)
        path_embeddings = self.path_encoder.forward(unique_paths)
        positional_encodings = self.path_encoder.apply_mapping(path_embeddings, inverse)

        content_embeddings = torch.zeros_like(positional_encodings)
        content_embeddings[sos_mask] = self.embeddings.weight[0]
        content_embeddings[bop_mask] = self.embeddings.forward(token_values[bop_mask] + 1)
        content_embeddings[nop_mask] = self.embeddings.forward(token_values[nop_mask] + 5)
        content_embeddings[db_mask] = self.path_encoder.apply_mapping(path_embeddings, db_paths)
        content_embeddings[oos_mask] = self.embeddings.weight[10]

        return torch.cat((content_embeddings, positional_encodings), -1)
