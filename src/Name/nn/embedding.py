from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module, Parameter, Embedding
from torch.nn.functional import linear
from torch.nn.utils.rnn import pad_sequence


class BinaryPathEncoder(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim: int = dim
        self._primitives = Parameter(torch.rand(2, self.dim, self.dim).softmax(dim=-1).cumsum(dim=-1))
        self.identity: Parameter = Parameter(torch.eye(dim).unsqueeze(0))
        self._pos_to_path: dict[int, list[bool]] = {}

    @property
    def hermitian(self) -> Tensor:
        return self._primitives - self._primitives.mH

    @property
    def primitives(self) -> Tensor:
        hermitian = self.hermitian
        return torch.matrix_exp(hermitian)

    def embed_positions(self, positions: list[int]) -> Tensor:
        primitives = self.primitives
        path_words = pad_sequence(
            [torch.tensor(self.pos_to_path(pos), device=self.primitives.device, dtype=torch.long)
             if pos > 0 else torch.empty(0, device=self.primitives.device, dtype=torch.long)
             for pos in positions], padding_value=2, batch_first=True)
        maps = self.identity.repeat(len(positions), 1, 1)

        left_mask = path_words == 0
        right_mask = path_words == 1

        for step in range(path_words.size(1)):
            maps[left_mask[:, step]] = linear(maps[left_mask[:, step]], primitives[0])
            maps[right_mask[:, step]] = linear(maps[right_mask[:, step]], primitives[1])
        return maps

    def forward(self, unique: Tensor) -> Tensor:
        return self.embed_positions(unique.cpu().tolist())

    def pos_to_path(self, idx: int) -> list[int]:
        if idx in self._pos_to_path:
            return self._pos_to_path[idx]
        self._pos_to_path[idx] = [] if idx == 1 else [idx % 2] + self.pos_to_path(idx // 2)
        return self._pos_to_path[idx]


class TokenEmbedding(Module):
    def __init__(self, dim: int, scope_dropout: float):
        super(TokenEmbedding, self).__init__()
        self.dim = dim
        self.scope_dropout = scope_dropout
        self.path_encoder = BinaryPathEncoder(dim=dim)
        self.embeddings = Embedding(num_embeddings=11, embedding_dim=dim)
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
                _ [Abs]
            _ UnaryOp
                8 [deBruijn]
            9 [oos]
            10 [mask]
        """

    def forward(self, dense_batch: Tensor) -> tuple[Tensor, Tensor]:
        token_types, token_values, node_positions = dense_batch

        sos_mask = token_types == 0
        bop_mask = token_types == 1
        nop_mask = token_types == 2
        scope_mask = (token_types == 3)
        oos_mask = scope_mask & (token_values == -1)
        if self.training and self.scope_dropout > 0:
            drop_mask = torch.rand(scope_mask.size(), device=oos_mask.device) < self.scope_dropout
            oos_mask = scope_mask & (drop_mask | oos_mask)
        db_mask = (token_types == 4)

        unique_paths, inverse = node_positions.unique(return_inverse=True)
        positional_encodings = self.path_encoder.forward(unique_paths)

        content_embeddings = torch.zeros(
            size=(*token_types.size(), self.dim),
            dtype=positional_encodings.dtype,
            device=token_types.device)
        content_embeddings[sos_mask] = self.embeddings.weight[0]
        content_embeddings[bop_mask] = self.embeddings.forward(token_values[bop_mask] + 1)
        content_embeddings[nop_mask] = self.embeddings.forward(token_values[nop_mask] + 5)
        content_embeddings[db_mask] = self.embeddings.weight[8]
        content_embeddings[oos_mask] = self.embeddings.weight[10]
        return content_embeddings, positional_encodings[inverse, :]
