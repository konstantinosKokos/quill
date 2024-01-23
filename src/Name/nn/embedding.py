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
        path_words = [torch.tensor(self.pos_to_path(pos), device=self.primitives.device, dtype=torch.long)
                      if pos > 0 else torch.empty(0, device=self.primitives.device, dtype=torch.long)
                      for pos in positions]
        path_words = pad_sequence(path_words, padding_value=2, batch_first=True)
        maps = self.identity.repeat(len(positions), 1, 1)
        for depth in range(path_words.shape[1]):
            maps[path_words[:, depth] == 0] = linear(maps[path_words[:, depth] == 0], primitives[0])
            maps[path_words[:, depth] == 1] = linear(maps[path_words[:, depth] == 1], primitives[1])
        return maps

    def forward(self, unique: Tensor) -> Tensor:
        return self.embed_positions(unique.cpu().tolist())

    def apply_mapping(
            self,
            maps: Tensor,
            embeddings: Tensor,
            index: Tensor) -> Tensor:
        flat_embeddings = embeddings.flatten(0, -2)
        flat_index = torch.ravel(index)
        return torch.einsum('bij,bj->bi', maps[flat_index, :], flat_embeddings)

    def pos_to_path(self, idx: int) -> list[int]:
        if idx in self._pos_to_path:
            return self._pos_to_path[idx]
        self._pos_to_path[idx] = [] if idx == 1 else [idx % 2] + self.pos_to_path(idx // 2)
        return self._pos_to_path[idx]


class TokenEmbedding(Module):
    def __init__(self,
                 dim: int):
        super(TokenEmbedding, self).__init__()
        self.dim = dim
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
                8 [Abs]
            _ UnaryOp
                8 [deBruijn]
            9 [oos]
            10 [mask]
        """

    def forward(self, dense_batch: Tensor) -> Tensor:
        token_types, token_values, node_positions = dense_batch

        pad_mask = token_types != -1
        sos_mask = token_types == 0
        bop_mask = token_types == 1
        nop_mask = token_types == 2
        oos_mask = (token_types == 3) & (token_values == -1)
        db_mask = (token_types == 4)

        unique_paths, inverse = node_positions.unique(return_inverse=True)
        db_paths = torch.bucketize(token_values[db_mask], unique_paths)
        positional_encodings = self.path_encoder.forward(unique_paths)

        content_embeddings = torch.zeros(
            size=(*token_types.size(), self.dim),
            dtype=positional_encodings.dtype,
            device=token_types.device)
        content_embeddings[sos_mask] = self.embeddings.weight[0]
        content_embeddings[bop_mask] = self.embeddings.forward(token_values[bop_mask] + 1)
        content_embeddings[nop_mask] = self.embeddings.forward(token_values[nop_mask] + 5)
        content_embeddings[oos_mask] = self.embeddings.weight[10]
        content_embeddings[db_mask] = self.path_encoder.apply_mapping(
            maps=positional_encodings,
            embeddings=torch.ones_like(content_embeddings[db_mask]),
            index=db_paths)
        content_embeddings[pad_mask] = self.path_encoder.apply_mapping(
            maps=positional_encodings,
            embeddings=content_embeddings[pad_mask],
            index=inverse[pad_mask])
        content_embeddings = content_embeddings
        return content_embeddings
