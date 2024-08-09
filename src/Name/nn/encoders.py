from torch.nn import ModuleList, Module
from .batching import BatchedASTs

from .utils.modules import EncoderLayer, Rotary
from .embedding import TokenEmbedding

import torch
from torch import Tensor


class FileEncoder(Module):
    def __init__(self, num_layers: int, num_heads: int, dim: int, head_dim: int, dropout_rate: float):
        super(FileEncoder, self).__init__()
        self.dim = dim
        self.term_encoder = TermEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            dim=dim,
            head_dim=head_dim,
            dropout_rate=dropout_rate)
        self.embedding = TokenEmbedding(dim=dim, scope_dropout=dropout_rate)

    def forward(self,
                scope_asts: BatchedASTs,
                scope_sort: Tensor,
                hole_asts: BatchedASTs) -> tuple[Tensor, Tensor]:

        scope_features = self.embedding.forward(scope_asts.tokens.permute(2, 0, 1))
        hole_features = self.embedding.forward(hole_asts.tokens.permute(2, 0, 1))
        scope_features = scope_features
        hole_features = hole_features

        scope_reprs = torch.zeros(
            scope_asts.num_trees,
            self.dim,
            dtype=scope_features.dtype,
            device=scope_sort.device)

        for rank in scope_sort.unique(sorted=True):
            rank_mask = scope_sort.eq(rank)
            padding_mask = scope_asts.padding_mask[rank_mask]
            max_seq_len = padding_mask.sum(-1).max().item()

            scope_reprs[rank_mask] = self.term_encoder.forward(
                dense_features=scope_features[rank_mask][:, :max_seq_len],
                padding_mask=padding_mask[:, :max_seq_len],
                reference_mask=scope_asts.reference_mask[rank_mask][:, :max_seq_len],
                reference_ids=scope_asts.tokens[rank_mask][scope_asts.reference_mask[rank_mask]][:, 1],
                reference_storage=scope_reprs,
            )
        hole_reprs = self.term_encoder.forward(
            dense_features=hole_features,
            padding_mask=hole_asts.padding_mask,
            reference_mask=hole_asts.reference_mask,
            reference_ids=hole_asts.tokens[hole_asts.reference_mask][:, 1],
            reference_storage=scope_reprs,
        )
        return scope_reprs, hole_reprs


class TermEncoder(Module):
    def __init__(self, num_layers: int, num_heads: int, dim: int, head_dim: int, dropout_rate: float):
        super(TermEncoder, self).__init__()
        self.encoder = ModuleList([
            EncoderLayer(num_heads=num_heads,
                         dim=dim,
                         head_dim=head_dim,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)])
        self.pe = Rotary(4000, head_dim)

    def forward(self,
                dense_features: Tensor,
                padding_mask: Tensor,
                reference_mask: Tensor,
                reference_ids: Tensor,
                reference_storage: Tensor
                ) -> Tensor:
        dense_features[reference_mask] = reference_storage[reference_ids]
        pe = self.pe.forward(dense_features.size(1))

        layer: EncoderLayer
        for layer in self.encoder:
            dense_features = layer.forward(dense_features, pe, padding_mask)
        return dense_features[:, 0]
